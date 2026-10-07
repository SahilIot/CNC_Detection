from datetime import datetime

from tracking.track_state import (deduplicate_persons,find_previous_track,match_locked_person,)

class SafetyRules:
    def __init__(self,camera_id,zones_count,event_manager,config,):
        self.camera_id = camera_id
        self.event_manager = event_manager

        self.multiple_limit = config.multiple_person_limit_seconds

        self.multiple_limit_seconds = config.multiple_person_limit_seconds
        configured_limits = getattr(config, "allowed_people_per_zone", 1)
        default_limit = max(0, int(getattr(config, "default_allowed_people", 1)))
        if isinstance(configured_limits, dict):
            self.allowed_people_per_zone = [
                max(0, int(configured_limits.get(
                    f"zone_{index + 1}", default_limit
                )))
                for index in range(zones_count)
            ]
        else:
            self.allowed_people_per_zone = [
                max(0, int(configured_limits))
                for _ in range(zones_count)
            ]

        self.cooldown_seconds = config.multiple_person_cooldown_seconds

        self.absence_limit = config.absence_limit_seconds
        self.head_down_limit = getattr(config, "head_down_violation_seconds", 120.0)

        self.track_grace = config.track_grace_seconds

        self.inside_grace = config.inside_grace_seconds

        self.switch_max = config.track_id_switch_max_seconds

        self.switch_center = config.track_id_switch_center_ratio

        self.switch_iou = config.track_id_switch_iou_threshold
        self.duplicate_iou = config.duplicate_iou_threshold
        self.duplicate_center = config.duplicate_center_ratio
        self.track_states = {}
        self.multiple_start = {
            i: None
            for i in range(zones_count)
        }
        self.cooldown_until = {
            i: None
            for i in range(zones_count)
        }
        self.multiple_logged = {
            i: False
            for i in range(zones_count)
        }
        self.zone_locks = {
            i: []
            for i in range(zones_count)
        }
        self.next_person_key = 1
        self.number_inside = 0
        self.current_zone_counts = [0 for _ in range(zones_count)]
        self.violation_started_at = {i: None for i in range(zones_count)}
        self.violation_ended_at = {i: None for i in range(zones_count)}
        self.violation_open = {i: False for i in range(zones_count)}
        self.violation_track_ids = {i: set() for i in range(zones_count)}

    def _record_absence_if_confirmed(self, lock, zone, current_time, events):
        away_start = lock.get("away_start")
        if away_start is None or current_time - away_start < self.absence_limit:
            return
        started_at = lock.get("away_started_at") or datetime.now().astimezone()
        ended_at = datetime.now().astimezone()
        duration = current_time - away_start
        details = (
            f"Person {lock['person_key']} absent from zone {zone + 1}: "
            f"{started_at.strftime('%I:%M:%S %p')} - "
            f"{ended_at.strftime('%I:%M:%S %p')} "
            f"({duration:.1f} seconds)"
        )
        timestamp = self.event_manager.log(
            self.camera_id, current_time, "PERSON_ABSENCE_INTERVAL",
            lock["track_id"], details,
        )
        events.append({
            "timestamp": timestamp,
            "event_type": "PERSON_ABSENCE_INTERVAL",
            "track_ids": {lock["track_id"]},
            "details": details,
            "zone": zone,
        })
        lock["away_logged"] = True

    def update(self,persons,current_time,zone_count,):
        events = []

        # Make sure zone structures exist.
        for z in range(zone_count):
            self.zone_locks.setdefault(z, [])
            self.multiple_start.setdefault(z, None)
            self.cooldown_until.setdefault(z, None)
            self.multiple_logged.setdefault(z, False)
            self.violation_started_at.setdefault(z, None)
            self.violation_ended_at.setdefault(z, None)
            self.violation_open.setdefault(z, False)
            self.violation_track_ids.setdefault(z, set())
            if z >= len(self.allowed_people_per_zone):
                self.allowed_people_per_zone.append(1)

        persons, suppressed = deduplicate_persons(persons,
            self.duplicate_iou,self.duplicate_center,
        )
        for track_id in suppressed:
            self.track_states.pop(track_id,None,)
        active_ids = {p["id"]
            for p in persons
        }

        for person in persons:
            track_id = person["id"]
            zone = person["zone"]
            if track_id not in self.track_states:
                old_id = find_previous_track(self.track_states,
                    person,current_time,
                    active_ids,
                    self.switch_max,
                    self.switch_center,
                    self.switch_iou,
                )
                if old_id is not None:
                    self.track_states[track_id] = (self.track_states
                        .pop(old_id).copy()
                    )
                    print(f"[{self.camera_id}] Track ID changed {old_id} -> {track_id}")
                else:
                    self.track_states[track_id] = {
                        "last_seen": current_time,
                        "last_inside": (
                            current_time
                            if zone is not None
                            else None
                        ),
                        "last_zone": zone,
                        "absence_logged": False,
                        "phone_logged": False,
                        "head_down_start": None,
                        "head_down_logged": False,
                        "last_box": (
                            person["x1"],person["y1"],
                            person["x2"],person["y2"],
                        ),
                    }
            state = self.track_states[track_id]
            if current_time - state.get("last_seen", current_time) > self.track_grace:
                state["head_down_start"] = None
                state["head_down_logged"] = False
            state["last_seen"] = current_time
            state["last_box"] = (
                person["x1"],person["y1"],
                person["x2"],person["y2"],
            )
            if zone is not None:
                state["last_inside"] = current_time
                state["last_zone"] = zone
                state["absence_logged"] = False
            state.setdefault("head_down_start", None)
            state.setdefault("head_down_logged", False)
            if zone is None or not person.get("head_down", False):
                state["head_down_start"] = None
                state["head_down_logged"] = False
            else:
                if state["head_down_start"] is None:
                    state["head_down_start"] = current_time
                elapsed = current_time - state["head_down_start"]
                limit = self.head_down_limit
                if elapsed >= limit and not state["head_down_logged"]:
                    state["head_down_logged"] = True
                    details = (
                        f"Person sustained a head-down posture in zone "
                        f"{zone + 1} for {elapsed:.1f} seconds "
                        f"(limit: {limit:.1f} seconds)"
                    )
                    timestamp = self.event_manager.log(
                        self.camera_id,
                        current_time,
                        "HEAD_DOWN_VIOLATION",
                        track_id,
                        details,
                    )
                    events.append({
                        "timestamp": timestamp,
                        "event_type": "HEAD_DOWN_VIOLATION",
                        "track_ids": {track_id},
                        "details": details,
                        "zone": zone,
                    })

        for track_id, state in self.track_states.items():
            missing_for = current_time - state.get("last_seen", current_time)
            if track_id not in active_ids and missing_for > self.track_grace:
                state["head_down_start"] = None
                state["head_down_logged"] = False

        inside = {z: set()
            for z in range(zone_count)
        }
        # A tracker ID can only belong to one physical lock
        # during this frame.
        used_ids = set()
        for z in range(zone_count):
            locks = self.zone_locks[z]
            for lock in locks:

                matched = match_locked_person(
                    lock["box"],
                    persons,
                    used_ids=used_ids,
                    center_ratio=self.switch_center,
                    iou_threshold=self.switch_iou,
                    preferred_track_id=lock["track_id"],
                )
                if matched is not None:
                    old_track_id = lock["track_id"]
                    new_track_id = matched["id"]
                    if old_track_id != new_track_id:
                        print(f"[{self.camera_id}] Zone {z + 1}: "
                            f"Physical person retained | ID {old_track_id} -> {new_track_id} "
                            f"| Person Key {lock['person_key']}"
                        )
                    lock["track_id"] = new_track_id
                    lock["box"] = (
                        matched["x1"],matched["y1"],
                        matched["x2"],matched["y2"],
                    )
                    lock["last_seen"] = current_time
                    used_ids.add(new_track_id)
                    if matched["zone"] == z:
                        self._record_absence_if_confirmed(
                            lock, z, current_time, events
                        )
                        # Person is PRESENT.
                        lock["inside"] = True
                        # Cancel any previous away timer.
                        lock["away_start"] = None
                        lock["away_started_at"] = None
                        lock["missing_started_at"] = None
                        # Allow a future departure to create
                        # a new event.
                        lock["away_logged"] = False
                        lock["missing_start"] = None
                        inside[z].add(new_track_id)
                        continue
                    else:
                        if lock["inside"]:
                            lock["inside"] = False
                            lock["away_start"] = current_time
                            lock["away_started_at"] = datetime.now().astimezone()
                            lock["away_logged"] = False
                            print(f"[{self.camera_id}] Zone {z + 1}: "
                                f"PERSON LEFT ZONE | Person Key "
                                f"{lock['person_key']} | Track ID "
                                f"{new_track_id}"
                            )
                        # Person is outside.
                        continue
                else:
                    if lock["inside"]:
                        if lock.get("missing_start") is None:
                            lock["missing_start"] = current_time
                            lock["missing_started_at"] = datetime.now().astimezone()
                        if current_time - lock["missing_start"] >= self.absence_limit:
                            lock["inside"] = False
                            # Keep the original disappearance time so the
                            # configured absence period is not counted twice.
                            lock["away_start"] = lock["missing_start"]
                            lock["away_started_at"] = lock.get(
                                "missing_started_at",
                                datetime.now().astimezone(),
                            )
                            lock["away_logged"] = False
                        elif current_time - lock["missing_start"] < self.inside_grace:
                            inside[z].add(lock["track_id"])
                            continue

        for person in persons:
            track_id = person["id"]
            # Only create a lock when the person is actually
            # inside a zone.
            zone = person["zone"]
            if zone is None:
                continue
            # Already assigned to an existing physical lock.
            if track_id in used_ids:
                continue

            matched_lock = None

            for lock in self.zone_locks[zone]:
                if lock["track_id"] in used_ids:
                    continue
                # Reuse a physical-person lock even when the tracker assigns a
                # new ID after a missed/overlapping detection. Otherwise the
                # old lock and the new lock are counted as two people.
                matched = match_locked_person(
                    lock["box"], [person],
                    used_ids=set(),
                    center_ratio=max(self.switch_center, 2.0),
                    iou_threshold=0.05,
                )
                if matched is not None:
                    matched_lock = lock
                    break

            if matched_lock is not None:
                lock = matched_lock
                old_track_id = lock["track_id"]
                lock["track_id"] = track_id
                lock["box"] = (
                    person["x1"],person["y1"],
                    person["x2"],person["y2"],
                )

                lock["last_seen"] = current_time
                lock["inside"] = True
                lock["away_start"] = None
                lock["away_started_at"] = None
                lock["missing_started_at"] = None
                lock["missing_start"] = None

                lock["away_logged"] = False
                used_ids.add(track_id)
                if old_track_id != track_id:
                    inside[zone].discard(old_track_id)
                inside[zone].add(track_id)

                print(f"[{self.camera_id}] Zone {zone + 1}: Person lock retained | Person Key: "
                    f"{lock['person_key']} | ID {old_track_id} -> {track_id}")
                continue
            person_key = self.next_person_key
            self.next_person_key += 1
            lock = {
                "person_key": person_key,
                "track_id": track_id,
                "box": (
                    person["x1"],person["y1"],
                    person["x2"],person["y2"],
                ),
                "last_seen": current_time,
                "away_start": None,
                "away_started_at": None,
                "missing_started_at": None,
                "away_logged": False,
                "missing_start": None,
                "inside": True,
            }

            self.zone_locks[zone].append(lock)
            used_ids.add(track_id)
            inside[zone].add(track_id)
            print(f"[{self.camera_id}] Zone {zone + 1}: "
                f"Person locked | Person Key: {person_key} "
                f"| Track ID: {track_id}"
            )

        for z in range(zone_count):
            count = len(inside[z])
            cooldown = self.cooldown_until[z]
            if cooldown is not None and current_time < cooldown:

                self.multiple_start[z] = None
                continue

            if cooldown is not None and current_time >= cooldown:
                self.cooldown_until[z] = None
                self.multiple_logged[z] = False

            allowed = self.allowed_people_per_zone[z]
            if count > allowed:
                if self.multiple_start[z] is None:

                    self.multiple_start[z] = current_time
                    self.multiple_logged[z] = False
                    print(f"[{self.camera_id}] Zone {z + 1}: "
                        f"Multiple people detected. Timer started at "
                        f"{self._ts(current_time)}"
                    )
                elapsed = current_time- self.multiple_start[z]
                if elapsed >= self.multiple_limit and not self.multiple_logged[z]:
                    self.multiple_logged[z] = True
                    self.violation_started_at[z] = datetime.now().astimezone()
                    self.violation_open[z] = True
                    self.violation_track_ids[z] = set(inside[z])

                    event_type = "PEOPLE_OVER_ALLOWED_LIMIT"
                    details = (f"{count} people inside zone {z + 1} "
                        f"(limit: {allowed}) "
                        f"for {elapsed:.1f} seconds; "
                        f"30 minutes cooldown started"
                    )
            else:
                if self.violation_open[z]:
                    ended_at = datetime.now().astimezone()
                    started_at = self.violation_started_at[z]
                    ids = set(self.violation_track_ids[z])
                    text = "|".join(str(i) for i in sorted(ids))
                    interval = (
                        f"{started_at.strftime('%I:%M:%S %p')} - "
                        f"{ended_at.strftime('%I:%M:%S %p')}"
                    )
                    event_type = "PEOPLE_OVER_ALLOWED_LIMIT"
                    details = (
                        f"Zone {z + 1}: {interval}; occupancy remained over "
                        f"the limit of {allowed}"
                    )
                    timestamp = self.event_manager.log(
                        self.camera_id, current_time, event_type, text, details
                    )
                    events.append({
                        "timestamp": timestamp,
                        "event_type": event_type,
                        "track_ids": ids,
                        "details": details,
                        "zone": z,
                    })
                    self.violation_ended_at[z] = ended_at
                    self.violation_open[z] = False
                self.cooldown_until[z] = None
                self.multiple_start[z] = None
                self.multiple_logged[z] = False

        self.number_inside = sum(len(v)
            for v in inside.values())
        self.current_zone_counts = [len(inside[z]) for z in range(zone_count)]
        return persons, inside, events

    def violation_status(self, zone):
        started = self.violation_started_at.get(zone)
        return {
            "active": bool(self.violation_open.get(zone)),
            "started_at": started.isoformat() if started else None,
            "started_display": started.strftime("%I:%M:%S %p") if started else None,
        }

    def get_display_persons(self, current_time):
        display_persons = []
        for zone, locks in self.zone_locks.items():
            for lock in locks:
                if not lock["inside"]:
                    continue
                # Safety state stays PRESENT even while YOLO is
                # momentarily missing this person (see the "continue"
                # in update() for matched is None). But we don't want
                # to keep drawing a box that hasn't been refreshed in
                # a while, so this is a display-only cutoff -- it does
                # NOT touch lock["inside"] or start the away timer.
                stale_for = current_time - lock.get("last_seen", current_time)
                if stale_for > self.inside_grace:
                    continue
                x1, y1, x2, y2 = lock["box"]
                display_persons.append({"id": lock["track_id"],
                    "person_key": lock["person_key"],
                    "x1": x1,"y1": y1,
                    "x2": x2,"y2": y2,
                    "zone": zone,
                    "confidence": 1.0,
                    "locked": True,
                })

        return display_persons

    @staticmethod
    def _ts(seconds):
        return (
            f"{int(seconds // 3600):02d}:"
            f"{int((seconds % 3600) // 60):02d}:"
            f"{int(seconds % 60):02d}"
        )

    def phone_allowed(self, track_id):
        return not (self.track_states
            .get(track_id, {})
            .get("phone_logged", False)
        )
    def mark_phone(self, track_id):
        if track_id in self.track_states:
            self.track_states[track_id]["phone_logged"] = True

    def reset_zones(self, zone_count):
        self.multiple_start = {i: None
            for i in range(zone_count)
        }
        self.cooldown_until = {i: None
            for i in range(zone_count)
        }
        self.multiple_logged = {i: False
            for i in range(zone_count)
        }
        self.violation_started_at = {i: None for i in range(zone_count)}
        self.violation_ended_at = {i: None for i in range(zone_count)}
        self.violation_open = {i: False for i in range(zone_count)}
        self.violation_track_ids = {i: set() for i in range(zone_count)}
        self.zone_locks = {i: []
            for i in range(zone_count)
        }
        self.next_person_key = 1