import cv2  # used to draw and modity video frame 
from events.event_manager import format_timestamp # function that converts seconds into a readable time format
from zones.zone_manager import ZoneManager # Zone management class 

class Renderer:
    def __init__(self, display_width):  
        self.display_width = display_width

    def render(self,frame,camera_id, video_time,zones,persons,inside,safety,processing_fps,is_live=False,): # creating monitoring display

        if is_live:
            # Live CCTV: readable but not excessively large
            person_font_scale = 0.40
            person_thickness = 1

            panel_font_scale = 0.42
            panel_thickness = 1

            zone_font_scale = 0.36
            zone_thickness = 1

            instruction_font_scale = 0.30
            panel_width = 450

            line_spacing = 22
            zone_spacing = 20

        else:
            # Normal recorded videos: small text
            person_font_scale = 0.30
            person_thickness = 1

            panel_font_scale = 0.38
            panel_thickness = 1

            zone_font_scale = 0.32
            zone_thickness = 1

            instruction_font_scale = 0.26
            panel_width = 380

            line_spacing = 20
            zone_spacing = 18

        frame = ZoneManager.draw(frame, zones)
        for person in persons:
            x1 = person["x1"]
            y1 = person["y1"]
            x2 = person["x2"]
            y2 = person["y2"]

            zone = person["zone"]
            person_key = person.get("person_key",person["id"])
            track_id = person["id"]

            status = (f"ZONE {zone + 1}"
                if zone is not None
                else "OUTSIDE"
            )

            if person.get("locked", False):
                color = (0, 0, 255) if person.get("head_down") else (0, 255, 0)
                label = f"PERSON {person_key} | ID {track_id} | {status}"
            else:
                color = ((0, 255, 0)
                    if zone is not None
                    else (0, 165, 255)
                )

                label = f"ID {track_id} | {status}"
            if person.get("head_down"):
                label += (f" | HEAD DOWN {person.get('head_down_elapsed', 0.0):.0f}s")
            # Bounding box
            cv2.rectangle(frame,
                (x1, y1),(x2, y2),
                color,
                2,
            )
            # Person label
            cv2.putText(frame,label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                person_font_scale,
                color,
                person_thickness,
                cv2.LINE_AA,
            )

        panel_height = max(125,95 + len(zones) * zone_spacing)

        # Keep panel within the frame dimensions
        panel_width = min(panel_width,frame.shape[1] - 20)
        panel_height = min( panel_height,frame.shape[0] - 20)

        overlay = frame.copy()
        cv2.rectangle(overlay,
            (10, 10),
            (panel_width, panel_height),
            (0, 0, 0),
            -1,
        )
        frame = cv2.addWeighted(overlay,
            0.60,frame,
            0.40,0,
        )
        lines = [
            f"{camera_id} | Video: "
            f"{format_timestamp(video_time)}",
            f"Processing FPS: "
            f"{processing_fps:.1f}",
            f"People in zones: "
            f"{sum(safety.current_zone_counts)}",
            f"Work zones: {len(zones)}",
        ]

        y = 32

        for line in lines:
            cv2.putText(frame,line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                panel_font_scale,
                (255, 255, 255),
                panel_thickness,
                cv2.LINE_AA,
            )
            y += line_spacing

        for z in range(len(zones)):
            # Use the safety engine's authoritative deduplicated count. The
            # display snapshot can be one frame behind during tracker changes.
            count = safety.current_zone_counts[z]
            cooldown = safety.cooldown_until.get(z)
            if cooldown is not None and video_time < cooldown:
                remaining = cooldown - video_time
                text = (f"Zone {z + 1}: COOLDOWN {format_timestamp(remaining)}")
                color = (0, 165, 255)
            elif count > safety.allowed_people_per_zone[z] and safety.multiple_start.get(z) is not None:
                elapsed = video_time - safety.multiple_start[z]
                
                text = (
                    f"Zone {z + 1}: {count}/{safety.allowed_people_per_zone[z]} PEOPLE "
                    f"{elapsed:.0f}/"
                    f"{safety.multiple_limit_seconds}s"
                )
                color = (0, 255, 255)

            else:
                text = f"Zone {z + 1}: {count}/{safety.allowed_people_per_zone[z]}"
                color = (0, 255, 0)

            cv2.putText(frame,text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                zone_font_scale,
                color,
                zone_thickness,
                cv2.LINE_AA,
            )
            y += zone_spacing

        instruction_y = min(frame.shape[0] - 10, y + 5)

        cv2.putText(frame,
            "Click = select | Z = redraw | Q = quit",
            (20, instruction_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            instruction_font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if frame.shape[1] > self.display_width:
            scale = self.display_width / frame.shape[1]
        
            frame = cv2.resize( frame,
                (self.display_width,int(frame.shape[0] * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )
        return frame
    #Its job is to take video frame, detect people, work zones, and safety information, then draw everything on the frame so we can see monitoring results in a OpenCV window
