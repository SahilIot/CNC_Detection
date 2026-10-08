import copy # copies conf. values
import asyncio # Handles async streaming
import logging
import shutil # Delete camera directories
import threading # Protects shared worker state
import time 
from types import SimpleNamespace # Creates a conf. object dynamically
from types import ModuleType # helps executes modules while copying conf
import json
import httpx

import cv2 # encodes video frames as jpeg images
from fastapi.responses import Response
import torch # checks for GPU

# web server,API error responses, streaming response, static file hosting, and request validation
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi import HTTPException

import config
from camera.worker import CameraWorker
from detection.phone_detector import PhoneDetector
from detection.pose_detector import PoseDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager

app = FastAPI(title="CNC Detection API")
logger = logging.getLogger(__name__)
workers: dict[str, CameraWorker] = {}
workers_lock = threading.Lock()
camera_start_locks: dict[str, threading.Lock] = {}
screenshot_cleanup_stop = threading.Event()
screenshot_cleanup_thread = None
events = EventManager(config.outputs_dir)
zones = ZoneManager(config.outputs_dir, config.zone_margin_px)
app.mount("/outputs", StaticFiles(directory=config.outputs_dir), name="outputs")  # Security consideration only for trusted local network

class StartRequest(BaseModel): # validate incoming JSON (Pydantic)
    camera_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    max_persons: int = Field(default=1, ge=1, le=50)
    zone_limits: dict[str, int] = Field(default_factory=dict)
    zone_count: int = Field(default=1, ge=1, le=20)
    zones: list[list[list[float]]] = Field(default_factory=list)
    multiple_limit_seconds: int = Field(default=120, ge=1)
    absence_limit_seconds: int = Field(default=300, ge=1)


class EventDeleteRequest(BaseModel):
    event_ids: list[str] = Field(min_length=1)

class ZoneSaveRequest(BaseModel):
    zones: list[list[list[float]]] = Field(min_length=1, max_length=20)

def cleanup_old_screenshots():
    deleted = events.cleanup_old_screenshots(retention_days=15)
    if deleted:
        logger.info("Deleted %d screenshots older than 15 days", deleted)

def run_screenshot_cleanup():
    while not screenshot_cleanup_stop.wait(24 * 60 * 60):
        try:
            cleanup_old_screenshots()
        except OSError:
            logger.exception("Could not complete scheduled screenshot cleanup")

@app.on_event("startup")
def start_screenshot_cleanup():
    global screenshot_cleanup_thread
    cleanup_old_screenshots()
    screenshot_cleanup_stop.clear()
    screenshot_cleanup_thread = threading.Thread(
        target=run_screenshot_cleanup,
        name="screenshot-cleanup",
        daemon=True,
    )
    screenshot_cleanup_thread.start()

@app.on_event("shutdown") 
def shutdown_workers():
    screenshot_cleanup_stop.set()
    if screenshot_cleanup_thread is not None:
        screenshot_cleanup_thread.join(timeout=5)
    with workers_lock:
        active_workers = list(workers.values())
    for worker in active_workers:
        worker.running = False
        if worker.reader is not None:
            worker.reader.stop()

def choose_device():
    if torch.cuda.is_available():
        return 0, True
    return "cpu", False

def make_worker(request: StartRequest):
    if request.zones:
        actual_zone_count = len(request.zones)
        if not 1 <= actual_zone_count <= 20:
            raise RuntimeError("Number of zones must be between 1 and 20")
        request.zone_count = actual_zone_count
    elif request.zone_count < 1:
        raise RuntimeError("At least one work zone is required")
    device, half = choose_device()
    phone_detector = PhoneDetector(
        config.phone_model_path,
        device,half,
        config.phone_confidence,
        config.phone_image_size,
        config.phone_class_id,
    )
    pose_detector = PoseDetector(
        config.head_pose_model_path,
        device,
        half,
        config.head_pose_confidence,
        config.head_pose_image_size,
        config.head_pose_keypoint_confidence,
        config.head_pose_keypoint_delta,
    )
    worker_config = SimpleNamespace()
    for name in dir(config):
        if not name.startswith("_"):
            value = getattr(config, name)
            if not callable(value) and not isinstance(value, ModuleType):
                setattr(worker_config, name, copy.deepcopy(value))

    worker_config.multiple_limit_seconds = request.multiple_limit_seconds
    worker_config.multiple_person_limit_seconds = request.multiple_limit_seconds
    worker_config.absence_limit_seconds = request.absence_limit_seconds
    worker_config.allowed_people_per_zone = request.zone_limits
    worker_config.default_allowed_people = request.max_persons
    # An empty zones payload means this is a restart/settings update. In that
    # case CameraWorker loads the already-saved polygons and determines their
    # count from the file.
    worker_config.zone_count = request.zone_count if request.zones else None
    if request.zones:
        for zone in request.zones:
            if len(zone) < 3 or any(len(point) != 2 for point in zone):
                raise RuntimeError("Each zone must contain at least three points")
        zones.save_normalized(request.camera_id, request.zones)

    worker = CameraWorker(
        request.camera_id,
        request.source,
        worker_config,
        device,
        half,
        phone_detector,
        pose_detector,
        events,
        zones,
    )
    if not worker.prepare():
        raise RuntimeError("Camera worker could not prepare the camera or zones")
    worker.startup_ready.set()
    worker.start()
    return worker

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/events")
def list_events(limit: int = Query(default=100, ge=1, le=10000)):
    records = events.list_events(limit=limit)
    for record in records:
        screenshot = record.pop("screenshot")
        record["screenshot_url"] = (
            f"/outputs/cameras/{record['camera_id']}/screenshots/{screenshot}"
            if screenshot
            else None
        )
    return records

@app.post("/events/delete")
def delete_events(request: EventDeleteRequest):
    deleted = 0
    for event_id in request.event_ids:
        if events.delete_event(event_id):
            deleted += 1
    return {"deleted": deleted}

@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    if not events.delete_event(event_id):
        raise HTTPException(404, "Event not found")
    return {"deleted": True}

@app.post("/detection/start")
def start_detection(request: StartRequest):
    if request.zones:
        if len(request.zones) > 20:
            raise HTTPException(422, "Number of zones must not exceed 20")
        try:
            zones.validate_normalized(request.zones)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    with workers_lock:
        camera_lock = camera_start_locks.setdefault(
            request.camera_id, threading.Lock()
        )

    with camera_lock:
        with workers_lock:
            existing = workers.get(request.camera_id)

        if existing and existing.is_alive() and existing.running:
            if not request.zones:
                return {"camera_id": request.camera_id, "running": True}
            existing.running = False
            if existing.reader is not None:
                existing.reader.stop()

        if existing and existing.is_alive():
            existing.join(timeout=5)
            if existing.is_alive():
                raise HTTPException(
                    503,
                    f"Previous worker for {request.camera_id} has not stopped yet",
                )

        try:
            worker = make_worker(request)
        except Exception as exc:
            raise HTTPException(500, f"Failed to start detection: {exc}") from exc

        with workers_lock:
            workers[request.camera_id] = worker
    return {"camera_id": request.camera_id, "running": True}

@app.post("/detection/preview")
def detection_preview(request: StartRequest):
    from camera.reader import CameraReader
    reader = CameraReader(request.camera_id, request.source)
    try:
        reader.start()
        deadline = time.time() + 10
        frame = None
        while frame is None and time.time() < deadline:
            frame, _, _ = reader.get_latest()
            if frame is None:
                time.sleep(0.05)
        if frame is None:
            raise HTTPException(504, "Camera did not produce a preview frame")
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise HTTPException(500, "Could not encode camera preview")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    finally:
        reader.stop()


@app.post("/detection/{camera_id}/stop")
def stop_detection(camera_id: str):
    with workers_lock:
        camera_lock = camera_start_locks.setdefault(
            camera_id, threading.Lock()
        )

    with camera_lock:
        with workers_lock:
            worker = workers.get(camera_id)
        if worker is not None:
            worker.running = False
            if worker.reader is not None:
                worker.reader.stop()
    return {"camera_id": camera_id, "running": False}

@app.delete("/detection/{camera_id}")
def delete_detection(camera_id: str):
    stop_detection(camera_id)
    camera_dir = events.base / camera_id
    if camera_dir.exists():
        shutil.rmtree(camera_dir)
    return {"camera_id": camera_id, "deleted": True}

@app.get("/detection/{camera_id}/status")
def detection_status(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)
        if worker is None:
            return {"camera_id": camera_id, "running": False, "people_inside": 0}
        safety = worker.safety
        zones = [
            {
                "zone": index + 1,
                "present": int(safety.current_zone_counts[index]),
                "allowed": safety.allowed_people_per_zone[index],
                "violation": safety.violation_status(index),
            }
            for index in range(len(safety.current_zone_counts))
        ]
        return {
            "camera_id": camera_id,
            "running": bool(worker.is_alive() and worker.running),
            "people_inside": sum(zone["present"] for zone in zones),
            "zones": zones,
            "last_error": str(worker.error) if worker.error else None,
        }

@app.get("/detection/{camera_id}/zones")
def detection_zones(camera_id: str):
    zone_file=(config.outputs_dir/"camera_zones"/f"{camera_id}_zones.json")
    if not zone_file.exists():
        return {"saved":False , "count":0,"zones":[],}
    try:
        with open(zone_file,"r",encoding="utf-8")as f:
            data=json.load(f)
        saved_zones=data.get("zones",[])
        if not isinstance(saved_zones,list):
            raise ValueError("Saved zones must be a list")
        return {"saved": bool(saved_zones),"count":len(saved_zones),"zones":saved_zones,}

    except (json.JSONDecodeError,ValueError) as exc:
        raise HTTPException( 
            status_code=500,
            detail=f"Could not read saved zones:{exc}",
        ) 

@app.post("/detection/{camera_id}/zones")
def save_detection_zones(camera_id: str, request: ZoneSaveRequest):
    try:
        zones.save_normalized(camera_id, request.zones)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"saved": True, "count": len(request.zones)}

@app.get("/detection/{camera_id}/stream")
async def detection_stream(camera_id: str, request: Request):
    with workers_lock:
        worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(404, "Detection camera is not running")

    async def frames():
        while worker.running and worker.reader is not None:
            if await request.is_disconnected():
                break
            frame = worker.get_display()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"Could not encode stream frame for {camera_id}")
            
            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
            await asyncio.sleep(0.04)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )