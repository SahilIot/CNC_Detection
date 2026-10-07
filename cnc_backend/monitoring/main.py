import threading
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from camera.worker import CameraWorker
from detection.phone_detector import PhoneDetector
from detection.pose_detector import PoseDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager


app = FastAPI(title="CNC Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    camera_id: str
    source: str
    max_persons: int = 1
    multiple_limit_seconds: int = 120
    absence_limit_seconds: int = 300


workers: Dict[str, CameraWorker] = {}
workers_lock = threading.Lock()
events = EventManager(config.outputs_dir)
zones = ZoneManager(config.outputs_dir, config.zone_margin_px)

def choose_device():
    if torch.cuda.is_available():
        return 0, True
    return "cpu", False

def create_worker(request: StartRequest):
    device, half = choose_device()

    phone_detector = PhoneDetector(
        config.phone_model_path,
        device,
        half,
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

    # Copy the configured values and override them for this dashboard machine.
    worker_config = config
    worker_config.multiple_person_limit_seconds = (
        request.multiple_limit_seconds
    )
    worker_config.absence_limit_seconds = request.absence_limit_seconds

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
        raise RuntimeError("CameraWorker could not prepare the camera")

    worker.startup_ready.set()
    worker.start()

    return worker


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detection/start")
def start_detection(request: StartRequest):
    with workers_lock:
        existing = workers.get(request.camera_id)

        if existing and existing.is_alive() and existing.running:
            return {
                "camera_id": request.camera_id,
                "running": True,
                "message": "Already running",
            }

        try:
            worker = create_worker(request)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start detection: {exc}",
            ) from exc

        workers[request.camera_id] = worker

    return {
        "camera_id": request.camera_id,
        "running": True,
    }

@app.post("/detection/{camera_id}/stop")
def stop_detection(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)

        if worker is None:
            return {
                "camera_id": camera_id,
                "running": False,
            }

        worker.running = False

        if worker.reader is not None:
            worker.reader.stop()

    return {
        "camera_id": camera_id,
        "running": False,
    }

@app.get("/detection/{camera_id}/status")
def detection_status(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)

        if worker is None:
            return {
                "camera_id": camera_id,
                "running": False,
                "people_inside": 0,
            }

        safety = getattr(worker, "safety", None)
        people_inside = int(
            getattr(safety, "number_inside", 0)
        ) if safety else 0

        return {
            "camera_id": camera_id,
            "running": bool(worker.is_alive() and worker.running),
            "people_inside": people_inside,
        }