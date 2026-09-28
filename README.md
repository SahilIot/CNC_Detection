# CNC Detection and Monitoring Dashboard

An on-premises, multi-camera monitoring application for CNC work areas. It
reads RTSP streams or local video files, detects and tracks people, checks for
phones in configured work zones, applies per-zone occupancy rules, records
events and screenshots, and serves a browser-based dashboard.

The application has two Python/FastAPI services:

1. **Detection backend** (`cnc_backend`) opens video sources, runs YOLO
   detection and tracking, applies safety rules, stores zone polygons and
   event files, and serves annotated MJPEG streams.
2. **Dashboard frontend** (`cnc_frontend`) stores machine settings in SQLite,
   provides the browser interface, and calls the backend over HTTP.

The backend and dashboard can run on the same Windows server. Keeping them on
the same host is the simplest setup; the dashboard can be exposed to a trusted
LAN while the backend remains bound to localhost.

> **Safety note:** This software uses computer-vision inference and is not a
> certified safety controller. Do not use it as a substitute for required
> machine guarding, emergency stops, or other certified safety systems.

## Contents

- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Install on Windows](#install-on-windows)
- [Configure the backend](#configure-the-backend)
- [Connect cameras or an NVR](#connect-cameras-or-an-nvr)
- [Run locally](#run-locally)
- [Run on a LAN server](#run-on-a-lan-server)
- [Dashboard workflow](#dashboard-workflow)
- [Zones and occupancy limits](#zones-and-occupancy-limits)
- [Detection and safety behavior](#detection-and-safety-behavior)
- [Events, screenshots, and persistent data](#events-screenshots-and-persistent-data)
- [HTTP API](#http-api)
- [Troubleshooting](#troubleshooting)
- [Operational and security notes](#operational-and-security-notes)

## Project layout

```text
CNC_Detection/
├── requirements.txt
├── README.md
├── cnc_backend/
│   ├── camera/                  # RTSP/file reader and per-camera worker
│   ├── detection/               # YOLO person tracking and phone detection
│   ├── display/                 # Annotated MJPEG frame rendering
│   ├── events/                  # CSV event log and screenshot management
│   ├── monitoring/
│   │   └── api.py               # Detection backend FastAPI application
│   ├── safety/                  # Occupancy, absence, and violation rules
│   ├── tracking/                # Duplicate suppression and track matching
│   ├── tracking_configs/        # Tracker configuration
│   ├── zones/                   # Zone polygon storage and geometry
│   ├── data/
│   │   ├── outputs/             # Events, screenshots, and saved zone files
│   │   └── videos/              # Optional local test videos
│   ├── models/                  # YOLO weights (model files are not in Git)
│   └── config.py                # Backend settings and environment loading
└── cnc_frontend/
    ├── database/
    │   ├── db.py                # SQLite persistence helpers
    │   └── cnc.sqlite3          # Created/updated at runtime
    ├── monitoring/adapter.py    # HTTP client for the backend
    ├── static/                  # Dashboard CSS and static assets
    ├── templates/               # Dashboard and machine pages
    └── web/app.py               # Dashboard FastAPI application
```

## Requirements

- Windows, macOS, or Linux. The commands below use Windows PowerShell.
- Python 3.10–3.12 is recommended. Python 3.14 is not recommended for the
  computer-vision dependency stack.
- Network access from the backend host to each camera/NVR RTSP stream.
- YOLO model weights compatible with Ultralytics. The repository's `.pt`
  model files are ignored by Git; provide the desired weights locally.
- NVIDIA GPU and a compatible CUDA/PyTorch setup are optional. The backend
  selects CPU when CUDA is unavailable.

Dependencies are listed in [`requirements.txt`](requirements.txt). If
installation or model loading fails, confirm the Python version and install
the PyTorch/Ultralytics build appropriate for the server hardware.

## Install on Windows

From the repository root:

```powershell
Set-Location C:\Users\Sahil\Downloads\CNC_Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not allow virtual-environment activation, use its Python
executable directly:

```powershell
Set-Location C:\Users\Sahil\Downloads\CNC_Detection
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

There may also be a separate `cnc_frontend\.venv` in a developer checkout.
Do not mix environments: each service must be run using a Python environment
that contains the project's dependencies. A single root `.venv` is the
simplest arrangement for both services.

Place the model weights on the server, for example:

```text
cnc_backend\models\yolo11n.pt
```

Set `MODEL_PATH` to the chosen person model. If `PHONE_MODEL_PATH` is not
provided, the phone detector uses the same model file. The phone model must
contain the configured phone class (COCO class ID `67` by default).

## Configure the backend

The backend loads the repository-root `.env` first, then
`cnc_backend\.env` if present; the backend-specific file overrides duplicate
values. Create a local `.env` rather than putting camera passwords in source
control. `.env` files and model weights are excluded by `.gitignore`.

Common environment variables read by `cnc_backend\config.py`:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PATH` | Person-detection YOLO weights | `cnc_backend\models\yolo11n.pt` |
| `PHONE_MODEL_PATH` | Phone-detection YOLO weights | Value of `MODEL_PATH` |
| `PERSON_CONFIDENCE` | Minimum person detection confidence | `0.40` |
| `PHONE_CONFIDENCE` | Minimum phone detection confidence | `0.50` |
| `PHONE_DETECTION_INTERVAL` | Run phone detection every N frames | `3` |
| `PHONE_CLASS_ID` | Phone class ID in the phone model | `67` |
| `PERSON_IMAGE_SIZE` | Person model inference image size | `640` |
| `PHONE_IMAGE_SIZE` | Phone model inference image size | `512` |
| `PERSON_USE_AUGMENT` | Enable person model inference augmentation (`true`/`false`) | `false` |
| `PERSON_TRACKER_CONFIG` | Ultralytics tracker YAML or path | `botsort.yaml` |
| `TRACK_GRACE_SECONDS` | Track matching grace period | `1.5` |
| `INSIDE_GRACE_SECONDS` | Keep a briefly missed person counted | `2.5` |

Camera sources for the dashboard application are registered as machines in
the dashboard and stored in SQLite. Although `config.py` still reads the
legacy `CAMERA_01` ... `CAMERA_04` and `CAMERA_SOURCE_01` ...
`CAMERA_SOURCE_04` variables, those values do not register dashboard
machines or automatically start monitoring.

Example local configuration (replace placeholders with values supplied by
your camera/NVR administrator):

```dotenv
MODEL_PATH=models/yolo11n.pt
PHONE_MODEL_PATH=models/yolo11n.pt
CAMERA_01=rtsp://<user>:<password>@<nvr-address>:554/<vendor-channel-path>
PERSON_CONFIDENCE=0.40
PHONE_CONFIDENCE=0.50
PHONE_DETECTION_INTERVAL=3
PHONE_CLASS_ID=67
PERSON_IMAGE_SIZE=640
PHONE_IMAGE_SIZE=512
```

The dashboard's machine settings are stored in its SQLite database and sent
to the backend when monitoring starts. The zone occupancy delay defaults to
120 seconds and the absence delay defaults to 300 seconds; both can be
adjusted per machine in the dashboard. These values are not controlled by
`CAMERA_XX` settings.

## Connect cameras or an NVR

The backend host—not the user's browser—must be able to open the RTSP source.
Confirm that:

1. The server can reach the camera or NVR on the network.
2. RTSP is enabled and the supplied account has permission to view the
   intended stream.
3. The complete RTSP URL works from the server (test it with VLC or another
   RTSP-capable player installed on the server).
4. The URL uses the correct vendor-specific channel path, port, and main or
   substream. NVR URL formats differ by manufacturer and model.

The dashboard's machine form currently accepts RTSP/RTSPS URLs. The backend
reader also supports local video files when a source path is supplied
directly to the backend API, which can be useful for development.

### Camera and NVR IP changes

The dashboard stores an RTSP URL for each machine. **The application does
not discover cameras, track a camera's changed IP, or update stored RTSP URLs
automatically.**

Recommended setup:

- Give the NVR a static address or DHCP reservation.
- When possible, connect the dashboard to an NVR channel URL rather than a
  camera's direct IP. This only remains stable if the NVR keeps the camera
  assigned to the same channel and exposes it through that URL.
- If using direct camera streams, set a static address or DHCP reservation
  for each camera.
- If an address or channel path changes, update that machine's RTSP URL in
  the application/database and restart monitoring.

The exact NVR RTSP path depends on its brand, model, channel numbering, and
stream selection. Obtain it from the NVR documentation or administrator; do
not assume one manufacturer's URL pattern will work on another.

The dashboard can request camera credentials after an RTSP authentication
failure. Credentials are stored in the frontend SQLite database keyed by
RTSP host and are reused for matching hosts. Protect that database like a
secret because the credentials are stored locally in it.

## Run locally

Open two PowerShell terminals from the repository. Start the backend first.

**Terminal 1 — backend:**

```powershell
Set-Location C:\Users\Sahil\Downloads\CNC_Detection\cnc_backend
..\.venv\Scripts\python.exe -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
```

Alternatively, activate the root `.venv` and run:

```powershell
python -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
```

Check backend health at <http://127.0.0.1:9000/health>.

**Terminal 2 — dashboard:**

```powershell
Set-Location C:\Users\Sahil\Downloads\CNC_Detection\cnc_frontend
..\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Or activate the root `.venv` first and use
`python -m uvicorn web.app:app --host 127.0.0.1 --port 8000`.

Open <http://127.0.0.1:8000>.

The frontend uses `http://127.0.0.1:9000` as its backend by default. If the
backend runs on another machine, set `CNC_BACKEND_URL` in the frontend
terminal to that backend's reachable HTTP address before starting Uvicorn:

```powershell
$env:CNC_BACKEND_URL = "http://<backend-server-ip>:9000"
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Use the correct ASGI import paths and working directories:

| Service | Working directory | Uvicorn application | Default port |
|---|---|---|---:|
| Backend | `cnc_backend` | `monitoring.api:app` | 9000 |
| Dashboard | `cnc_frontend` | `web.app:app` | 8000 |

Do not use `uvicorn api:app`; neither service has a top-level `api.py`.

## Run on a LAN server

For a single Windows server connected to the cameras/NVR and serving browser
clients on the local network:

1. Install Python dependencies and model weights on the server.
2. Configure and test the NVR/camera RTSP URL from that server.
3. Start the backend bound to localhost:

   ```powershell
   Set-Location C:\Users\Sahil\Downloads\CNC_Detection\cnc_backend
   ..\.venv\Scripts\python.exe -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
   ```

4. Start the dashboard listening on the server's network interfaces:

   ```powershell
   Set-Location C:\Users\Sahil\Downloads\CNC_Detection\cnc_frontend
   ..\.venv\Scripts\python.exe -m uvicorn web.app:app --host 0.0.0.0 --port 8000
   ```

5. Allow inbound TCP port `8000` through Windows Firewall only from trusted
   LAN clients. Keep backend port `9000` private when both services run on
   the same server.
6. On a client computer, open `http://<server-LAN-IP>:8000`.

For a separate backend host, bind its backend service to an interface
reachable by the frontend server, restrict port `9000` to that frontend
server/network, and configure `CNC_BACKEND_URL` to the backend's address.
Do not expose either service directly to the public internet.

Uvicorn commands shown here run in the foreground and stop when the terminal
or process closes. For unattended operation, configure both commands to run
at server startup using an approved Windows service manager or Task Scheduler
under a dedicated account. Set the working directory, Python executable,
environment variables, log handling, and restart behavior explicitly. Ensure
the server has reliable power/network and that the configured account can
read model files and write to the database/output directories.

## Dashboard workflow

1. Open the dashboard and choose **Add Machine**.
2. Enter a machine name, RTSP URL, zone count, per-zone allowed-person
   limits, and timing settings.
3. Save the machine. Click **Start** for its row.
4. In the zone editor, draw each work area on the camera preview and click
   **Finish zone** for each polygon. Draw the configured number of zones and
   choose **Start monitoring**.
5. Use the machine card's settings form to adjust each zone's allowed-person
   count or the per-machine event/absence delays. **Update settings** restarts
   a running monitor with the new configuration.
6. Use **Stop** to stop monitoring but retain the machine, settings, polygons,
   and history. Use **Delete** to remove the machine and its backend output
   data.
7. Use the preview link to open the per-machine monitor page. Recent events
   and saved screenshots are available from the dashboard.

Zone polygons are saved per generated camera ID (`camera_01`, `camera_02`,
etc.). Re-draw or update zones through the dashboard if the camera view or
work area changes.

## Zones and occupancy limits

Each zone has an independently configurable **maximum allowed number of
people from 0 to 50**. The machine-wide **Maximum people** value is a fallback
for zones without an explicit setting; it is not a lower bound on a zone's
limit.

- **Allowed = 0:** any person detected inside the zone is over the limit.
- **Allowed = 1 or more:** a violation timer starts when the deduplicated
  count is greater than that zone's limit.
- The dashboard's **Event delay (seconds)** is the over-limit delay. It
  defaults to 120 seconds (2 minutes).
- While over the limit, the dashboard status shows the violation start time
  after the delay has elapsed. When occupancy returns to the allowed count,
  the violation interval is closed and recorded.
- The timer is based on processed video time; displayed event timestamps use
  the backend computer's local clock.

For example, if a zone's limit is zero and a person enters at 09:00, the
two-minute timer starts. If the person remains in the zone, the violation
becomes active at about 09:02. If the person leaves at 09:05, the recorded
violation interval runs from the activation time (about 09:02) until the
violation clears (about 09:05).

## Detection and safety behavior

- Person detection uses an Ultralytics YOLO model and persistent tracking.
- The worker assigns a person to a zone using the bottom-center point of the
  person's bounding box (an estimate of foot position), not general box
  overlap.
- Duplicate detections are suppressed before zone counts are calculated.
- Track changes can be matched to existing physical-person locks to reduce
  double counting.
- A temporarily missing person remains counted during
  `INSIDE_GRACE_SECONDS` (2.5 seconds by default).
- **Over-limit timer:** starts when zone count exceeds the allowed count. It
  defaults to 120 seconds; the machine's **Event delay** setting controls it.
- **Absence timer:** starts when a tracked person leaves a zone, or when a
  previously present detection remains missing. It defaults to 300 seconds
  (5 minutes); the machine's **Absence delay** setting controls it. This is
  separate from the over-limit timer. It does not replace or delay a
  zero-person-zone violation.
- A `PERSON_ABSENCE_INTERVAL` event is recorded if the person is matched back
  to the zone after the configured absence period. The five-minute setting
  does not itself activate the over-limit violation.
- Phone detection runs on person crops at the configured frame interval.
  A phone detected inside a work zone creates a `PHONE_DETECTED` event.
- Violation and phone/absence event screenshots are saved with the event
  data. A violation interval is recorded when the over-limit condition
  clears; the dashboard can show the active violation's start time while it
  is ongoing.
- The computer's local wall clock is used for event timestamps. The CCTV
  timestamp overlay is not read by OCR.

## Events, screenshots, and persistent data

Persistent data is stored in different places:

| Data | Location |
|---|---|
| Machine names, RTSP URLs, occupancy settings, and camera credentials | `cnc_frontend\database\cnc.sqlite3` |
| Saved normalized zone polygons | `cnc_backend\data\outputs\camera_zones\` |
| Event CSV files and screenshots | `cnc_backend\data\outputs\cameras\<camera_id>\` |
| Model weights | `cnc_backend\models\` or the configured `MODEL_PATH` |

Each camera's event history is stored in an `events.csv` file and screenshots
in its `screenshots` directory. Events include a wall-clock timestamp,
video-relative time, event type, track IDs, and details. The dashboard supports
opening screenshots and deleting individual or selected events.

The frontend SQLite database and backend output directory are runtime data;
back them up before replacing or reinstalling the project. The Git ignore
rules exclude database files, local `.env` files, model weights, videos, logs,
and generated outputs.

Machine configuration and zone polygons persist across service restarts.
Active camera workers are in memory and are not automatically restarted by
the application after a server reboot. Start monitoring again from the
dashboard unless you have separately configured a reliable service startup
workflow.

## HTTP API

### Dashboard frontend

The dashboard serves HTML pages and proxies selected backend resources:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/machines/{machine_id}/monitor` | Machine preview/settings page |
| `POST` | `/machines` | Add a machine |
| `POST` | `/machines/{machine_id}/start-monitoring` | Start monitoring and save submitted zones |
| `GET` | `/machines/{machine_id}/preview` | Fetch a one-frame camera preview |
| `GET` | `/machines/{machine_id}/zones` | Check whether saved zones exist |
| `POST` | `/machines/{machine_id}/settings` | Update machine settings |
| `POST` | `/machines/{machine_id}/stop` | Stop monitoring |
| `POST` | `/machines/{machine_id}/delete` | Delete machine and backend outputs |
| `GET` | `/api/machines/{machine_id}/status` | Machine runtime status |
| `POST` | `/api/camera-credentials` | Save credentials for an RTSP host |
| `GET` | `/backend/detection/{camera_id}/stream` | Proxied MJPEG stream |
| `GET` | `/backend/outputs/{file_path}` | Proxied screenshot/output |
| `GET` | `/api/events` | Recent events |
| `POST` | `/api/events/delete` | Delete selected events |
| `POST` | `/api/events/{event_id}/delete` | Delete an event |

### Detection backend

The backend listens on port `9000` by default:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/detection/preview` | Fetch a camera preview frame |
| `POST` | `/detection/start` | Start a worker and optionally save zones |
| `POST` | `/detection/{camera_id}/stop` | Stop a worker |
| `DELETE` | `/detection/{camera_id}` | Stop a worker and delete that camera's output directory |
| `GET` | `/detection/{camera_id}/status` | Running state and zone occupancy |
| `GET` | `/detection/{camera_id}/zones` | Saved zone status |
| `GET` | `/detection/{camera_id}/stream` | Annotated MJPEG stream |
| `GET` | `/events` | List event records |
| `POST` | `/events/delete` | Delete selected event IDs |
| `DELETE` | `/events/{event_id}` | Delete one event |
| `GET` | `/outputs/...` | Serve generated screenshots |

The interactive API documentation is available at `/docs` on each FastAPI
service while it is running.

## Troubleshooting

### `Could not import module` when starting Uvicorn

Use the service's working directory and import path shown in
[Run locally](#run-locally). The backend is `monitoring.api:app` from
`cnc_backend`; the frontend is `web.app:app` from `cnc_frontend`.

### Dashboard returns `502 Bad Gateway`

Confirm the backend is running and responds at
<http://127.0.0.1:9000/health>. If it runs on another server, verify
`CNC_BACKEND_URL`, firewall rules, and network reachability from the frontend
host.

### Camera preview or live stream is blank

- Check RTSP URL syntax, credentials, channel path, and camera/NVR permissions.
- Test the stream from the backend host with an RTSP-capable player.
- Confirm the backend host can reach the NVR/camera and the RTSP port is
  allowed through network firewalls.
- Check the backend console for camera-open, frame-read, or worker errors.
- The browser does not open RTSP directly; it displays the backend's MJPEG
  stream.

### Monitoring will not start

- Draw and save the configured number of zones using the dashboard zone
  editor.
- Check that saved zone count matches the configured zone count.
- Confirm model weights exist at `MODEL_PATH` and are compatible with
  Ultralytics.
- Check the backend logs for camera, model, or phone-class configuration
  errors.

### `python-multipart` or other dependency errors

Install requirements into the same Python environment used to run the failing
service:

```powershell
Set-Location C:\Users\Sahil\Downloads\CNC_Detection
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Dashboard says Running but the worker stopped

Check backend output for per-camera processing errors and verify the current
camera/NVR RTSP URL. A worker runs in process memory; after a server or
backend restart, start the camera again from the dashboard.

### Counts or violation status do not match expectations

Check the machine's assigned zone polygons, person detection/tracking
quality, and configured per-zone allowed count. Only a person's bottom-center
point assigns them to a zone. The active over-limit timer is separate from
the five-minute absence timer.

### Uvicorn does not stop immediately

Stop active monitoring and close browser pages holding live MJPEG streams,
then stop the dashboard and backend processes. For controlled shutdowns,
Uvicorn supports `--timeout-graceful-shutdown`.

## Operational and security notes

- The dashboard and backend do **not** provide user login or role-based
  access control. Run them only on a trusted, restricted network.
- Do not expose ports `8000` or `9000` directly to the public internet.
- Treat the SQLite database as sensitive because it can contain RTSP
  credentials. Restrict filesystem permissions and include it in secure
  backups.
- Do not commit `.env` files, RTSP URLs containing credentials, database
  files, generated event data, or model weights.
- Keep the server clock synchronized; event timestamps are taken from the
  backend machine's local clock.
- Monitor disk space for event CSVs and screenshots, and establish a retention
  and backup policy appropriate for the site.
- The backend serves annotated frames and event images to the frontend; size
  CPU/GPU resources and network capacity for the number and resolution of
  simultaneous camera streams.
