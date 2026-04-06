# 🚦 Traffic Monitoring System

A real-time traffic monitoring backend built with Flask and YOLOv8. Streams annotated video, exposes live metrics via Server-Sent Events (SSE), and provides REST APIs for dashboards, alerts, and CSV export.

---

## Features

- **Live video stream** — annotated MJPEG feed with bounding boxes and a HUD overlay
- **Real-time SSE stream** — per-frame metrics pushed to connected dashboards (~2×/sec)
- **Vehicle detection** — YOLOv8-powered detection with per-class counts (car, truck, bus, motorcycle, bicycle)
- **Congestion prediction** — five levels: FREE → LIGHT → MODERATE → HEAVY → GRIDLOCK
- **Alert system** — auto-triggers on HEAVY / GRIDLOCK congestion with a cooldown
- **Rolling history** — last 120 data-points (~2 min) for charting
- **CSV export** — full session log downloadable via REST endpoint
- **Demo / stub mode** — runs without real ML modules for testing

---

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/traffic-monitor.git
cd traffic-monitor

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Install the GPU build of PyTorch before installing `ultralytics` for significantly faster inference. See [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

```bash
python app.py --video traffic.mp4
```

Then open your browser at **http://127.0.0.1:5000**

### CLI Arguments

| Argument  | Default        | Description                        |
|-----------|----------------|------------------------------------|
| `--video` | `traffic4.mp4` | Path to the input video file       |
| `--port`  | `5000`         | Port for the Flask web server      |

### Demo Mode

If `vehicle_detection` or `congestion_prediction` modules are not found, the app automatically falls back to **stub mode** — simulated detections and congestion levels — so the full dashboard and API surface can be tested without ML dependencies.

---

## API Reference

### GET `/video_feed`
MJPEG stream of the annotated video. Embed directly in an `<img>` tag:
```html
<img src="http://127.0.0.1:5000/video_feed" />
```

### GET `/stream`
SSE stream of live metrics. Each event carries:
```json
{
  "density": 14,
  "congestion": "MODERATE",
  "fps": 28.3,
  "frame_count": 420,
  "class_counts": { "car": 9, "truck": 3, "bus": 2 },
  "ts": "14:32:07",
  "history": { "timestamps": [...], "density": [...], "fps": [...], "congestion": [...] }
}
```

### GET `/api/status`
Snapshot of current state (density, congestion, FPS, alerts, class counts).

### GET `/api/history`
Rolling history arrays for charts.

### GET `/api/alerts`
List of the last 20 congestion alerts.

### GET `/api/export/csv`
Download the full session log as `traffic_session.csv`.

### POST `/api/control`
Control playback. Body: `{ "action": "pause" | "resume" | "stop" }`.

---

## Project Structure

```
traffic-monitor/
├── app.py                  # Main application (Flask server + processing loop)
├── dashboard.html          # Frontend dashboard (loaded by Flask)
├── vehicle_detection.py    # (Optional) YOLOv8 vehicle detector
├── congestion_prediction.py# (Optional) Congestion predictor
├── requirements.txt
└── README.md
```

---

## Configuration

Key constants at the top of `app.py`:

| Constant         | Default | Description                              |
|------------------|---------|------------------------------------------|
| `HISTORY_LEN`    | `120`   | Number of data-points kept in memory     |
| `alert_cooldown` | `60`    | Frames between repeated HEAVY/GRIDLOCK alerts |
| MJPEG quality    | `80`    | JPEG quality for the video stream (0–100)|

---

## License

MIT
