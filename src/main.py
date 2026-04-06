import os
import cv2
import time
import json
import csv
import queue
import threading
import argparse
import io
from collections import deque
from datetime import datetime

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, render_template_string

try:
    from vehicle_detection import VehicleDetector
    from congestion_prediction import CongestionPredictor
    REAL_MODULES = True
except ImportError:
    REAL_MODULES = False

# ---------------------------------------------
# Stub classes (used when real modules absent)
# ---------------------------------------------

class StubVehicleDetector:
    CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

    def detect(self, frame):
        h, w = frame.shape[:2]
        n = int(5 + 10 * abs(np.sin(time.time() * 0.3)) + np.random.randint(0, 4))
        vehicles = []
        for _ in range(n):
            x1 = np.random.randint(0, w - 80)
            y1 = np.random.randint(0, h - 60)
            x2 = x1 + np.random.randint(40, 120)
            y2 = y1 + np.random.randint(30, 80)
            conf = round(np.random.uniform(0.55, 0.99), 2)
            cls = np.random.choice(self.CLASSES)
            vehicles.append(([x1, y1, x2, y2], conf, cls))
        return vehicles


class StubCongestionPredictor:
    LEVELS = ["FREE", "LIGHT", "MODERATE", "HEAVY", "GRIDLOCK"]

    def predict(self, density):
        if density < 5:   return "FREE"
        if density < 10:  return "LIGHT"
        if density < 16:  return "MODERATE"
        if density < 22:  return "HEAVY"
        return "GRIDLOCK"

# ---------------------------------------------
# Global state
# ---------------------------------------------

HISTORY_LEN = 120

state = {
    "running":       False,
    "paused":        False,
    "density":       0,
    "congestion":    "-",
    "fps":           0.0,
    "frame_count":   0,
    "alerts":        [],
    "class_counts":  {},
    "history": {
        "timestamps":  deque(maxlen=HISTORY_LEN),
        "density":     deque(maxlen=HISTORY_LEN),
        "fps":         deque(maxlen=HISTORY_LEN),
        "congestion":  deque(maxlen=HISTORY_LEN),
    },
    "session_log":   [],
    "start_time":    None,
}

frame_queue = queue.Queue(maxsize=2)
sse_clients  = []
sse_lock     = threading.Lock()

CONGESTION_ORDER = ["FREE", "LIGHT", "MODERATE", "HEAVY", "GRIDLOCK"]
CONGESTION_COLORS = {
    "FREE":     (0, 220, 120),
    "LIGHT":    (0, 200, 255),
    "MODERATE": (0, 140, 255),
    "HEAVY":    (0, 60,  255),
    "GRIDLOCK": (0, 0,   220),
}

# ---------------------------------------------
# SSE helpers
# ---------------------------------------------

def broadcast_sse(data: dict):
    payload = "data: " + json.dumps(data) + "\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

# ---------------------------------------------
# Video processing thread
# ---------------------------------------------

def processing_loop(video_path: str, detector, predictor):
    global state

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    state["running"] = True
    state["start_time"] = datetime.now().isoformat()

    fps_times = deque(maxlen=30)
    last_sample = 0.0
    alert_cooldown = 0

    print(f"[INFO] Processing: {video_path}")

    while state["running"]:
        if state["paused"]:
            time.sleep(0.05)
            continue

        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        vehicles = detector.detect(frame)
        density  = len(vehicles)

        class_counts = {}
        for _, _, cls in vehicles:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        congestion = predictor.predict(density)

        fps_times.append(time.time())
        if len(fps_times) > 1:
            fps = len(fps_times) / (fps_times[-1] - fps_times[0])
        else:
            fps = 0.0

        state["frame_count"] += 1
        state["density"]      = density
        state["congestion"]   = congestion
        state["fps"]          = round(fps, 1)
        state["class_counts"] = class_counts

        alert_cooldown -= 1
        if congestion in ("HEAVY", "GRIDLOCK") and alert_cooldown <= 0:
            alert = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "msg":  f"{congestion} congestion - {density} vehicles",
                "level": congestion,
            }
            state["alerts"].insert(0, alert)
            state["alerts"] = state["alerts"][:20]
            alert_cooldown = 60

        annotated = annotate(frame.copy(), vehicles, density, congestion, fps)

        try:
            frame_queue.put_nowait(annotated)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(annotated)
            except Exception:
                pass

        now = time.time()
        if now - last_sample >= 0.5:
            last_sample = now
            ts = datetime.now().strftime("%H:%M:%S")
            state["history"]["timestamps"].append(ts)
            state["history"]["density"].append(density)
            state["history"]["fps"].append(round(fps, 1))
            state["history"]["congestion"].append(
                CONGESTION_ORDER.index(congestion) if congestion in CONGESTION_ORDER else 0
            )

            log_row = {
                "timestamp": ts,
                "density": density,
                "congestion": congestion,
                "fps": round(fps, 1),
                **{f"cls_{k}": v for k, v in class_counts.items()},
            }
            state["session_log"].append(log_row)

            broadcast_sse({
                "density":      density,
                "congestion":   congestion,
                "fps":          round(fps, 1),
                "frame_count":  state["frame_count"],
                "class_counts": class_counts,
                "ts":           ts,
                "history": {
                    "timestamps": list(state["history"]["timestamps"]),
                    "density":    list(state["history"]["density"]),
                    "fps":        list(state["history"]["fps"]),
                    "congestion": list(state["history"]["congestion"]),
                },
            })

        elapsed = time.time() - t0
        time.sleep(max(0, 1/30 - elapsed))

    cap.release()
    state["running"] = False
    print("[INFO] Processing stopped.")


def annotate(frame, vehicles, density, congestion, fps):
    h, w = frame.shape[:2]
    color = CONGESTION_COLORS.get(congestion, (255, 255, 255))

    for box, conf, cls in vehicles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"DENSITY: {density}", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"CONGESTION: {congestion}", (14, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    return frame

# ---------------------------------------------
# Flask app
# ---------------------------------------------

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=1.0)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            except queue.Empty:
                pass
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=10)
    with sse_lock:
        sse_clients.append(q)

    def generate():
        yield "data: " + json.dumps({
            "density":      state["density"],
            "congestion":   state["congestion"],
            "fps":          state["fps"],
            "frame_count":  state["frame_count"],
            "class_counts": state["class_counts"],
            "ts":           datetime.now().strftime("%H:%M:%S"),
            "history": {
                "timestamps": list(state["history"]["timestamps"]),
                "density":    list(state["history"]["density"]),
                "fps":        list(state["history"]["fps"]),
                "congestion": list(state["history"]["congestion"]),
            },
        }) + "\n\n"

        try:
            while True:
                try:
                    data = q.get(timeout=20)
                    yield data
                except queue.Empty:
                    yield ": keep-alive\n\n"
        except GeneratorExit:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/status")
def api_status():
    return jsonify({
        "running":      state["running"],
        "paused":       state["paused"],
        "density":      state["density"],
        "congestion":   state["congestion"],
        "fps":          state["fps"],
        "frame_count":  state["frame_count"],
        "class_counts": state["class_counts"],
        "alerts":       state["alerts"][:5],
    })

@app.route("/api/history")
def api_history():
    return jsonify({
        "timestamps": list(state["history"]["timestamps"]),
        "density":    list(state["history"]["density"]),
        "fps":        list(state["history"]["fps"]),
        "congestion": list(state["history"]["congestion"]),
    })

@app.route("/api/alerts")
def api_alerts():
    return jsonify(state["alerts"])

@app.route("/api/control", methods=["POST"])
def api_control():
    action = request.json.get("action")
    if action == "pause":
        state["paused"] = True
    elif action == "resume":
        state["paused"] = False
    elif action == "stop":
        state["running"] = False
    return jsonify({"ok": True, "paused": state["paused"], "running": state["running"]})

@app.route("/api/export/csv")
def export_csv():
    if not state["session_log"]:
        return jsonify({"error": "No data"}), 404
    si = io.StringIO()
    keys = list(state["session_log"][0].keys())
    w = csv.DictWriter(si, fieldnames=keys)
    w.writeheader()
    w.writerows(state["session_log"])
    output = io.BytesIO(si.getvalue().encode())
    return send_file(output, mimetype="text/csv",
                     as_attachment=True, download_name="traffic_session.csv")

# ---------------------------------------------
# Dashboard HTML
# ---------------------------------------------

_here = os.path.dirname(__file__)
_html_path = os.path.join(_here, "dashboard.html")
DASHBOARD_HTML = open(_html_path).read() if os.path.exists(_html_path) else "<h1>dashboard.html not found</h1>"

# ---------------------------------------------
# Entry point
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="traffic4.mp4")
    parser.add_argument("--port",  default=5000, type=int)
    args = parser.parse_args()

    detector  = VehicleDetector(model_path="yolov8n.pt") if REAL_MODULES else StubVehicleDetector()
    predictor = CongestionPredictor()                     if REAL_MODULES else StubCongestionPredictor()

    mode = "REAL" if REAL_MODULES else "DEMO (stub)"
    print(f"\nTraffic Monitor starting - {mode} mode")
    print(f"   Video : {args.video}")
    print(f"   Dashboard: http://127.0.0.1:{args.port}\n")

    t = threading.Thread(target=processing_loop,
                         args=(args.video, detector, predictor),
                         daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
