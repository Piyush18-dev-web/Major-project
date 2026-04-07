import streamlit as st
import cv2
import time
import numpy as np
from collections import deque
from datetime import datetime
import os

st.set_page_config(page_title="Traffic Monitor", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #0a0e1a; color: #e0e6f0; }
.stApp { background-color: #0a0e1a; }
.metric-card { background: linear-gradient(135deg, #0f1629 0%, #151d35 100%); border: 1px solid #1e2d50; border-radius: 12px; padding: 20px 24px; text-align: center; }
.metric-label { font-family: 'Share Tech Mono', monospace; font-size: 11px; letter-spacing: 2px; color: #4a6fa5; text-transform: uppercase; margin-bottom: 8px; }
.metric-value { font-family: 'Share Tech Mono', monospace; font-size: 42px; font-weight: 700; line-height: 1; }
.metric-sub { font-size: 12px; color: #4a6fa5; margin-top: 6px; }
.badge { display: inline-block; padding: 6px 18px; border-radius: 20px; font-family: 'Share Tech Mono', monospace; font-size: 13px; font-weight: 700; letter-spacing: 2px; }
.badge-FREE     { background: #0d3320; color: #00dc78; border: 1px solid #00dc78; }
.badge-LIGHT    { background: #0d2e3a; color: #00c8ff; border: 1px solid #00c8ff; }
.badge-MODERATE { background: #2a2000; color: #ffc800; border: 1px solid #ffc800; }
.badge-HEAVY    { background: #2a1000; color: #ff6400; border: 1px solid #ff6400; }
.badge-GRIDLOCK { background: #2a0000; color: #ff2020; border: 1px solid #ff2020; }
.alert-item { background: #1a0f0f; border-left: 3px solid #ff4040; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-family: 'Share Tech Mono', monospace; font-size: 12px; color: #ff9090; }
.section-title { font-family: 'Share Tech Mono', monospace; font-size: 11px; letter-spacing: 3px; color: #2d4a7a; text-transform: uppercase; border-bottom: 1px solid #1a2540; padding-bottom: 8px; margin-bottom: 16px; }
section[data-testid="stSidebar"] { background: #080c18 !important; border-right: 1px solid #1a2540; }
#MainMenu, footer, header { visibility: hidden; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; animation: pulse 1.5s infinite; }
.dot-live { background: #00dc78; } .dot-paused { background: #ffc800; } .dot-stopped { background: #ff4040; animation: none; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
""", unsafe_allow_html=True)

try:
    from vehicle_detection import VehicleDetector
    from congestion_prediction import CongestionPredictor
    REAL_MODULES = True
except ImportError:
    REAL_MODULES = False

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
    def predict(self, density):
        if density < 5:   return "FREE"
        if density < 10:  return "LIGHT"
        if density < 16:  return "MODERATE"
        if density < 22:  return "HEAVY"
        return "GRIDLOCK"

CONGESTION_COLORS = {
    "FREE": (0, 220, 120), "LIGHT": (0, 200, 255),
    "MODERATE": (0, 200, 255), "HEAVY": (0, 100, 255), "GRIDLOCK": (0, 0, 220),
}
HISTORY_LEN = 60
VIDEO_CACHE_DIR = "/tmp/traffic_videos"
os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)

SAMPLE_URLS = {
    "-- Select a sample --": "",
    "NYC Times Square":      "https://www.youtube.com/watch?v=gSu_Y5OEofc",
    "Tokyo Highway":         "https://www.youtube.com/watch?v=MNn9qKG2UFI",
    "India Street Traffic":  "https://www.youtube.com/watch?v=wqctLW0Hb7s",
    "Highway Cam USA":       "https://www.youtube.com/watch?v=1EiC9bvVGnk",
}

def download_youtube(url: str) -> str:
    from pytubefix import YouTube
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").first()
    if not stream:
        stream = yt.streams.filter(file_extension="mp4").order_by("resolution").first()
    vid_id = yt.video_id
    local_path = os.path.join(VIDEO_CACHE_DIR, f"{vid_id}.mp4")
    if not os.path.exists(local_path):
        stream.download(output_path=VIDEO_CACHE_DIR, filename=f"{vid_id}.mp4")
    return local_path

def init_state():
    defaults = {
        "running": False, "paused": False, "density": 0,
        "congestion": "FREE", "fps": 0.0, "frame_count": 0,
        "alerts": [], "class_counts": {},
        "history_density": deque(maxlen=HISTORY_LEN),
        "history_ts": deque(maxlen=HISTORY_LEN),
        "alert_cooldown": 0, "current_frame": None,
        "video_loaded": False, "cap": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

@st.cache_resource
def get_resources():
    d = VehicleDetector(model_path="yolov8n.pt") if REAL_MODULES else StubVehicleDetector()
    p = CongestionPredictor() if REAL_MODULES else StubCongestionPredictor()
    return d, p

def load_video(path):
    if st.session_state.cap is not None:
        try: st.session_state.cap.release()
        except: pass
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        st.session_state.cap = cap
        st.session_state.video_loaded = True
        return True
    return False

def annotate_frame(frame, vehicles, density, congestion, fps):
    h, w = frame.shape[:2]
    color = CONGESTION_COLORS.get(congestion, (255, 255, 255))
    for box, conf, cls in vehicles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"DENSITY: {density}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"{congestion}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"FPS:{fps:.1f}  {datetime.now().strftime('%H:%M:%S')}", (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return frame

def process_one_frame():
    if not st.session_state.running or st.session_state.paused: return
    if st.session_state.cap is None or not st.session_state.cap.isOpened(): return
    detector, predictor = get_resources()
    cap = st.session_state.cap
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    if not ret: return
    vehicles   = detector.detect(frame)
    density    = len(vehicles)
    congestion = predictor.predict(density)
    class_counts = {}
    for _, _, cls in vehicles:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    st.session_state.density      = density
    st.session_state.congestion   = congestion
    st.session_state.fps          = 15.0
    st.session_state.frame_count += 1
    st.session_state.class_counts = class_counts
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history_density.append(density)
    st.session_state.history_ts.append(ts)
    st.session_state.alert_cooldown -= 1
    if congestion in ("HEAVY", "GRIDLOCK") and st.session_state.alert_cooldown <= 0:
        st.session_state.alerts.insert(0, {"time": ts, "msg": f"{congestion} - {density} vehicles detected"})
        st.session_state.alerts = st.session_state.alerts[:10]
        st.session_state.alert_cooldown = 10
    annotated = annotate_frame(frame.copy(), vehicles, density, congestion, 15.0)
    st.session_state.current_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

process_one_frame()

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
with st.sidebar:
    st.markdown("## Traffic Monitor")
    st.markdown(f"*{'REAL' if REAL_MODULES else 'DEMO'} mode*")
    st.divider()

    st.markdown('<div class="section-title">YouTube Video Source</div>', unsafe_allow_html=True)

    selected = st.selectbox("Quick pick", list(SAMPLE_URLS.keys()))
    if selected != "-- Select a sample --":
        yt_url = SAMPLE_URLS[selected]
    else:
        yt_url = st.text_input("Or paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Download & Load", type="primary", disabled=not yt_url):
        with st.spinner("Downloading from YouTube... this may take 1-2 min"):
            try:
                local_path = download_youtube(yt_url)
                if load_video(local_path):
                    st.success("Ready! Press Start.")
                else:
                    st.error("Could not open video after download.")
            except Exception as e:
                st.error(f"Failed: {str(e)}")

    st.divider()
    st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", use_container_width=True, type="primary",
                     disabled=st.session_state.running or not st.session_state.video_loaded):
            st.session_state.running = True
            st.session_state.paused  = False
            st.rerun()
    with col2:
        if st.button("Stop", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False
            st.session_state.paused  = False
            st.rerun()
    if st.session_state.running:
        if st.button("Pause" if not st.session_state.paused else "Resume", use_container_width=True):
            st.session_state.paused = not st.session_state.paused
            st.rerun()

    st.divider()
    if st.session_state.running and not st.session_state.paused:
        st.markdown('<span class="status-dot dot-live"></span> **LIVE**', unsafe_allow_html=True)
    elif st.session_state.paused:
        st.markdown('<span class="status-dot dot-paused"></span> **PAUSED**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot dot-stopped"></span> **STOPPED**', unsafe_allow_html=True)
    st.markdown(f"**Frames:** {st.session_state.frame_count:,}")

    st.divider()
    st.markdown('<div class="section-title">Congestion Scale</div>', unsafe_allow_html=True)
    for level, color in [("FREE","#00dc78"),("LIGHT","#00c8ff"),("MODERATE","#ffc800"),("HEAVY","#ff6400"),("GRIDLOCK","#ff2020")]:
        st.markdown(f'<span style="color:{color}">&#9632;</span> {level}', unsafe_allow_html=True)

# -----------------------------------------------
# Main layout
# -----------------------------------------------
st.markdown("# Traffic Monitoring Dashboard")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
congestion = st.session_state.congestion
color_map = {"FREE":"#00dc78","LIGHT":"#00c8ff","MODERATE":"#ffc800","HEAVY":"#ff6400","GRIDLOCK":"#ff2020"}
col = color_map.get(congestion, "#ffffff")

with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Vehicle Density</div><div class="metric-value" style="color:{col}">{st.session_state.density}</div><div class="metric-sub">vehicles in frame</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Congestion Level</div><div style="margin-top:8px"><span class="badge badge-{congestion}">{congestion}</span></div><div class="metric-sub" style="margin-top:10px">current status</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Processing FPS</div><div class="metric-value" style="color:#4a9eff">{st.session_state.fps:.1f}</div><div class="metric-sub">frames / second</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Frames</div><div class="metric-value" style="color:#a78bfa">{st.session_state.frame_count:,}</div><div class="metric-sub">processed</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.markdown('<div class="section-title">Live Video Feed</div>', unsafe_allow_html=True)
    if st.session_state.current_frame is not None:
        st.image(st.session_state.current_frame, use_container_width=True)
    else:
        st.markdown('<div style="height:300px;background:#050810;border:1px solid #1e2d50;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#2d4a7a;font-family:monospace;font-size:13px;">Download a YouTube video and press START</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Density History</div>', unsafe_allow_html=True)
    if st.session_state.history_density:
        import pandas as pd
        st.line_chart(pd.DataFrame({"Density": list(st.session_state.history_density)}), height=180)
    else:
        st.caption("No data yet")
    st.markdown('<div class="section-title">Vehicle Classes</div>', unsafe_allow_html=True)
    if st.session_state.class_counts:
        import pandas as pd
        st.bar_chart(pd.DataFrame.from_dict(st.session_state.class_counts, orient='index', columns=['Count']), height=160)
    else:
        st.caption("No data yet")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Congestion Alerts</div>', unsafe_allow_html=True)
if st.session_state.alerts:
    for alert in st.session_state.alerts[:5]:
        st.markdown(f'<div class="alert-item">[{alert["time"]}] {alert["msg"]}</div>', unsafe_allow_html=True)
else:
    st.caption("No alerts - traffic flowing normally.")

if st.session_state.running and not st.session_state.paused:
    time.sleep(0.5)
    st.rerun()
