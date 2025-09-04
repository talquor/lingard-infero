# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io, time
from PIL import Image
import numpy as np
import mediapipe as mp

# ---------- State ----------
class State:
    def __init__(self):
        self.start_ts = time.time()
        self.req_count = 0
        self.last_ok_ts = 0.0
        self.last_error = ""
        self.last_latency_ms = 0.0
        self.last_boxes = 0
        self.last_face_conf = 0.0

S = State()

# ---------- App ----------
app = FastAPI(title="Lingard Infer Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# MediaPipe face detection (BlazeFace)
mp_fd = mp.solutions.face_detection
detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def np_from_jpeg(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "uptime_s": round(time.time() - S.start_ts, 1)}

@app.get("/", response_class=HTMLResponse)
def home():
    uptime = round(time.time() - S.start_ts, 1)
    last_ok = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(S.last_ok_ts)) if S.last_ok_ts else "never"
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Lingard Infer Status</title>
<meta http-equiv="refresh" content="2">
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e8edf3; }}
.card {{ background:#121822; border:1px solid #1f2a3a; border-radius:12px; padding:16px; max-width:680px; margin:24px auto; }}
.kv {{ display:grid; grid-template-columns: 220px 1fr; gap:8px; }}
.key {{ color:#9fb0c3; }}
.val {{ color:#cfe5ff; }}
h1 {{ font-size:20px; margin:0 0 12px 0; }}
small {{ color:#9fb0c3; }}
</style>
</head>
<body>
<div class="card">
<h1>Lingard Infer Service <small>v0.1.0</small></h1>
<div class="kv">
  <div class="key">Uptime</div><div class="val">{uptime} s</div>
  <div class="key">Total POST /infer</div><div class="val">{S.req_count}</div>
  <div class="key">Last OK</div><div class="val">{last_ok}</div>
  <div class="key">Last latency</div><div class="val">{S.last_latency_ms:.1f} ms</div>
  <div class="key">Last face conf</div><div class="val">{S.last_face_conf:.3f}</div>
  <div class="key">Last boxes</div><div class="val">{S.last_boxes}</div>
  <div class="key">Health JSON</div><div class="val"><a style="color:#7fd0ff" href="/health">/health</a></div>
  <div class="key">OpenAPI</div><div class="val"><a style="color:#7fd0ff" href="/docs">/docs</a></div>
</div>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.post("/infer")
async def infer(request: Request):
    t0 = time.perf_counter()
    S.req_count += 1
    try:
        data = await request.body()
        frame = np_from_jpeg(data)  # HxWx3 RGB uint8

        # Face detection
        results = detector.process(frame)
        ann_boxes = []
        face_conf = 0.0
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            ann_boxes.append([bbox.xmin, bbox.ymin, bbox.width, bbox.height])
            face_conf = float(det.score[0])

        # KPIs (same as before, shortened here)
        attention = 80 if ann_boxes else 20
        gaze_on_road = 90 if ann_boxes else 40
        gaze_zone = "Road" if ann_boxes else "Unknown"

        if ann_boxes:
            cx = ann_boxes[0][0] + ann_boxes[0][2]/2
            cy = ann_boxes[0][1] + ann_boxes[0][3]/2
            yaw = (cx - 0.5) * 60.0
            pitch = (0.5 - cy) * 40.0
        else:
            yaw = pitch = 0.0
        roll = 0.0

        perclos = 8.0 if ann_boxes else 15.0
        blinks_per_min = 16 if ann_boxes else 8

        # Update status
        S.last_ok_ts = time.time()
        S.last_error = ""
        S.last_boxes = len(ann_boxes)
        S.last_face_conf = face_conf
        S.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        payload = {
            "ann_boxes": ann_boxes,
            "dms_attention_pct": attention,
            "dms_gaze_on_road_pct": gaze_on_road,
            "dms_gaze_zone": gaze_zone,
            "dms_head_yaw_deg": round(yaw,1),
            "dms_head_pitch_deg": round(pitch,1),
            "dms_head_roll_deg": roll,
            "dms_perclos_pct": perclos,
            "dms_blinks_per_min": blinks_per_min,
            "ncap_score_overall": 88 if ann_boxes else 60
        }
        return JSONResponse(payload)

    except Exception as e:
        S.last_error = str(e)
        S.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return JSONResponse({"error": S.last_error}, status_code=500)
