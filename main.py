# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io, time
from PIL import Image
import numpy as np
import mediapipe as mp
from collections import deque

# KPI algorithms
from algorithms.dms.drowsiness import DrowsinessEstimator
from algorithms.dms.yawn import YawnEstimator
from algorithms.dms.head_gaze import head_pose_from_box, gaze_zone_from_head
from algorithms.dms.face_mesh_utils import extract_landmarks
from algorithms.oms.occupants import occupant_metrics_from_faces

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
        self.last_kpis = {}

S = State()

# ---------- App ----------
app = FastAPI(title="Lingard Infer Service", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# MediaPipe face detection (BlazeFace)
mp_fd = mp.solutions.face_detection
mp_fm = mp.solutions.face_mesh
detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# KPI estimators (stateful)
drowsy = DrowsinessEstimator()
yawn = YawnEstimator()

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
    k = S.last_kpis or {}
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Lingard Infer Status</title>
<meta http-equiv="refresh" content="2">
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e8edf3; margin:0; }}
.container {{ max-width:1100px; margin:24px auto; padding:0 16px; }}
.card {{ background:#121822; border:1px solid #1f2a3a; border-radius:14px; padding:16px; }}
.kv {{ display:grid; grid-template-columns: 220px 1fr; gap:8px; }}
.key {{ color:#9fb0c3; }}
.val {{ color:#cfe5ff; }}
h1 {{ font-size:22px; margin:0 0 12px 0; letter-spacing:0.3px; }}
small {{ color:#9fb0c3; }}
.row {{ display:flex; gap:14px; overflow-x:auto; padding-bottom:8px; scroll-snap-type:x mandatory; }}
.kpi {{ min-width:220px; background:linear-gradient(180deg,#141b28,#0f1520); border:1px solid #1f2a3a; border-radius:14px; padding:14px; scroll-snap-align:start; box-shadow:0 6px 24px rgba(0,0,0,0.25); }}
.kpi h3 {{ margin:0 0 6px 0; font-size:16px; color:#cfe5ff; }}
.kpi .big {{ font-size:28px; color:#7fd0ff; font-weight:600; }}
.kpi .sub {{ color:#9fb0c3; font-size:12px; }}
.pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#1a2535; color:#9bd6ff; font-size:12px; }}
.grid {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }}
</style>
</head>
<body>
<div class="container">
  <div class="card" style="margin-bottom:16px;">
    <h1>Lingard Infer Service <small>v0.2.0</small></h1>
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

  <div class="row" aria-label="DMS KPIs">
    <div class="kpi"><h3>Drowsiness</h3><div class="big">{k.get('dms_drowsiness_score','-')}</div><div class="sub">score</div></div>
    <div class="kpi"><h3>PERCLOS</h3><div class="big">{k.get('dms_perclos_pct','-')}%</div><div class="sub">eye-closure</div></div>
    <div class="kpi"><h3>Blinks</h3><div class="big">{k.get('dms_blinks_per_min','-')}/min</div><div class="sub">rate</div></div>
    <div class="kpi"><h3>Microsleep</h3><div class="big">{('Yes' if k.get('dms_microsleep') else 'No') if k else '-'}</div><div class="sub">>1s eye closure</div></div>
    <div class="kpi"><h3>Yawn</h3><div class="big">{('Yes' if k.get('dms_yawning') else 'No') if k else '-'}</div><div class="sub">sustained mouth open</div></div>
    <div class="kpi"><h3>Gaze</h3><div class="big">{k.get('dms_gaze_zone','-')}</div><div class="sub">on-road {k.get('dms_gaze_on_road_pct','-')}%</div></div>
    <div class="kpi"><h3>Head Yaw</h3><div class="big">{k.get('dms_head_yaw_deg','-')}°</div><div class="sub">pitch {k.get('dms_head_pitch_deg','-')}°</div></div>
  </div>

  <div class="row" aria-label="OMS KPIs" style="margin-top:12px;">
    <div class="kpi"><h3>Occupants</h3><div class="big">{k.get('oms_occupant_count','-')}</div><div class="sub">cabin occupied: {k.get('oms_cabin_occupied','-')}</div></div>
    <div class="kpi"><h3>NCAP Score</h3><div class="big">{k.get('ncap_score_overall','-')}</div><div class="sub">heuristic</div></div>
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

        # Face detection (multi)
        results = detector.process(frame)
        ann_boxes = []
        face_conf = 0.0
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                ann_boxes.append([bbox.xmin, bbox.ymin, bbox.width, bbox.height])
            face_conf = float(results.detections[0].score[0])

        # Face landmarks for DMS KPIs
        h, w, _ = frame.shape
        lm_res = face_mesh.process(frame)
        landmarks = extract_landmarks(lm_res, w, h) if ann_boxes else None

        ts = time.time()
        drowsy_out = drowsy.update(ts, landmarks)
        yawn_out = yawn.update(ts, landmarks)

        # Head pose and gaze
        if ann_boxes:
            hp = head_pose_from_box(ann_boxes[0])
        else:
            hp = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
        gz = gaze_zone_from_head(hp["yaw_deg"], hp["pitch_deg"]) if ann_boxes else {"gaze_zone": "Unknown", "gaze_on_road_pct": 0.0}

        # Update status
        S.last_ok_ts = time.time()
        S.last_error = ""
        S.last_boxes = len(ann_boxes)
        S.last_face_conf = face_conf
        S.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # OMS metrics
        oms = occupant_metrics_from_faces(ann_boxes)

        # Aggregate payload
        payload = {
            "ann_boxes": ann_boxes,
            # DMS
            "dms_gaze_on_road_pct": gz["gaze_on_road_pct"],
            "dms_gaze_zone": gz["gaze_zone"],
            "dms_head_yaw_deg": round(hp["yaw_deg"], 1),
            "dms_head_pitch_deg": round(hp["pitch_deg"], 1),
            "dms_head_roll_deg": round(hp["roll_deg"], 1),
            "dms_perclos_pct": drowsy_out["perclos_pct"],
            "dms_blinks_per_min": drowsy_out["blinks_per_min"],
            "dms_microsleep": drowsy_out["microsleep"],
            "dms_drowsiness_score": drowsy_out["drowsiness_score"],
            "dms_yawning": yawn_out["yawning"],
            # OMS
            "oms_occupant_count": oms["occupant_count"],
            "oms_cabin_occupied": oms["cabin_occupied"],
            # Heuristic NCAP score placeholder
            "ncap_score_overall": 88 if ann_boxes else 60,
        }
        S.last_kpis = payload
        return JSONResponse(payload)

    except Exception as e:
        S.last_error = str(e)
        S.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return JSONResponse({"error": S.last_error}, status_code=500)
