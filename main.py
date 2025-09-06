# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io, time
from PIL import Image
import numpy as np
import mediapipe as mp
import threading
from collections import deque
import logging
from pydantic import BaseModel

# KPI algorithms
from algorithms.dms.drowsiness import DrowsinessEstimator
from algorithms.dms.yawn import YawnEstimator
from algorithms.dms.head_gaze import head_pose_from_box, gaze_zone_from_head, head_pose_from_landmarks
from algorithms.dms.face_mesh_utils import extract_landmarks, extract_all_landmarks, eye_aperture_points, lizard_direction_from_iris
from algorithms.oms.occupants import occupant_metrics_from_faces
from algorithms.dms.distraction import DistractionEstimator
from algorithms.scoring.ncap_scoring import NCAPScorer
from algorithms.oms.phone import PhoneUseEstimator
from algorithms.oms.seatbelt import SeatbeltEstimator
from algorithms.oms.child_presence import ChildPresenceEstimator
from algorithms.oms.smoking import SmokingEstimator
from algorithms.oms.hands_on_wheel import HandsOnWheelEstimator
from algorithms.dms.occlusion import OcclusionEstimator
from algorithms.utils.geometry import boxnorm_points, subsample, select_semantic_points
from algorithms.utils.tracker import BoxTracker

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
     
        # Tracking and per-person DMS
        self.tracker = BoxTracker()
        self.dms_by_id: dict[int, DrowsinessEstimator] = {}
        self.dms_last_seen: dict[int, float] = {}

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
face_mesh = mp_fm.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)
mp_lock = threading.Lock()
cfg_lock = threading.Lock()

# Logging (one-line INFO summaries per inference)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("lingard")

# KPI estimators (stateful)
drowsy = DrowsinessEstimator()
yawn = YawnEstimator()
distraction = DistractionEstimator()
ncap = NCAPScorer(config_path="config/ncap_config.json")
phone = PhoneUseEstimator()
seatbelt = SeatbeltEstimator()
child = ChildPresenceEstimator()
smoking = SmokingEstimator()
hands = HandsOnWheelEstimator()
occlusion = OcclusionEstimator()

def np_from_jpeg(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


# ---------- Config Models ----------
class DMSConfig(BaseModel):
    use_dynamic_threshold: bool | None = None
    ear_thresh_closed: float | None = None
    dynamic_rel_close: float | None = None
    dynamic_rel_open: float | None = None
    absolute_min_close: float | None = None
    microsleep_ms: int | None = None
    blink_min_s: float | None = None
    blink_max_s: float | None = None
    perclos_window_s: float | None = None
    sleep_min_s: float | None = None


class DistractionConfig(BaseModel):
    onroad_threshold_pct: float | None = None
    lizard_short_min_s: float | None = None
    lizard_long_s: float | None = None
    window_s: float | None = None
    owl_yaw_th_deg: float | None = None
    owl_short_min_s: float | None = None
    owl_long_s: float | None = None
    owl_pitch_th_deg: float | None = None


class TrackerConfig(BaseModel):
    iou_th: float | None = None
    center_th: float | None = None
    max_miss: int | None = None
    match_by_center: bool | None = None


def get_dms_config_dict():
    return {
        "use_dynamic_threshold": drowsy.use_dynamic_threshold,
        "ear_thresh_closed": drowsy.ear_thresh_closed,
        "dynamic_rel_close": drowsy.dynamic_rel_close,
        "dynamic_rel_open": drowsy.dynamic_rel_open,
        "absolute_min_close": drowsy.absolute_min_close,
        "microsleep_ms": drowsy.microsleep_ms,
        "blink_min_s": drowsy.blink_min_s,
        "blink_max_s": drowsy.blink_max_s,
        "perclos_window_s": getattr(drowsy, 'perclos_window_s', 30.0),
        "sleep_min_s": getattr(drowsy, 'sleep_min_s', 3.0),
    }


def get_distraction_config_dict():
    return {
        "onroad_threshold_pct": distraction.onroad_threshold_pct,
        "lizard_short_min_s": distraction.lizard_short_min_s,
        "lizard_long_s": distraction.lizard_long_s,
        "window_s": distraction.window_s,
        "owl_yaw_th_deg": distraction.owl_yaw_th_deg,
        "owl_short_min_s": distraction.owl_short_min_s,
        "owl_long_s": distraction.owl_long_s,
        "owl_pitch_th_deg": distraction.owl_pitch_th_deg,
    }


def get_tracker_config_dict():
    t = S.tracker
    return {
        "iou_th": t.iou_th,
        "center_th": t.center_th,
        "max_miss": t.max_miss,
        "match_by_center": t.match_by_center,
    }

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "uptime_s": round(time.time() - S.start_ts, 1)}


@app.get("/config/dms")
def get_dms_config():
    return {"ok": True, "config": get_dms_config_dict()}


@app.patch("/config/dms")
async def patch_dms_config(cfg: DMSConfig):
    # Apply non-None values atomically
    with cfg_lock:
        for k, v in cfg.dict(exclude_none=True).items():
            if hasattr(drowsy, k):
                setattr(drowsy, k, v)
    return {"ok": True, "config": get_dms_config_dict()}


@app.post("/config/dms/reset")
def reset_dms_state():
    # Clear history to re-baseline thresholds and rates
    with cfg_lock:
        from algorithms.dms.drowsiness import DrowsinessState
        drowsy.state = DrowsinessState()
    return {"ok": True, "message": "DMS state reset"}


@app.get("/config/distraction")
def get_distraction_config():
    return {"ok": True, "config": get_distraction_config_dict()}


@app.patch("/config/distraction")
async def patch_distraction_config(cfg: DistractionConfig):
    with cfg_lock:
        for k, v in cfg.dict(exclude_none=True).items():
            if hasattr(distraction, k):
                setattr(distraction, k, v)
    return {"ok": True, "config": get_distraction_config_dict()}


@app.get("/config/tracker")
def get_tracker_config():
    return {"ok": True, "config": get_tracker_config_dict()}


@app.patch("/config/tracker")
async def patch_tracker_config(cfg: TrackerConfig):
    with cfg_lock:
        S.tracker.configure(**cfg.dict(exclude_none=True))
    return {"ok": True, "config": get_tracker_config_dict()}

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
    <div class="kpi"><h3>Sleep</h3><div class="big">{('Yes' if k.get('dms_sleep_active') else 'No') if k else '-'}</div><div class="sub">dwell {k.get('dms_sleep_dwell_s','-')} s</div></div>
    <div class="kpi"><h3>Yawn</h3><div class="big">{('Yes' if k.get('dms_yawning') else 'No') if k else '-'}</div><div class="sub">sustained mouth open</div></div>
    <div class="kpi"><h3>Gaze</h3><div class="big">{k.get('dms_gaze_zone','-')}</div><div class="sub">on-road {k.get('dms_gaze_on_road_pct','-')}%</div></div>
    <div class="kpi"><h3>Head Yaw</h3><div class="big">{k.get('dms_head_yaw_deg','-')}°</div><div class="sub">pitch {k.get('dms_head_pitch_deg','-')}°</div></div>
    <div class="kpi"><h3>Eyes Off-Road</h3><div class="big">{k.get('dms_eyes_off_road_pct','-')}%</div><div class="sub">short {k.get('dms_eyes_off_short_per_min','-')}/m • long {k.get('dms_eyes_off_long_per_min','-')}/m</div></div>
    <div class="kpi"><h3>Owl</h3><div class="big">{('Long' if k.get('dms_owl_long_active') else ('Short' if k.get('dms_owl_short_active') else 'No'))}</div><div class="sub">dwell {k.get('dms_owl_yaw_dwell_s','-')} s</div></div>
    <div class="kpi"><h3>Blink Stats</h3><div class="big">{k.get('dms_avg_blink_dur_ms','-')} ms</div><div class="sub">last {k.get('dms_time_since_last_blink_s','-')} s, fps {k.get('dms_fps_est','-')}</div></div>
    <div class="kpi"><h3>Eyes (EAR)</h3><div class="big">L {k.get('dms_left_ear','-')} | R {k.get('dms_right_ear','-')}</div><div class="sub">closed L:{k.get('dms_left_eye_closed','-')} R:{k.get('dms_right_eye_closed','-')}</div></div>
    <div class="kpi"><h3>Look Dir</h3><div class="big">{k.get('dms_look_direction','-')}</div><div class="sub">yaw {k.get('dms_head_yaw_deg','-')}°, pitch {k.get('dms_head_pitch_deg','-')}°</div></div>
  </div>

  <div class="row" aria-label="OMS KPIs" style="margin-top:12px;">
    <div class="kpi"><h3>Occupants</h3><div class="big">{k.get('oms_occupant_count','-')}</div><div class="sub">cabin occupied: {k.get('oms_cabin_occupied','-')}</div></div>
    <div class="kpi"><h3>NCAP</h3><div class="big">{k.get('ncap_overall','-')}</div><div class="sub">mode: {k.get('ncap_mode','-')}</div></div>
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
        # MediaPipe detectors are not thread-safe; lock around calls
        with mp_lock:
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
        if ann_boxes:
            with mp_lock:
                lm_res = face_mesh.process(frame)
            landmarks = extract_landmarks(lm_res, w, h)
            all_landmarks = extract_all_landmarks(lm_res, w, h)
        else:
            landmarks = None
            all_landmarks = []

        ts = time.time()
        # Assign persistent IDs for current detections
        try:
            ids = S.tracker.assign(ann_boxes, ts)
        except Exception:
            ids = list(range(1, len(ann_boxes) + 1))
        drowsy_out = drowsy.update(ts, landmarks)
        yawn_out = yawn.update(ts, landmarks)
        landmarks_ok = landmarks is not None

        # Head pose and gaze (prefer landmarks when available)
        if landmarks is not None:
            hp = head_pose_from_landmarks(landmarks)
        elif ann_boxes:
            hp = head_pose_from_box(ann_boxes[0])
            hp["look_dir"] = "Straight" if abs(hp["yaw_deg"]) < 10 and abs(hp["pitch_deg"]) < 10 else ("Right" if hp["yaw_deg"] > 0 else "Left")
        else:
            hp = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "look_dir": "Unknown"}
        gz = gaze_zone_from_head(hp["yaw_deg"], hp["pitch_deg"]) if ann_boxes else {"gaze_zone": "Unknown", "gaze_on_road_pct": None}

        # Distraction KPIs from on-road estimate and head yaw
        if gz["gaze_on_road_pct"] is not None:
            dis_out = distraction.update(ts, gz["gaze_on_road_pct"], hp["yaw_deg"], hp["pitch_deg"])
        else:
            dis_out = distraction.update(ts, None, hp["yaw_deg"], hp["pitch_deg"])

        # (moved) iris-based lizard direction computed later after det->lm matching and active selection

        # Update status
        S.last_ok_ts = time.time()
        S.last_error = ""
        S.last_boxes = len(ann_boxes)
        S.last_face_conf = face_conf
        S.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # OMS metrics
        oms = occupant_metrics_from_faces(ann_boxes)
        # Placeholder additional KPIs (unknown by default)
        phone_out = phone.update(ts, frame)
        seatbelt_out = seatbelt.update(ts, frame)
        child_out = child.update(ts, frame)
        smoking_out = smoking.update(ts, frame)
        hands_out = hands.update(ts, frame)
        occlusion_out = occlusion.update(ts, landmarks)

        # Aggregate payload
        payload = {
            "ann_boxes": ann_boxes,
            # DMS
            "dms_gaze_on_road_pct": gz["gaze_on_road_pct"],
            "dms_gaze_zone": gz["gaze_zone"],
            "dms_head_yaw_deg": round(hp["yaw_deg"], 1),
            "dms_head_pitch_deg": round(hp["pitch_deg"], 1),
            "dms_head_roll_deg": round(hp["roll_deg"], 1),
            "dms_look_direction": hp.get("look_dir", "Unknown"),
            # Debug/diagnostics to validate blink/PERCLOS
            "dms_landmarks_ok": landmarks_ok,
            "dms_ear": drowsy_out.get("ear"),
            "dms_left_ear": drowsy_out.get("left_ear"),
            "dms_right_ear": drowsy_out.get("right_ear"),
            "dms_left_eye_closed": drowsy_out.get("left_eye_closed"),
            "dms_right_eye_closed": drowsy_out.get("right_eye_closed"),
            "dms_mar": yawn_out.get("mar"),
            "dms_perclos_pct": drowsy_out["perclos_pct"],
            "dms_blinks_per_min": drowsy_out["blinks_per_min"],
            "dms_blink_detected": drowsy_out.get("blink_detected"),
            "dms_avg_blink_dur_ms": drowsy_out["avg_blink_dur_ms"],
            "dms_time_since_last_blink_s": drowsy_out["time_since_last_blink_s"],
            "dms_microsleep": drowsy_out["microsleep"],
            "dms_microsleep_dwell_s": drowsy_out.get("microsleep_dwell_s"),
            "dms_sleep_active": drowsy_out.get("sleep_event_active"),
            "dms_sleep_dwell_s": drowsy_out.get("sleep_dwell_s"),
            "dms_drowsiness_score": drowsy_out["drowsiness_score"],
            "dms_fps_est": drowsy_out.get("fps_est"),
            "dms_ear_thresh_closed": drowsy_out.get("ear_thresh_closed"),
            "dms_ear_thresh_open": drowsy_out.get("ear_thresh_open"),
            "dms_perclos_window_s": drowsy_out.get("perclos_window_s"),
            "dms_yawning": yawn_out["yawning"],
            "dms_yawns_per_min": yawn_out["yawns_per_min"],
            # Distraction (lizard: eyes-off-road; owl: head yaw)
            "dms_eyes_off_road_pct": dis_out["eyes_off_road_pct"],
            "dms_eyes_off_road_dwell_s": dis_out["eyes_off_road_dwell_s"],
            "dms_eyes_off_short_active": dis_out["eyes_off_short_active"],
            "dms_eyes_off_long_active": dis_out["eyes_off_long_active"],
            "dms_eyes_off_short_per_min": dis_out["eyes_off_short_per_min"],
            "dms_eyes_off_long_per_min": dis_out["eyes_off_long_per_min"],
            "dms_owl_yaw_dwell_s": dis_out["owl_yaw_dwell_s"],
            "dms_owl_short_active": dis_out["owl_short_active"],
            "dms_owl_long_active": dis_out["owl_long_active"],
            "dms_owl_short_per_min": dis_out["owl_short_per_min"],
            "dms_owl_long_per_min": dis_out["owl_long_per_min"],
            "dms_owl_direction": dis_out["owl_direction"],
            "dms_lizard_direction": dis_out["lizard_direction"],
            # OMS
            "oms_occupant_count": oms["occupant_count"],
            "oms_cabin_occupied": oms["cabin_occupied"],
            "oms_phone_use": phone_out["phone_use"],
            "oms_seatbelt_fastened": seatbelt_out["seatbelt_fastened"],
            "oms_child_present": child_out["child_present"],
            "oms_smoking": smoking_out["smoking"],
            "oms_hands_on_wheel": hands_out["hands_on_wheel"],
            # DMS occlusion
            "dms_face_occluded": occlusion_out["face_occluded"],
            # Multi-person head pose + keypoints (box-normalized)
            "persons": [],
            # NCAP scoring (heuristic until protocol-config integrated)
        }
        # Build persons array aligned to detections; fill from FaceMesh when available
        persons = []
        count = len(ann_boxes)
        # If we have multiple landmark sets, match them to detection boxes by IoU of their min/max envelope
        def _iou(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
            inter = iw*ih
            if inter <= 0: return 0.0
            return inter / (aw*ah + bw*bh - inter + 1e-12)
        det_to_lm = {}
        if all_landmarks:
            lm_boxes = []
            for lm in all_landmarks:
                xs = [p[0]/w for p in lm]
                ys = [p[1]/h for p in lm]
                minx, maxx = max(0.0, min(xs)), min(1.0, max(xs))
                miny, maxy = max(0.0, min(ys)), min(1.0, max(ys))
                lm_boxes.append([minx, miny, max(1e-6, maxx-minx), max(1e-6, maxy-miny)])
            pairs = []
            for i, db in enumerate(ann_boxes):
                for j, lb in enumerate(lm_boxes):
                    pairs.append((i, j, _iou(db, lb)))
            pairs.sort(key=lambda t: t[2], reverse=True)
            used_d, used_l = set(), set()
            for i, j, u in pairs:
                if u <= 0: continue
                if i in used_d or j in used_l: continue
                det_to_lm[i] = j
                used_d.add(i); used_l.add(j)
        for i in range(count):
            box = ann_boxes[i]
            lm_idx = det_to_lm.get(i)
            if lm_idx is not None and lm_idx < len(all_landmarks):
                lm = all_landmarks[lm_idx]
                pose = head_pose_from_landmarks(lm)
                kp_px = select_semantic_points(lm)
                if not kp_px:
                    kp_px = subsample(lm.tolist(), target=24)
                kp_box = boxnorm_points(kp_px, box, w, h)
                # Eye keypoints for blink/microsleep (four points)
                eye_px = eye_aperture_points(lm)
                eye_box = boxnorm_points(eye_px, box, w, h)
                # Per-person DMS (stateful) keyed by ID
                pid = ids[i] if i < len(ids) else None
                person_kpis = None
                if pid is not None:
                    est = S.dms_by_id.get(pid)
                    if est is None:
                        est = DrowsinessEstimator()
                        S.dms_by_id[pid] = est
                    S.dms_last_seen[pid] = ts
                    outp = est.update(ts, lm)
                    person_kpis = {
                        "ear": outp.get("ear"),
                        "left_ear": outp.get("left_ear"),
                        "right_ear": outp.get("right_ear"),
                        "blink_detected": outp.get("blink_detected"),
                        "microsleep": outp.get("microsleep"),
                        "perclos_pct": outp.get("perclos_pct"),
                    }
                entry = {"index": i, "id": pid, **pose, "keypoints_box": kp_box, "eye_points_box": eye_box, "kpis": person_kpis}
            else:
                pose = head_pose_from_box(box)
                pose["look_dir"] = "Straight" if abs(pose["yaw_deg"]) < 10 and abs(pose["pitch_deg"]) < 10 else ("Right" if pose["yaw_deg"] > 0 else "Left")
                entry = {"index": i, "id": ids[i] if i < len(ids) else None, **pose, "keypoints_box": [], "eye_points_box": []}
            persons.append(entry)
        # Active person: choose largest box area
        active_idx = None
        if ann_boxes:
            areas = [b[2]*b[3] for b in ann_boxes]
            active_idx = int(max(range(len(areas)), key=lambda k: areas[k])) if areas else None
        active_id = (ids[active_idx] if active_idx is not None and active_idx < len(ids) else None) if ann_boxes else None
        payload["persons"] = persons
        payload["person_ids"] = ids
        payload["active_person_index"] = active_idx
        payload["active_person_id"] = active_id

        # If iris landmarks are present, override lizard direction using iris-based gaze for active face
        if all_landmarks and active_idx is not None:
            lm_idx2 = det_to_lm.get(active_idx)
            if lm_idx2 is not None and lm_idx2 < len(all_landmarks):
                lm_active = all_landmarks[lm_idx2]
                iris_dir, _ = lizard_direction_from_iris(lm_active, w, h, yaw_pitch_dir=dis_out.get("owl_direction"))
                if iris_dir:
                    payload["dms_lizard_direction"] = iris_dir

        # Cleanup stale per-person DMS states
        to_del = [pid for pid, last in S.dms_last_seen.items() if ts - last > 10.0]
        for pid in to_del:
            S.dms_last_seen.pop(pid, None)
            S.dms_by_id.pop(pid, None)
        # NCAP scoring
        ncap_out = ncap.score(payload)
        payload.update({
            "ncap_overall": ncap_out["overall"],
            "ncap_mode": ncap_out["mode"],
            "ncap_sections": ncap_out["sections"],
            "ncap_notes": ncap_out["notes"],
        })

        # ---------- Structured KPIs (nested) ----------
        # High-level classifications
        def classify_sleep():
            if payload.get("dms_sleep_active"):
                return "Sleep"
            if payload.get("dms_microsleep"):
                return "Microsleep"
            return "Awake"

        def classify_attention():
            # Attentive if on-road >= threshold and no active distraction
            onroad = payload.get("dms_gaze_on_road_pct")
            eyes_short = payload.get("dms_eyes_off_short_active")
            eyes_long = payload.get("dms_eyes_off_long_active")
            owl_short = payload.get("dms_owl_short_active")
            owl_long = payload.get("dms_owl_long_active")
            if onroad is None:
                return "Unknown"
            if eyes_long or owl_long:
                return "Distracted-Long"
            if eyes_short or owl_short:
                return "Distracted-Short"
            if onroad >= max(0.0, float(distraction.onroad_threshold_pct)):
                return "Attentive"
            return "Distracted"

        sleep_state = classify_sleep()
        attention_state = classify_attention()

        # Build nested KPI structure; keep existing flat fields for backward compatibility
        payload["kpi_version"] = "1.0"
        payload["kpis"] = {
            "dms": {
                "high_level": {
                    "sleep_state": sleep_state,
                    "drowsiness_score": payload.get("dms_drowsiness_score"),
                    "yawning": payload.get("dms_yawning"),
                    "yawns_per_min": payload.get("dms_yawns_per_min"),
                    "attention_state": attention_state,
                    "owl_direction": payload.get("dms_owl_direction"),
                    "lizard_direction": payload.get("dms_lizard_direction"),
                },
                "mid_level": {
                    "ears": {
                        "left": payload.get("dms_left_ear"),
                        "right": payload.get("dms_right_ear"),
                        "left_closed": payload.get("dms_left_eye_closed"),
                        "right_closed": payload.get("dms_right_eye_closed"),
                        "thresholds": {
                            "close": payload.get("dms_ear_thresh_closed"),
                            "open": payload.get("dms_ear_thresh_open"),
                        }
                    },
                    "perclos_pct": payload.get("dms_perclos_pct"),
                    "blinks_per_min": payload.get("dms_blinks_per_min"),
                    "blink": {
                        "avg_duration_ms": payload.get("dms_avg_blink_dur_ms"),
                        "time_since_last_s": payload.get("dms_time_since_last_blink_s"),
                        "detected": payload.get("dms_blink_detected"),
                    },
                    "microsleep": {
                        "active": payload.get("dms_microsleep"),
                        "dwell_s": payload.get("dms_microsleep_dwell_s"),
                    },
                    "sleep": {
                        "active": payload.get("dms_sleep_active"),
                        "dwell_s": payload.get("dms_sleep_dwell_s"),
                    },
                    "head_pose": {
                        "yaw_deg": payload.get("dms_head_yaw_deg"),
                        "pitch_deg": payload.get("dms_head_pitch_deg"),
                        "roll_deg": payload.get("dms_head_roll_deg"),
                        "look_dir": payload.get("dms_look_direction"),
                    },
                    "gaze": {
                        "zone": payload.get("dms_gaze_zone"),
                        "on_road_pct": payload.get("dms_gaze_on_road_pct"),
                    },
                    "distraction": {
                        "lizard": {
                            "off_road_pct": payload.get("dms_eyes_off_road_pct"),
                            "dwell_s": payload.get("dms_eyes_off_road_dwell_s"),
                            "short_active": payload.get("dms_eyes_off_short_active"),
                            "long_active": payload.get("dms_eyes_off_long_active"),
                            "short_per_min": payload.get("dms_eyes_off_short_per_min"),
                            "long_per_min": payload.get("dms_eyes_off_long_per_min"),
                            "direction": payload.get("dms_lizard_direction"),
                        },
                        "owl": {
                            "yaw_dwell_s": payload.get("dms_owl_yaw_dwell_s"),
                            "short_active": payload.get("dms_owl_short_active"),
                            "long_active": payload.get("dms_owl_long_active"),
                            "short_per_min": payload.get("dms_owl_short_per_min"),
                            "long_per_min": payload.get("dms_owl_long_per_min"),
                            "direction": payload.get("dms_owl_direction"),
                        }
                    },
                    "quality": {
                        "landmarks_ok": payload.get("dms_landmarks_ok"),
                        "fps_est": payload.get("dms_fps_est"),
                        "perclos_window_s": payload.get("dms_perclos_window_s"),
                    }
                },
            },
            "oms": {
                "high_level": {
                    "occupants": payload.get("oms_occupant_count"),
                    "cabin_occupied": payload.get("oms_cabin_occupied"),
                },
                "mid_level": {
                    "flags": {
                        "phone_use": payload.get("oms_phone_use"),
                        "seatbelt_fastened": payload.get("oms_seatbelt_fastened"),
                        "child_present": payload.get("oms_child_present"),
                        "smoking": payload.get("oms_smoking"),
                        "hands_on_wheel": payload.get("oms_hands_on_wheel"),
                    }
                }
            },
            "scoring": {
                "ncap": {
                    "overall": payload.get("ncap_overall"),
                    "mode": payload.get("ncap_mode"),
                    "sections": payload.get("ncap_sections"),
                    "notes": payload.get("ncap_notes"),
                }
            },
            "persons": {
                "active_index": payload.get("active_person_index"),
                "active_id": payload.get("active_person_id"),
                "ids": payload.get("person_ids"),
            }
        }
        # Compact INFO log to verify KPIs are sane
        try:
            perclos = float(payload.get("dms_perclos_pct") or 0.0)
            blinks = float(payload.get("dms_blinks_per_min") or 0.0)
            yawns_pm = float(payload.get("dms_yawns_per_min") or 0.0)
            eor_pct = float(payload.get("dms_eyes_off_road_pct") or 0.0)
            ear = payload.get("dms_ear")
            ear_s = f"{ear:.3f}" if isinstance(ear, (float, int)) and ear is not None else "NA"
            log.info(
                "200 OK faces=%d lat=%.1fms ear=%s perclos=%.1f blinks/min=%.1f yawns/min=%.2f microsleep=%s gaze=%s eor=%.1f ncap=%s ids=%s active=%s",
                S.last_boxes,
                S.last_latency_ms,
                ear_s,
                perclos,
                blinks,
                yawns_pm,
                str(payload.get("dms_microsleep")),
                str(payload.get("dms_gaze_zone")),
                eor_pct,
                str(payload.get("ncap_overall")),
                str(payload.get("person_ids")),
                str(payload.get("active_person_id")),
            )
        except Exception:
            # Never break response due to logging
            pass
        S.last_kpis = payload
        return JSONResponse(payload)

    except Exception as e:
        # Log full traceback for rapid debugging
        log.exception("/infer failed")
        S.last_error = str(e)
        S.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return JSONResponse({"error": S.last_error}, status_code=500)
