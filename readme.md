python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install fastapi uvicorn numpy pillow opencv-python mediapipe


uvicorn main.main:app --host 127.0.0.1 --port 8000


# (from your server folder, e.g. ~/Workspace/lingard-infero)
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# ✅ quote the extra requirement in zsh
python -m pip install 'uvicorn[standard]' fastapi pillow numpy opencv-python mediapipe
# or escape it:
# python -m pip install uvicorn\[standard] fastapi pillow numpy opencv-python mediapipe


 python -m uvicorn main:app --host 127.0.0.1 --port 8000


## KPIs implemented

- DMS vigilance: PERCLOS, blink rate, blink duration, microsleep (>1s), yawn detection (MAR), head pose (yaw/pitch), coarse gaze zone and on-road percentage.
- DMS distraction: eyes-off-road percentage, events per minute (dwell >=2s), current off-road dwell.
- OMS basics: occupant count, cabin occupied.
- NCAP scoring: heuristic NCAP-like scoring via `algorithms/scoring/ncap_scoring.py`.

Notes on NCAP
- The NCAP module ships with placeholder thresholds and weights. Replace them with the official Euro NCAP protocol parameters to claim compliance.
- Integration point: `NCAPConfig` in `algorithms/scoring/ncap_scoring.py`. Wire your protocol/points table and test cases there; the app exposes aggregated metrics in `/infer` responses and summarizes on `/`.
