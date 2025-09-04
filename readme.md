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