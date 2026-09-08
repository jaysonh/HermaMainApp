"""Capture / streaming endpoints: /, /stream, /api/start, /api/stop,
/api/auto, /api/status, /api/snapshot.

/api/start opens the capture window (the live feed the visitor arranges organs
in); /api/stop — the [COMPLETE] button — closes it, and the render thread saves
the frame that was on screen and sends it off for analysis."""

import time
from urllib.parse import urlparse

import cv2
import requests
from flask import Blueprint, Response, jsonify

from .. import config, state
from . import localhost_or_api_key

bp = Blueprint("recording", __name__)


def _mjpeg_generator():
    while True:
        with state.jpeg_lock:
            frame_bytes = state.latest_jpeg
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
               + frame_bytes + b"\r\n")
        time.sleep(0.001)


@bp.get("/")
def index():
    return f"""<!doctype html><html><head><title>Herma Monitor</title>
<style>body{{font-family:system-ui;margin:24px}}img{{max-width:100%}}</style></head>
<body><h2>Herma Webcam Stream</h2>
<p>Capture timeout: {state.RECORDING_TIMEOUT}s</p>
<img src="/stream"/></body></html>"""


@bp.get("/stream")
def stream():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.post("/api/start")
@localhost_or_api_key
def api_start():
    with state.control_lock:
        state.manual_record_command = "start"
        state.recording_start_time = time.time()
    with state.organism_lock:
        state.show_organism = False
    with state.chat_lock:
        state.show_chat = False
        state.chat_messages = []
    print(f"API: Capture window open (auto-capture in {state.RECORDING_TIMEOUT}s)")
    return jsonify({"status": "ok", "action": "start_recording",
                    "timeout_seconds": state.RECORDING_TIMEOUT,
                    "output_dir": str(config.OUTPUT_DIR_API)})


@bp.post("/api/stop")
@localhost_or_api_key
def api_stop():
    with state.control_lock:
        state.manual_record_command = "stop"
        state.recording_start_time = None
    with state.organism_lock:
        state.show_organism = True
    with state.chat_lock:
        state.show_chat = False
    state.set_organism_text(config.LOADING_TEXT)
    print("API: Capture requested - screenshot goes off for analysis, waiting for sentences")
    return jsonify({"status": "ok", "action": "stop_recording", "waiting_for_sentences": True})


@bp.post("/api/auto")
@localhost_or_api_key
def api_auto():
    with state.control_lock:
        state.manual_record_command = None
        state.recording_start_time = None
    return jsonify({"status": "ok", "action": "auto_mode"})


@bp.post("/api/snapshot")
@localhost_or_api_key
def api_snapshot():
    with state.jpeg_lock:
        frame_bytes = state.latest_jpeg
    if frame_bytes is None and state.webcam_cap is not None:
        ret, frame = state.webcam_cap.read()
        if ret:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
            if ok:
                frame_bytes = buf.tobytes()
    if frame_bytes is None:
        return jsonify({"status": "error", "message": "No webcam frame available"}), 503
    try:
        parsed = urlparse(state.AWS_UPLOAD_URL)
        snapshot_url = f"http://{parsed.hostname}:5002/api/snapshot"
        resp = requests.post(
            snapshot_url,
            files={"image": ("snapshot.jpg", frame_bytes, "image/jpeg")},
            timeout=10,
        )
        print(f"API: Snapshot sent to {snapshot_url} — {resp.status_code}")
        return jsonify({"status": "ok", "action": "snapshot", "server_status": resp.status_code})
    except Exception as e:
        print(f"API: Snapshot upload failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 502


@bp.get("/api/status")
def api_status():
    with state.control_lock:
        if state.manual_record_command == "start":
            mode = "manual_record"
        elif state.manual_record_command == "stop":
            mode = "manual_stop"
        else:
            mode = "auto"
        time_remaining = None
        if state.recording_start_time is not None and state.manual_record_command == "start":
            time_remaining = max(0, state.RECORDING_TIMEOUT - (time.time() - state.recording_start_time))
        with state.intro_lock, state.organism_lock, state.chat_lock, state.sentences_lock:
            status = {
                **state.current_status,
                "mode": mode,
                "time_remaining": time_remaining,
                "intro_state": state.intro_state,
                "show_organism": state.show_organism,
                "show_chat": state.show_chat,
                "chat_messages": state.chat_messages,
                "chat_message_count": len(state.chat_messages),
                "sentences_active": state.sentences_active,
            }
            status.setdefault("images_captured", status.get("frames_captured"))
            return jsonify(status)
