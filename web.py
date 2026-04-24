"""Flask web server for the Herma webcam shader.

Provides the MJPEG stream and HTTP control API. Shared mutable state and
configuration live in herma_webcam_shader.py; this module imports it as
``state`` and accesses everything through that reference. The circular
import is safe because ``state.X`` is only resolved at request-handling
time, not at module load.
"""

import functools
import json
import threading
import time
from urllib.parse import urlparse

import cv2
import requests
from flask import Flask, Response, jsonify, request as flask_request
from flask_cors import CORS

import herma_webcam_shader as state

flask_app = Flask(__name__)
CORS(flask_app)


def localhost_or_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if flask_request.remote_addr in ("127.0.0.1", "::1"):
            return f(*args, **kwargs)
        key = (flask_request.headers.get("X-API-Key")
               or flask_request.args.get("api_key"))
        if key != state.API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


def mjpeg_generator():
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


@flask_app.get("/")
def index():
    return f"""<!doctype html><html><head><title>Herma Monitor</title>
<style>body{{font-family:system-ui;margin:24px}}img{{max-width:100%}}</style></head>
<body><h2>Herma Webcam Stream</h2>
<p>Recording timeout: {state.RECORDING_TIMEOUT}s | Capture interval: {state.CAPTURE_INTERVAL}s</p>
<img src="/stream"/></body></html>"""


@flask_app.get("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.post("/api/end")
@localhost_or_api_key
def api_end():
    # Just request restart; render loop will perform it safely
    print("received /api/end")
    state.request_restart()
    print("API: End requested - restarting to initial state")
    return jsonify({"status": "ok", "action": "restart"})


@flask_app.post("/api/start")
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
    print(f"API: Start recording (timeout in {state.RECORDING_TIMEOUT}s)")
    return jsonify({"status": "ok", "action": "start_recording",
                    "timeout_seconds": state.RECORDING_TIMEOUT,
                    "output_dir": str(state.OUTPUT_DIR_API)})


@flask_app.post("/api/stop")
@localhost_or_api_key
def api_stop():
    with state.control_lock:
        state.manual_record_command = "stop"
        state.recording_start_time = None
    with state.organism_lock:
        state.show_organism = True
    with state.chat_lock:
        state.show_chat = False
    state._set_organism_text("Loading")
    print("API: Stop recording requested - waiting for sentences")
    return jsonify({"status": "ok", "action": "stop_recording", "waiting_for_sentences": True})


@flask_app.post("/api/auto")
@localhost_or_api_key
def api_auto():
    with state.control_lock:
        state.manual_record_command = None
        state.recording_start_time = None
    return jsonify({"status": "ok", "action": "auto_mode"})


@flask_app.post("/api/begin")
@localhost_or_api_key
def api_begin():
    with state.intro_lock:
        if state.intro_state == "logo":
            state.intro_state = "instructions"
            print("API: Transitioning from logo to instructions")
            return jsonify({"status": "ok", "action": "show_instructions",
                            "state": state.intro_state})
        else:
            return jsonify({"status": "error", "message": "Not in logo state",
                            "current_state": state.intro_state}), 400


@flask_app.post("/api/next")
@localhost_or_api_key
def api_next():
    with state.intro_lock:
        if state.intro_state == "instructions":
            state.intro_state = "running"
            print("API: Transitioning from instructions to running")
            return jsonify({"status": "ok", "action": "start_webcam",
                            "state": state.intro_state})
        else:
            return jsonify({"status": "error", "message": "Not in instructions state",
                            "current_state": state.intro_state}), 400


@flask_app.post("/api/chat")
@localhost_or_api_key
def api_chat():
    data = flask_request.get_json() or {}
    message = data.get("message", "Hello")
    sender = data.get("sender", "user")  # "organism" or "user"

    # Validate sender
    if sender not in ["organism", "user"]:
        return jsonify({"error": "sender must be 'organism' or 'user'"}), 400

    with state.chat_lock:
        state.show_chat = True
        # Add new message to list
        state.chat_messages.append({
            "sender": sender,
            "message": message,
            "timestamp": time.time()
        })
        # Keep only last MAX_CHAT_MESSAGES
        if len(state.chat_messages) > state.MAX_CHAT_MESSAGES:
            state.chat_messages = state.chat_messages[-state.MAX_CHAT_MESSAGES:]

    with state.organism_lock:
        state.show_organism = False

    print(f"API: Chat message from {sender}: {message}")
    return jsonify({"status": "ok", "action": "show_chat",
                    "sender": sender, "message": message,
                    "show_chat": True, "total_messages": len(state.chat_messages)})


@flask_app.post("/api/clear_chat")
@localhost_or_api_key
def api_clear_chat():
    with state.chat_lock:
        state.chat_messages = []
        state.show_chat = False
    print("API: Chat messages cleared")
    return jsonify({"status": "ok", "action": "clear_chat", "show_chat": False})


def _sentences_worker(sentences, seconds_per_char):
    """Background thread that cycles through sentences, displaying each one.
    Expects show_organism to already be True (set by /api/stop)."""
    try:
        for sentence in sentences:
            with state.sentences_lock:
                if state.sentences_stop:
                    break

            state._set_organism_text(sentence)

            # Display time proportional to sentence length (min 1s)
            display_time = max(1.0, len(sentence) * seconds_per_char)
            print(f"API: Displaying sentence ({display_time:.1f}s): {sentence[:60]}...")

            # Sleep in small increments so we can respond to stop requests
            elapsed = 0.0
            while elapsed < display_time:
                with state.sentences_lock:
                    if state.sentences_stop:
                        break
                time.sleep(0.1)
                elapsed += 0.1
    finally:
        with state.sentences_lock:
            was_stopped = state.sentences_stop
            state.sentences_active = False
            state.sentences_stop = False

        # Show thank-you message for 10 seconds, then restart both machines
        if not was_stopped:
            state._set_organism_text("Thank you for your experience")
            print("API: Showing thank-you message for 10s")
            time.sleep(10)

        # Send /api/restart to the SentiChat backend on the other computer
        try:
            parsed = urlparse(state.AWS_UPLOAD_URL)
            remote_url = f"http://{parsed.hostname}:5002/api/restart"
            resp = requests.post(remote_url,
                                 headers={"X-API-Key": state.API_KEY},
                                 timeout=5)
            print(f"API: Sent /api/restart to {remote_url} — {resp.status_code}")
        except Exception as e:
            print(f"API: Failed to send /api/restart to remote: {e}")

        # Restart this machine back to logo / HERMAPHROGENESIS screen
        state.request_restart()
        print("API: Sentences finished - restarting to initial state")


@flask_app.post("/api/sentences")
@localhost_or_api_key
def api_sentences():
    data = flask_request.get_json() or {}
    sentences = data.get("sentences", [])
    seconds_per_char = data.get("seconds_per_char", 0.05)

    if not isinstance(sentences, list) or len(sentences) == 0:
        return jsonify({"error": "Must provide a non-empty 'sentences' array"}), 400

    # Stop any currently running sequence
    with state.sentences_lock:
        if state.sentences_active:
            state.sentences_stop = True

    # Wait briefly for previous worker to finish
    for _ in range(20):
        with state.sentences_lock:
            if not state.sentences_active:
                break
        time.sleep(0.1)

    with state.sentences_lock:
        state.sentences_active = True
        state.sentences_stop = False

    threading.Thread(target=_sentences_worker, args=(sentences, seconds_per_char), daemon=True).start()

    total_time = sum(max(1.0, len(s) * seconds_per_char) for s in sentences)
    print(f"API: Starting sentences display ({len(sentences)} sentences, ~{total_time:.1f}s total)")
    return jsonify({
        "status": "ok",
        "action": "sentences",
        "sentence_count": len(sentences),
        "estimated_total_seconds": round(total_time, 1),
        "seconds_per_char": seconds_per_char,
    })


@flask_app.post("/api/sentences/stop")
@localhost_or_api_key
def api_sentences_stop():
    with state.sentences_lock:
        if not state.sentences_active:
            return jsonify({"status": "ok", "message": "No sentences playing"})
        state.sentences_stop = True
    return jsonify({"status": "ok", "action": "sentences_stop"})


@flask_app.post("/api/snapshot")
@localhost_or_api_key
def api_snapshot():
    with state.jpeg_lock:
        frame_bytes = state.latest_jpeg
    if frame_bytes is None and state.webcam_cap is not None:
        ret, frame = state.webcam_cap.read()
        if ret:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), state.JPEG_QUALITY])
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


@flask_app.get("/api/status")
def api_status():
    with state.control_lock:
        mode = ("manual_record" if state.manual_record_command == "start" else
                "manual_stop" if state.manual_record_command == "stop" else "auto")
        time_remaining = None
        if state.recording_start_time is not None and state.manual_record_command == "start":
            time_remaining = max(0, state.RECORDING_TIMEOUT - (time.time() - state.recording_start_time))
        with state.intro_lock:
            with state.organism_lock:
                with state.chat_lock:
                    with state.sentences_lock:
                        status = {**state.current_status,
                                  "mode": mode,
                                  "time_remaining": time_remaining,
                                  "intro_state": state.intro_state,
                                  "show_organism": state.show_organism,
                                  "show_chat": state.show_chat,
                                  "chat_messages": state.chat_messages,
                                  "chat_message_count": len(state.chat_messages),
                                  "sentences_active": state.sentences_active}
                        # back-compat alias
                        status.setdefault("images_captured", status.get("frames_captured"))
                        return jsonify(status)


def run_web_server():
    flask_app.run(host=state.WEB_HOST, port=state.WEB_PORT, threaded=True, use_reloader=False)
