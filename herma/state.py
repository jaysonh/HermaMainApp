"""Shared mutable state for Herma.

This module is imported by both the GLFW render thread and the Flask
thread. Python's import cache guarantees a single module object, so
``from herma import state`` from anywhere refers to the same instance.

Every mutable global has an associated ``threading.Lock``. Always hold
the matching lock when reading or writing — see how the render loop and
the Flask blueprints do it.
"""

import os
import threading

from . import config

# ─── Webcam ─────────────────────────────────────────────────────────────────

latest_jpeg = None
jpeg_lock = threading.Lock()
webcam_cap = None  # cv2.VideoCapture, set in main()

# ─── Recording control ──────────────────────────────────────────────────────

control_lock = threading.Lock()
manual_record_command = None        # None | "start" | "stop"
recording_start_time = None
current_status = {
    "recording": False,
    "video_path": None,
    "frames_captured": 0,
    "last_motion_time": None,
    "time_remaining": None,
}

# ─── Intro / Lifecycle ──────────────────────────────────────────────────────

# "logo" → "instructions" → "running"
intro_state = "logo"
intro_lock = threading.Lock()

# ─── Organism Display ───────────────────────────────────────────────────────

show_organism = False
organism_lock = threading.Lock()
organism_overlay_text = config.ORGANISM_TEXT
organism_text_dirty = False  # render loop re-uploads texture when True

# ─── Chat Display ───────────────────────────────────────────────────────────

show_chat = False
chat_messages = []  # list of {"sender", "message", "timestamp"}
chat_lock = threading.Lock()

# ─── Restart Flag ───────────────────────────────────────────────────────────

restart_requested = False
restart_lock = threading.Lock()

# ─── Sentences Playback ─────────────────────────────────────────────────────

sentences_lock = threading.Lock()
sentences_active = False
sentences_stop = False

# ─── Runtime-mutable Config ─────────────────────────────────────────────────
# These look like config but are mutated at startup (from TOML / env vars)
# or at runtime (RECORDING_INPUT_TYPE), so they live here rather than in
# herma.config which is import-time-only.

API_KEY = os.environ.get("MONITOR_API_KEY", "jayson")
AWS_UPLOAD_URL = os.environ.get("AWS_UPLOAD_URL", "http://10.142.77.6:5009/api/upload")
AWS_UPLOAD_KEY = os.environ.get("AWS_UPLOAD_KEY", "jayson")
RECORDING_INPUT_TYPE = "video"  # "video" or "image_sequence" — refreshed per-recording


# ─── State Mutations ────────────────────────────────────────────────────────

def request_restart():
    global restart_requested
    with restart_lock:
        restart_requested = True


def consume_restart_request() -> bool:
    global restart_requested
    with restart_lock:
        if restart_requested:
            restart_requested = False
            return True
        return False


def set_organism_text(text):
    """Update the organism overlay text from any thread."""
    global organism_overlay_text, organism_text_dirty
    with organism_lock:
        organism_overlay_text = text
        organism_text_dirty = True


def reset_to_initial_state():
    """Reset all shared state back to initial startup defaults."""
    global manual_record_command, recording_start_time, current_status
    global intro_state, show_organism, show_chat, chat_messages, latest_jpeg
    global organism_overlay_text, organism_text_dirty, sentences_stop

    with control_lock:
        manual_record_command = None
        recording_start_time = None
        current_status = {
            "recording": False,
            "video_path": None,
            "frames_captured": 0,
            "last_motion_time": None,
            "time_remaining": None,
        }

    with intro_lock:
        intro_state = "logo"

    with organism_lock:
        show_organism = False
        organism_overlay_text = config.ORGANISM_TEXT
        organism_text_dirty = True

    with chat_lock:
        show_chat = False
        chat_messages = []

    with sentences_lock:
        sentences_stop = True

    with jpeg_lock:
        latest_jpeg = None
