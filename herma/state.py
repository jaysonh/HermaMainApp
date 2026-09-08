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
import time

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
    "capture_path": None,
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

# The whole set of sentences is shown as one page, typed out a character at a
# time. The render loop derives how much is visible from the start time, so the
# worker thread only has to publish the text once.
sentences_page_lock = threading.Lock()
sentences_page_text = ""
sentences_page_start = None   # time.monotonic() when typing began
sentences_page_cps = 20.0     # characters per second
sentences_page_align = "left"  # "left" for sentences, "center" for the thank-you
sentences_page_inset = False   # play the recording alongside this page?

# Path of the still captured for this run, shown beside the sentences.
recording_source_lock = threading.Lock()
last_recording_source = None

# While the page (and the thank-you that follows it) is up, the terrain shader
# is replaced by a looping background video.
show_end_video = False

# ─── Runtime-mutable Config ─────────────────────────────────────────────────
# These look like config but are mutated at startup (from TOML / env vars),
# so they live here rather than in herma.config which is import-time-only.

API_KEY = os.environ.get("MONITOR_API_KEY", "jayson")
AWS_UPLOAD_URL = os.environ.get("AWS_UPLOAD_URL", "http://10.142.77.6:5009/api/upload")
AWS_UPLOAD_KEY = os.environ.get("AWS_UPLOAD_KEY", "jayson")
END_VIDEO_PATH = config.END_VIDEO_PATH  # overridden from config.toml in main()


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


def set_last_recording(source):
    """Remember the still just captured, so the page can show it."""
    global last_recording_source
    with recording_source_lock:
        last_recording_source = source
    if source:
        print(f"Capture available for the sentences page: {source}")


def get_last_recording():
    with recording_source_lock:
        return last_recording_source


def start_sentences_page(text, chars_per_second, align="left", inset=False):
    """Publish a page of text and start the typewriter clock."""
    global sentences_page_text, sentences_page_start, sentences_page_cps
    global sentences_page_align, sentences_page_inset, show_end_video
    with sentences_page_lock:
        sentences_page_text = text
        sentences_page_align = align
        sentences_page_inset = inset
        sentences_page_cps = max(1.0, float(chars_per_second))
        sentences_page_start = time.monotonic()
        show_end_video = True


def clear_sentences_page():
    """Take the page down, leaving the background video running."""
    global sentences_page_text, sentences_page_start
    with sentences_page_lock:
        sentences_page_text = ""
        sentences_page_start = None


def get_sentences_page():
    """Return ``(text, start, cps, align, inset, show_end_video)``."""
    with sentences_page_lock:
        return (sentences_page_text, sentences_page_start, sentences_page_cps,
                sentences_page_align, sentences_page_inset, show_end_video)


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
    global sentences_page_text, sentences_page_start, show_end_video
    global sentences_page_inset, last_recording_source

    with control_lock:
        manual_record_command = None
        recording_start_time = None
        current_status = {
            "recording": False,
            "capture_path": None,
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

    with sentences_page_lock:
        sentences_page_text = ""
        sentences_page_start = None
        sentences_page_inset = False
        show_end_video = False

    with recording_source_lock:
        last_recording_source = None

    with jpeg_lock:
        latest_jpeg = None
