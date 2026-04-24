#!/usr/bin/env python3
"""Herma Webcam Shader

Displays webcam feed embedded in a melting terrain shader.

Requirements:
    pip install glfw PyOpenGL opencv-python numpy flask flask-cors requests Pillow
    (If Python < 3.11) pip install tomli

Controls:
    ESC    - Quit
    Tab    - Toggle fullscreen/windowed
    Space  - Toggle HUD status bar
    Scroll - Zoom in/out
"""

import sys
import time
import threading
import ctypes
import json
import math
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import os
import argparse

import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli

try:
    import glfw
    from OpenGL.GL import *  # noqa: F403
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install glfw PyOpenGL opencv-python numpy flask flask-cors requests Pillow")
    sys.exit(1)

from shaders import VERT_SRC, FRAG_SRC, HUD_VERT_SRC, HUD_FRAG_SRC

# ─── Configuration ──────────────────────────────────────────────────────────

CAM_INDEX = 1
WINDOW_W = 1280
WINDOW_H = 720
GRID_RES = 400
TARGET_ASPECT = 16.0 / 9.0

# Default shader parameters (from herma-record.html)
# rect3 is the top green square, rect4 is the bottom purple square
P = dict(
    heightScale=0.3, dripSpeed=0.4, distortion=0.5, ringCount=8.0,
    terrainHue=-0.08, terrainSat=1.0, terrainBright=1.0, terrainContrast=1.0,
    rect1X=0.5, rect1Y=0.5, rect1W=0.42, rect1H=0.4,
    rect1Elev=0.3, rect1Blend=0.25,
    rect1Hue=0.0, rect1Sat=1.0, rect1Bright=1.0, rect1Contrast=1.0,
    rect3X=0.0, rect3Y=0.86, rect3W=0.22, rect3H=0.06,
    rect3Elev=0.0, rect3Blend=0.0,
    rect3Hue=0.2, rect3Sat=1.0, rect3Bright=1.1, rect3Contrast=1.0,
    rect4X=0.06, rect4Y=0.14, rect4W=0.11, rect4H=0.05,
    rect4Elev=0.0, rect4Blend=0.00,
    rect4Hue=0.2, rect4Sat=1.3, rect4Bright=1.0, rect4Contrast=1.0,
)

# ─── Recording / Web Server Settings ────────────────────────────────────────

MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 2500

CAPTURE_INTERVAL = 0.5
RECORDING_TIMEOUT = 30.0
STILL_SECONDS_TO_STOP = 5.0

DOWNSCALE_WIDTH = 1920
OUTPUT_DIR_API = Path("recorded")
OUTPUT_DIR_AUTO = Path("tmp")

WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
STREAM_FPS = 10
JPEG_QUALITY = 80

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 10
VIDEO_FRAME_INTERVAL = 1.0 / VIDEO_FPS

API_KEY = os.environ.get("MONITOR_API_KEY", "jayson")
AWS_UPLOAD_URL = os.environ.get("AWS_UPLOAD_URL", "http://10.142.77.6:5009/api/upload")
AWS_UPLOAD_KEY = os.environ.get("AWS_UPLOAD_KEY", "jayson")
RECORDING_INPUT_TYPE = "video"  # "video" or "image_sequence" — updated at startup from HermaSentiChat

# ─── Font Loading ────────────────────────────────────────────────────────────
_FONT_PATH = str(Path(__file__).resolve().parent / "assets" / "sylfaen.ttf")

def _load_font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except (IOError, OSError):
        print(f"Warning: Could not load {_FONT_PATH}, falling back to default font")
        return ImageFont.load_default()

ONBOARDING_TEXT = "The objects in front of you are precise replicas of human internal organs.You are invited to make your own arrangement, using as many or as few as you wish.Pick up the first organ to start creating a new anatomy."

ORGANISM_TEXT = ""

# ─── Shared State ───────────────────────────────────────────────────────────

latest_jpeg = None
jpeg_lock = threading.Lock()
webcam_cap = None  # set once the camera is opened in main()

control_lock = threading.Lock()
manual_record_command = None
recording_start_time = None
current_status = {
    "recording": False,
    "video_path": None,
    "frames_captured": 0,
    "last_motion_time": None,
    "time_remaining": None,
}

# Intro state: "logo", "instructions", "running"
intro_state = "logo"
intro_lock = threading.Lock()

# Organism display state
show_organism = False
organism_lock = threading.Lock()
organism_overlay_text = ORGANISM_TEXT  # Dynamic text shown on organism screen
organism_text_dirty = False  # Flag to signal render loop to re-upload texture

# Chat display state
show_chat = False
chat_messages = []  # List of {"sender": "organism"|"user", "message": "text", "timestamp": float}
chat_lock = threading.Lock()
MAX_CHAT_MESSAGES = 10  # Maximum messages to display on screen

# Restart request flag (set by Flask thread, acted on by render loop)
restart_requested = False
restart_lock = threading.Lock()

# Sentences playback state (driven by /api/sentences worker, observed by render loop)
sentences_lock = threading.Lock()
sentences_active = False  # True while a sentence sequence is playing
sentences_stop = False    # Flag to cancel a running sequence

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

def reset_to_initial_state():
    """Reset all shared state back to initial startup defaults."""
    global manual_record_command, recording_start_time, current_status
    global intro_state, show_organism, show_chat, chat_messages, latest_jpeg
    global organism_overlay_text, organism_text_dirty

    # Stop recording mode + timers
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

    # Reset intro back to logo
    with intro_lock:
        intro_state = "logo"

    # Hide overlays
    with organism_lock:
        show_organism = False
        organism_overlay_text = ORGANISM_TEXT
        organism_text_dirty = True

    with chat_lock:
        show_chat = False
        chat_messages = []

    # Stop any running sentence sequence
    global sentences_stop
    with sentences_lock:
        sentences_stop = True

    # Clear MJPEG frame
    with jpeg_lock:
        latest_jpeg = None

# ─── Config helpers ─────────────────────────────────────────────────────────

def _deep_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config_toml(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("rb") as f:
        return tomllib.load(f)


def pick_monitor_from_config(display_cfg: dict):
    """
    Uses:
      [display].display_primary_monitor (bool)
      [display].monitor_index (int, 0-based)
    """
    use_primary = bool(_deep_get(display_cfg, "display_primary_monitor", default=True))
    if use_primary:
        return glfw.get_primary_monitor()

    monitors = glfw.get_monitors()
    if not monitors:
        return glfw.get_primary_monitor()

    idx = int(_deep_get(display_cfg, "monitor_index", default=0))
    idx = max(0, min(idx, len(monitors) - 1))
    return monitors[idx]


HUD_WIDTH = 800
HUD_HEIGHT = 36


def render_hud_text(mode_str, rec_state, img_index, time_remaining, has_motion):
    pil_img = Image.new("RGBA", (HUD_WIDTH, HUD_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(20)

    parts = [mode_str, rec_state]
    if rec_state != "IDLE":
        parts.append(f"imgs={img_index}")
    if time_remaining is not None:
        parts.append(f"timeout={time_remaining:.1f}s")
    if has_motion:
        parts.append("MOTION")
    text = "  |  ".join(parts)

    draw.text((12, 6), text, font=font, fill=(255, 255, 255, 255))
    img = np.array(pil_img, dtype=np.uint8)
    img = cv2.flip(img, 0)  # flip vertically for GL texture origin
    return img


def render_overlay_text(text, width, height):
    """Render multi-line text centered on a semi-transparent background with word wrapping."""
    pil_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(64)

    margin = 80
    max_text_width = width - margin * 2
    line_height = 90

    # Word-wrap each paragraph
    wrapped_lines = []
    for paragraph in text.strip().split('\n'):
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        words = paragraph.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                current_line = word
        if current_line:
            wrapped_lines.append(current_line)

    # Calculate total text block height
    total_height = len(wrapped_lines) * line_height
    y_start = (height - total_height) // 2

    # Draw semi-transparent grey background behind text
    pad = 30
    bg_x0 = margin - pad
    bg_y0 = y_start - pad
    bg_x1 = width - margin + pad
    bg_y1 = y_start + total_height + pad
    draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=(128, 128, 128, 160))

    for i, line in enumerate(wrapped_lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, font=font, fill=(0, 0, 0, 255))

    img = np.array(pil_img, dtype=np.uint8)
    img = cv2.flip(img, 0)  # flip vertically for GL texture origin
    return img


def _animated_dots():
    """Return 1-3 dots cycling based on current time."""
    return "." * (int(time.time() * 2) % 3 + 1)


def render_organism_overlay(text, width, height):
    """Render organism name + description with a grey translucent panel sized to fit the text."""
    pil_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)

    title_font = _load_font(44)
    body_font = _load_font(26)

    # Animate dots on any "Loading" text
    dots = _animated_dots()
    text = text.replace("Loading organism", f"Loading organism{dots}")

    # Layout constants
    panel_padding = 30
    line_height = 38
    title_line_height = 56
    title_bottom_gap = 20

    # Max text area width (screen minus padding on both sides)
    max_text_area_w = width - panel_padding * 4

    # Split title from description
    parts = text.strip().split('\n', 1)
    title = parts[0].strip() if parts else ""
    description = parts[1].strip() if len(parts) > 1 else ""

    # --- First pass: compute wrapped lines and total content height ---
    title_lines = []
    if title:
        words = title.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=title_font)
            if bbox[2] - bbox[0] <= max_text_area_w:
                current_line = test_line
            else:
                if current_line:
                    title_lines.append(current_line)
                current_line = word
        if current_line:
            title_lines.append(current_line)

    body_lines = []
    if description:
        for paragraph in description.split('\n'):
            if not paragraph.strip():
                body_lines.append("")
                continue
            words = paragraph.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=body_font)
                if bbox[2] - bbox[0] <= max_text_area_w:
                    current_line = test_line
                else:
                    if current_line:
                        body_lines.append(current_line)
                    current_line = word
            if current_line:
                body_lines.append(current_line)

    # Calculate content height
    content_h = 0
    if title_lines:
        content_h += len(title_lines) * title_line_height + title_bottom_gap
    content_h += len(body_lines) * line_height

    # Panel dimensions sized to fit content
    panel_h = content_h + panel_padding * 2
    panel_w = max_text_area_w + panel_padding * 2

    # Centre the panel on screen
    panel_x = (width - panel_w) // 2
    panel_y = (height - panel_h) // 2

    # Draw semi-transparent grey background
    draw.rectangle(
        [(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
        fill=(30, 30, 30, 200))

    # --- Second pass: draw text ---
    text_x = panel_x + panel_padding
    text_y = panel_y + panel_padding

    for tl in title_lines:
        draw.text((text_x, text_y), tl, font=title_font, fill=(255, 255, 255, 255))
        text_y += title_line_height
    if title_lines:
        text_y += title_bottom_gap

    for bl in body_lines:
        if bl:
            draw.text((text_x, text_y), bl, font=body_font, fill=(220, 220, 220, 255))
        text_y += line_height

    img = np.array(pil_img, dtype=np.uint8)
    img = cv2.flip(img, 0)  # flip vertically for GL texture origin
    return img


def render_chat_messages(messages, width, height):
    """Render chat messages in bubble style - organism on left, user on right."""
    pil_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(22)

    if not messages:
        return np.array(pil_img, dtype=np.uint8)

    padding = 20
    bubble_padding = 15
    message_spacing = 15
    line_height = 35
    max_bubble_width = int(width * 0.4)  # 40% of screen width per message

    margin = 50

    # First pass: compute bubble layout for every message (oldest to newest)
    bubble_data = []
    for msg in messages:
        sender = msg.get("sender", "user")
        text = msg.get("message", "")

        # Wrap text to fit in bubble
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_bubble_width - (bubble_padding * 2):
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        if not lines:
            continue

        # Calculate bubble dimensions
        max_line_width = max([draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines])
        bubble_width = max_line_width + (bubble_padding * 2)
        bubble_height = len(lines) * line_height + (bubble_padding * 2)

        bubble_data.append({
            'sender': sender,
            'lines': lines,
            'bubble_width': bubble_width,
            'bubble_height': bubble_height,
        })

    # Total height of all messages + typing indicator
    dots_indicator_height = line_height + (bubble_padding * 2)
    total_height = (sum(b['bubble_height'] for b in bubble_data)
                    + message_spacing * len(bubble_data)
                    + dots_indicator_height)

    # Anchor newest message at the bottom of the image; older messages above it.
    # After cv2.flip this puts newest at the bottom of the GL screen.
    # If total_height exceeds the available space, start_y goes negative and
    # old messages are clipped off the top (= top of GL screen after flip).
    start_y = min(margin, (height - margin) - total_height)
    y_position = start_y

    for data in bubble_data:
        bh = data['bubble_height']

        # Skip messages fully above the image (fallen off the top)
        if y_position + bh < 0:
            y_position += bh + message_spacing
            continue

        bw = data['bubble_width']
        sender = data['sender']
        lines = data['lines']

        # Calculate bubble position based on sender
        if sender == "organism":
            # Left side - organism messages
            bubble_x = padding
            text_color = (255, 255, 255, 255)  # White text
            bubble_color = (0, 0, 0, 220)      # Black background
        else:
            # Right side - user messages
            bubble_x = width - bw - padding
            text_color = (0, 0, 0, 255)        # Black text
            bubble_color = (255, 255, 255, 220) # White background

        bubble_y = y_position

        # Draw bubble background
        draw.rectangle(
            [(bubble_x, bubble_y), (bubble_x + bw, bubble_y + bh)],
            fill=bubble_color)

        # Draw text lines
        text_y = bubble_y + bubble_padding
        for line in lines:
            draw.text((bubble_x + bubble_padding, text_y), line,
                      font=font, fill=text_color)
            text_y += line_height

        # Move down for next message
        y_position = bubble_y + bh + message_spacing

    # Animated typing indicator: show on the opposite side of the last message
    last_sender = messages[-1].get("sender", "user")
    dots_sender = "organism" if last_sender == "user" else "user"
    dot_count = int(time.time() * 2) % 3 + 1  # cycles 1,2,3
    dots_text = "." * dot_count

    dots_bbox = draw.textbbox((0, 0), "...", font=font)
    min_dots_width = dots_bbox[2] - dots_bbox[0]
    dots_bw = min_dots_width + (bubble_padding * 2)
    dots_bh = line_height + (bubble_padding * 2)

    if dots_sender == "organism":
        dots_x = padding
        dots_text_color = (255, 255, 255, 255)
        dots_bubble_color = (0, 0, 0, 220)
    else:
        dots_x = width - dots_bw - padding
        dots_text_color = (0, 0, 0, 255)
        dots_bubble_color = (255, 255, 255, 220)

    if y_position + dots_bh >= 0:
        draw.rectangle(
            [(dots_x, y_position), (dots_x + dots_bw, y_position + dots_bh)],
            fill=dots_bubble_color)
        draw.text((dots_x + bubble_padding, y_position + bubble_padding),
                  dots_text, font=font, fill=dots_text_color)

    img = np.array(pil_img, dtype=np.uint8)
    img = cv2.flip(img, 0)  # flip vertically for GL texture origin
    return img


# ─── Recording Helpers ──────────────────────────────────────────────────────

def timestamp_folder_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _set_organism_text(text):
    """Update the organism overlay text from any thread."""
    global organism_overlay_text, organism_text_dirty
    with organism_lock:
        organism_overlay_text = text
        organism_text_dirty = True


def upload_video(video_path: Path):
    if not video_path.exists():
        print(f"No video to upload: {video_path}")
        return False
    try:
        print(f"Uploading video {video_path.name}...")
        with open(video_path, "rb") as fh:
            response = requests.post(
                AWS_UPLOAD_URL,
                headers={"X-API-Key": AWS_UPLOAD_KEY},
                files={"files": (video_path.name, fh, "video/mp4")},
                data={
                    "sequence_name": video_path.parent.name,
                    "timestamp": video_path.parent.name,
                },
                timeout=120,
            )
        if response.ok:
            print(f"Upload successful: {response.json()}")
            _set_organism_text("Loading organism")
            analyse_video(video_path)
            return True
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Upload error: {e}")
        return False


def fetch_recording_mode() -> str:
    """Fetch the recording input_type from HermaSentiChat settings. Returns 'video' or 'image_sequence'."""
    parsed = urlparse(AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/settings/recording-mode"
    try:
        resp = requests.get(url, timeout=5)
        if resp.ok:
            mode = resp.json().get('input_type', 'video')
            print(f"Recording mode from server: {mode}")
            return mode
    except Exception as e:
        print(f"Could not fetch recording mode (defaulting to 'video'): {e}")
    return 'video'


def upload_and_analyse_images(image_paths: list, seq_name: str):
    """Upload image frames to HermaUploadReceiver then send to HermaSentiChat for analysis."""
    if not image_paths:
        print("No image frames to upload")
        return
    parsed = urlparse(AWS_UPLOAD_URL)

    # Upload to receiver
    try:
        print(f"Uploading {len(image_paths)} frames to receiver as sequence '{seq_name}'...")
        open_files = []
        multipart = []
        for p in image_paths:
            fh = open(p, 'rb')
            open_files.append(fh)
            multipart.append(('files', (Path(p).name, fh, 'image/jpeg')))
        response = requests.post(
            AWS_UPLOAD_URL,
            headers={"X-API-Key": AWS_UPLOAD_KEY},
            files=multipart,
            data={"sequence_name": seq_name, "image_count": str(len(image_paths)), "timestamp": seq_name},
            timeout=120,
        )
        for fh in open_files:
            fh.close()
        if response.ok:
            print(f"Image upload successful: {response.json()}")
        else:
            print(f"Image upload failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Image upload error: {e}")

    # Analyse
    _set_organism_text("Loading organism")
    analyse_images(image_paths)


def analyse_images(image_paths: list):
    """Call the analyse endpoint with image frames (SSE), wait for agent2b, and display results."""
    parsed = urlparse(AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/analyse-video-agents"
    print(f"Starting image sequence analysis ({len(image_paths)} frames)...")
    organism_info = {}
    try:
        open_files = []
        multipart = []
        for p in image_paths:
            fh = open(p, 'rb')
            open_files.append(fh)
            multipart.append(('images', (Path(p).name, fh, 'image/jpeg')))
        response = requests.post(
            url,
            files=multipart,
            params={"agent2c": "false", "agent3": "false"},
            stream=True,
            timeout=300,
        )
        for fh in open_files:
            fh.close()
        if not response.ok:
            print(f"Analysis request failed: {response.status_code} - {response.text}")
            return
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    print(f"Analysis: {data}")
                    if (data.get('stage') == 'agent2b'
                            and data.get('status') == 'complete'
                            and data.get('data')):
                        desc = data['data'].get('visual_description', '')
                        name = data['data'].get('organism_name', '')
                        sys_desc = data['data'].get('system_description', '')
                        organism_info = {
                            'organism_name': name,
                            'visual_description': desc,
                            'system_description': sys_desc,
                        }
                        if desc:
                            overlay_text = f"{name}\n\n{desc}" if name else desc
                            _set_organism_text(overlay_text)
                            print(f"Organism visual description set: {name}")
                            try:
                                chatready_url = f"http://{parsed.hostname}:5002/api/chatready"
                                requests.post(chatready_url, json=organism_info, timeout=5)
                                print(f"Sent /api/chatready to {chatready_url}")
                            except Exception as e:
                                print(f"Failed to send /api/chatready: {e}")
                    if data.get('done') or data.get('error'):
                        break
    except requests.exceptions.RequestException as e:
        print(f"Analysis error: {e}")


def analyse_video(video_path: Path):
    """Call the video analysis endpoint (SSE), wait for agent2b, and display results."""
    parsed = urlparse(AWS_UPLOAD_URL)
    url = f"http://{parsed.hostname}:5002/api/analyse-video-agents"
    print(f"Starting video analysis for: {video_path}")
    organism_info = {}  # track organism fields for /api/chatready
    try:
        with open(video_path, "rb") as fh:
            response = requests.post(
               url,
               files={"video": (video_path.name, fh, "video/mp4")},
               params={"agent2c": "false", "agent3": "false"},
               stream=True,
               timeout=300,
            )
        if not response.ok:
            print(f"Analysis request failed: {response.status_code} - {response.text}")
            return
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    print(f"Analysis: {data}")

                    # When agent2b completes, show the visual description and send chatready
                    if (data.get('stage') == 'agent2b'
                            and data.get('status') == 'complete'
                            and data.get('data')):
                        desc = data['data'].get('visual_description', '')
                        name = data['data'].get('organism_name', '')
                        sys_desc = data['data'].get('system_description', '')
                        organism_info = {
                            'organism_name': name,
                            'visual_description': desc,
                            'system_description': sys_desc,
                        }
                        if desc:
                            overlay_text = f"{name}\n\n{desc}" if name else desc
                            _set_organism_text(overlay_text)
                            print(f"Organism visual description set: {name}")
                            # Notify herma server that chat is ready
                            try:
                                chatready_url = f"http://{parsed.hostname}:5002/api/chatready"
                                requests.post(chatready_url, json=organism_info, timeout=5)
                                print(f"Sent /api/chatready to {chatready_url}")
                            except Exception as e:
                                print(f"Failed to send /api/chatready: {e}")

                    if data.get('done') or data.get('error'):
                        break
    except requests.exceptions.RequestException as e:
        print(f"Analysis error: {e}")


# ─── Helper Functions ───────────────────────────────────────────────────────

def compile_shader(src, stype):
    shader = glCreateShader(stype)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader


def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog)
        if isinstance(log, bytes):
            log = log.decode()
        raise RuntimeError(f"Program link error:\n{log}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def make_grid(res, aspect=1.0):
    n = res + 1
    positions = np.zeros((n * n, 3), dtype=np.float32)
    uvs = np.zeros((n * n, 2), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            i = y * n + x
            nx, ny = x / res, y / res
            uv_x = (nx - 0.5) * aspect + 0.5
            uv_y = ny
            positions[i] = [(nx - 0.5) * 2 * aspect, (ny - 0.5) * 2, 0]
            uvs[i] = [uv_x, uv_y]
    indices = np.zeros(res * res * 6, dtype=np.uint32)
    k = 0
    for y in range(res):
        for x in range(res):
            i = y * n + x
            indices[k:k + 6] = [i, i + 1, i + n, i + 1, i + n + 1, i + n]
            k += 6
    return positions.flatten(), uvs.flatten(), indices


def make_ortho(l, r, b, t, near, far):
    return np.array([
        2 / (r - l), 0, 0, 0,
        0, 2 / (t - b), 0, 0,
        0, 0, -2 / (far - near), 0,
        -(r + l) / (r - l), -(t + b) / (t - b), -(far + near) / (far - near), 1
    ], dtype=np.float32)


def load_image_texture(image_path):
    """Load image and create OpenGL texture."""
    if not image_path.exists():
        print(f"Warning: Image file not found: {image_path}")
        return None, 0, 0

    try:
        img = Image.open(image_path).convert("RGBA")
        img_data = np.array(img, dtype=np.uint8)
        img_data = np.flipud(img_data)  # Flip for OpenGL

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        print(f"Loaded image: {image_path} ({img.width}x{img.height})")
        return tex, img.width, img.height
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, 0, 0


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Herma Webcam Shader")
    parser.add_argument("--config", type=str, default=str(script_dir / "config.toml"),
                        help="Path to config.toml")
    parser.add_argument("camera_index", nargs="?", default=None,
                        help="Optional camera index override (int)")
    args = parser.parse_args()

    cfg = load_config_toml(Path(args.config))

    # Read TOML values with YOUR names
    toml_cam_index = int(_deep_get(cfg, "video", "cam_index", default=CAM_INDEX))
    display_cfg = _deep_get(cfg, "display", default={}) or {}
    herma_host = str(_deep_get(cfg, "herma_server", "host", default="10.142.77.6"))
    herma_port = int(_deep_get(cfg, "herma_server", "port", default=5009))
    shader_cfg_path = str(_deep_get(cfg, "shader_settings", "shader_config_path", default="shader-settings.json"))

    # CLI camera override
    cam_idx = toml_cam_index
    if args.camera_index is not None:
        try:
            cam_idx = int(args.camera_index)
        except ValueError:
            print(f"Usage: {sys.argv[0]} [--config path/to/config.toml] [camera_index]")
            sys.exit(1)

    # Init GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)

    # Start in fullscreen on monitor from config
    monitor = pick_monitor_from_config(display_cfg)
    mode = glfw.get_video_mode(monitor)
    win = glfw.create_window(mode.size.width, mode.size.height,
                             "Herma Webcam Shader", monitor, None)
    if not win:
        glfw.terminate()
        print("Failed to create window")
        sys.exit(1)

    glfw.make_context_current(win)
    glfw.swap_interval(1)

    # Build AWS upload URL from [herma_server] unless AWS_UPLOAD_URL env var overrides it
    global AWS_UPLOAD_URL, RECORDING_INPUT_TYPE
    AWS_UPLOAD_URL = os.environ.get("AWS_UPLOAD_URL", f"http://{herma_host}:{herma_port}/api/upload")

    # Resolve shader settings path (relative to script dir)
    shader_path = Path(shader_cfg_path)
    if not shader_path.is_absolute():
        shader_path = (script_dir / shader_path).resolve()

    print(f"Config: {Path(args.config)}")
    print(f"Camera index: {cam_idx}")
    print(f"Herma server: {herma_host}:{herma_port}")
    print(f"Upload URL: {AWS_UPLOAD_URL}")
    print(f"Shader settings file: {shader_path}")

    # Track fullscreen state and saved windowed geometry for restore
    is_fullscreen = [True]
    windowed_pos = [100, 100]
    windowed_size = [WINDOW_W, WINDOW_H]

    # Open webcam
    global webcam_cap
    cap = cv2.VideoCapture(cam_idx)
    webcam_cap = cap
    if not cap.isOpened():
        print(f"Cannot open camera index {cam_idx}")
        glfw.terminate()
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera")
        cap.release()
        glfw.terminate()
        sys.exit(1)

    # Apply same downscale that the render loop will use
    if DOWNSCALE_WIDTH is not None:
        fh, fw = frame.shape[:2]
        if fw > DOWNSCALE_WIDTH:
            scale = DOWNSCALE_WIDTH / float(fw)
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                               interpolation=cv2.INTER_AREA)

    cam_h, cam_w = frame.shape[:2]
    print(f"Camera opened: {cam_w}x{cam_h} (index {cam_idx})")

    # Adjust rect1 aspect ratio to match camera
    cam_aspect = cam_w / cam_h
    area = P['rect1W'] * P['rect1H']
    P['rect1H'] = (area / cam_aspect) ** 0.5
    P['rect1W'] = P['rect1H'] * cam_aspect

    # Compile & link shaders
    prog = link_program(VERT_SRC, FRAG_SRC)

    # Cache uniform locations
    uniform_names = [
        'u_modelViewProjection', 'u_time',
        'u_heightScale', 'u_dripSpeed', 'u_distortion', 'u_ringCount', 'u_lightDir',
        'u_rect1Pos', 'u_rect1Size', 'u_rect1Elevation', 'u_rect1Blend',
        'u_rect1Hue', 'u_rect1Sat', 'u_rect1Bright', 'u_rect1Contrast',
        'u_rect3Pos', 'u_rect3Size', 'u_rect3Elevation', 'u_rect3Blend',
        'u_rect3Hue', 'u_rect3Sat', 'u_rect3Bright', 'u_rect3Contrast',
        'u_rect4Pos', 'u_rect4Size', 'u_rect4Elevation', 'u_rect4Blend',
        'u_rect4Hue', 'u_rect4Sat', 'u_rect4Bright', 'u_rect4Contrast',
        'u_terrainHue', 'u_terrainSat', 'u_terrainBright', 'u_terrainContrast',
        'u_webcamTex', 'u_aspectRatio',
    ]
    u = {}
    for name in uniform_names:
        u[name] = glGetUniformLocation(prog, name)

    # Attribute locations
    pos_loc = glGetAttribLocation(prog, 'a_position')
    uv_loc = glGetAttribLocation(prog, 'a_uv')

    # Create grid mesh
    print(f"Building {GRID_RES}x{GRID_RES} grid mesh (aspect {TARGET_ASPECT:.2f})...")
    positions, uvs, indices = make_grid(GRID_RES, TARGET_ASPECT)
    num_indices = len(indices)

    vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)

    vbo_uv = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv)
    glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Create webcam texture
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Upload initial frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    # ── Load logo texture ──
    logo_path = script_dir / "assets/LogoV2.png"
    logo_tex, logo_w, logo_h = load_image_texture(logo_path)

    # ── Logo quad (full width, aspect-ratio-preserving height) ──
    logo_vbo = None
    if logo_tex is not None and logo_w > 0 and logo_h > 0:
        init_fb_w, init_fb_h = glfw.get_framebuffer_size(win)
        logo_aspect = logo_w / logo_h
        screen_aspect = init_fb_w / init_fb_h
        # Full width in NDC is 2.0; compute height that preserves logo aspect ratio
        ndc_h = (screen_aspect / logo_aspect) * 2.0
        ndc_h = min(ndc_h, 2.0)  # clamp so it doesn't exceed screen
        half_h = ndc_h / 2.0
        logo_quad = np.array([
            # x     y           u    v
            -1.0, -half_h,     0.0, 0.0,
             1.0, -half_h,     1.0, 0.0,
             1.0,  half_h,     1.0, 1.0,
            -1.0,  half_h,     0.0, 1.0,
        ], dtype=np.float32)
        logo_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, logo_vbo)
        glBufferData(GL_ARRAY_BUFFER, logo_quad.nbytes, logo_quad, GL_STATIC_DRAW)

    # ── HUD overlay resources ──
    hud_prog = link_program(HUD_VERT_SRC, HUD_FRAG_SRC)
    hud_pos_loc = glGetAttribLocation(hud_prog, 'a_pos')
    hud_uv_loc = glGetAttribLocation(hud_prog, 'a_uv')
    hud_tex_loc = glGetUniformLocation(hud_prog, 'u_tex')

    # Quad: full-width strip at bottom of screen (NDC)
    hud_quad = np.array([
        # x     y     u    v
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0, -0.92, 1.0, 1.0,
        -1.0, -0.92, 0.0, 1.0,
    ], dtype=np.float32)
    hud_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, hud_vbo)
    glBufferData(GL_ARRAY_BUFFER, hud_quad.nbytes, hud_quad, GL_STATIC_DRAW)

    hud_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, hud_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Allocate initial texture
    blank = np.zeros((HUD_HEIGHT, HUD_WIDTH, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, HUD_WIDTH, HUD_HEIGHT, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank)

    # ── Overlay quad (fullscreen for logo/instructions) ──
    overlay_quad = np.array([
        # x     y     u    v
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0,  1.0,  1.0, 1.0,
        -1.0,  1.0,  0.0, 1.0,
    ], dtype=np.float32)
    overlay_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo)
    glBufferData(GL_ARRAY_BUFFER, overlay_quad.nbytes, overlay_quad, GL_STATIC_DRAW)

    # Texture for instructions text overlay
    overlay_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, overlay_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Pre-render instructions text
    instructions_img = render_overlay_text(ONBOARDING_TEXT, 1920, 1080)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, instructions_img)

    # Texture for organism text overlay
    organism_text_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, organism_text_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Pre-render organism text
    organism_text_img = render_organism_overlay(ORGANISM_TEXT, 1920, 1080)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, organism_text_img)

    # Texture for chat text overlay (dynamic - updated in render loop)
    chat_text_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, chat_text_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    # Initialize with blank texture
    blank_chat = np.zeros((1080, 1920, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank_chat)

    # Zoom & HUD state
    zoom = [1.0]
    show_hud = [False]

    def on_scroll(_win, _xoff, yoff):
        zoom[0] = max(0.5, min(3.0, zoom[0] + yoff * 0.1))

    def on_key(_win, key, _sc, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(win, True)
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            mon = glfw.get_primary_monitor()
            vid = glfw.get_video_mode(mon)
            if is_fullscreen[0]:
                glfw.set_window_monitor(win, None,
                                        windowed_pos[0], windowed_pos[1],
                                        windowed_size[0], windowed_size[1], 0)
            else:
                windowed_pos[0], windowed_pos[1] = glfw.get_window_pos(win)
                windowed_size[0], windowed_size[1] = glfw.get_window_size(win)
                glfw.set_window_monitor(win, mon, 0, 0,
                                        vid.size.width, vid.size.height,
                                        vid.refresh_rate)
            is_fullscreen[0] = not is_fullscreen[0]
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            show_hud[0] = not show_hud[0]
        if key == glfw.KEY_R and action == glfw.PRESS:
            print("Key: Reset to initial state")
            request_restart()

    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_key_callback(win, on_key)

    glEnable(GL_DEPTH_TEST)
    t0 = time.time()

    # ── Recording / motion state ──
    global latest_jpeg, manual_record_command, current_status, recording_start_time, intro_state, show_organism, show_chat, chat_messages
    global organism_text_dirty, organism_overlay_text
    OUTPUT_DIR_API.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_AUTO.mkdir(parents=True, exist_ok=True)

    bg = None
    recording = False
    seq_dir = None
    video_writer = None
    video_path = None
    image_frame_paths = []
    last_motion_time = 0.0
    last_capture_time = 0.0
    last_video_frame_time = 0.0
    img_index = 0
    last_stream_time = 0.0
    stream_interval = 1.0 / float(STREAM_FPS)
    api_triggered_recording = False
    last_cmd = None
    hud_mode = "AUTO"
    hud_state = "IDLE"
    hud_time_remaining = None
    hud_has_motion = False

    # Start Flask web server in background. Imported here (rather than at the
    # top of the file) so all module-level state is fully initialised before
    # web.py's `import herma_webcam_shader as state` resolves.
    # Make web.py's import point at this same module object (not a duplicate).
    sys.modules['herma_webcam_shader'] = sys.modules['__main__']
    import web
    threading.Thread(target=web.run_web_server, daemon=True).start()

    print(f"Web server: http://{WEB_HOST}:{WEB_PORT}/")
    print(f"API endpoints: /api/start, /api/stop, /api/auto, /api/status, /api/begin, /api/next, /api/end, /api/snapshot, /api/sentences")
    print(f"Upload URL: {AWS_UPLOAD_URL}")
    print(f"Recording timeout: {RECORDING_TIMEOUT}s, Capture interval: {CAPTURE_INTERVAL}s")
    print("Intro state: logo (waiting for /api/begin)")
    print("Running. Press ESC to quit. Tab to toggle fullscreen. Scroll to zoom.")

    # ── Render loop ──
    while not glfw.window_should_close(win):
        glfw.poll_events()

        # Get current intro state
        with intro_lock:
            current_intro_state = intro_state
        if current_intro_state != getattr(main, '_last_intro_state', None):
            print(f"Render loop: intro_state → {current_intro_state}")
            main._last_intro_state = current_intro_state

        # Get current chat state
        with chat_lock:
            current_show_chat = show_chat
            current_chat_messages = list(chat_messages)  # Make a copy

        # Get current organism state
        with organism_lock:
            current_show_organism = show_organism
            current_organism_text_dirty = organism_text_dirty
            current_organism_text = organism_overlay_text
            organism_text_dirty = False
        
        if consume_restart_request():
            print("Restarting to initial state...")

            reset_to_initial_state()

            # Reset local loop state too
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            video_path = None
            bg = None
            recording = False
            seq_dir = None
            last_motion_time = 0.0
            last_capture_time = 0.0
            last_video_frame_time = 0.0
            img_index = 0
            last_stream_time = 0.0
            api_triggered_recording = False
            last_cmd = None
            hud_mode = "AUTO"
            hud_state = "IDLE"
            hud_time_remaining = None
            hud_has_motion = False

            # Clear webcam texture (avoid stale frame)
            black_frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            black_frame = cv2.flip(black_frame, 0)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, black_frame)
    
            continue

        # Re-render organism text texture when it changes, or every frame while loading (to animate dots)
        organism_is_loading = current_show_organism and "Loading" in current_organism_text
        if current_organism_text_dirty or organism_is_loading:
            organism_text_img = render_organism_overlay(current_organism_text, 1920, 1080)
            glBindTexture(GL_TEXTURE_2D, organism_text_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, organism_text_img)

        # Clear webcam texture if chat or organism is showing
        if current_show_chat or current_show_organism:
            # Upload black frame to clear the frozen webcam image
            black_frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            black_frame = cv2.flip(black_frame, 0)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, black_frame)

        # Only process webcam if in running state AND not showing chat
        if current_intro_state == "running" and not current_show_chat:
            # Read webcam frame
            ret, frame = cap.read()
            if not ret:
                print("WARNING: cap.read() failed - webcam not delivering frames")
            if ret:
                # Downscale if needed
                if DOWNSCALE_WIDTH is not None:
                    fh, fw = frame.shape[:2]
                    if fw > DOWNSCALE_WIDTH:
                        scale = DOWNSCALE_WIDTH / float(fw)
                        frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                           interpolation=cv2.INTER_AREA)

                # ── Motion detection ──
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                has_motion = False
                if bg is None:
                    bg = gray.astype("float")
                else:
                    cv2.accumulateWeighted(gray, bg, 0.02)
                    bg_uint8 = cv2.convertScaleAbs(bg)
                    delta = cv2.absdiff(gray, bg_uint8)
                    thresh = cv2.threshold(delta, MOTION_THRESHOLD, 255,
                                           cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        if cv2.contourArea(c) >= MIN_MOTION_AREA:
                            has_motion = True
                            break

                # ── Recording state machine ──
                now = time.time()

                with control_lock:
                    cmd = manual_record_command
                    rec_start = recording_start_time

                api_start_just_called = (cmd == "start" and last_cmd != "start")
                last_cmd = cmd

                if api_start_just_called and recording and not api_triggered_recording:
                    print(f"Stopping motion recording for API recording. "
                          f"Saved {img_index} images to {seq_dir}")
                    recording = False

                if cmd == "start" and rec_start is not None:
                    if (now - rec_start) >= RECORDING_TIMEOUT:
                        print(f"Recording timeout after {RECORDING_TIMEOUT}s")
                        with control_lock:
                            manual_record_command = "stop"
                            recording_start_time = None
                        cmd = "stop"

                should_record = False
                is_api_trigger = False

                if cmd == "start":
                    should_record = True
                    is_api_trigger = True
                    last_motion_time = now
                elif cmd == "stop":
                    should_record = False
                else:
                    should_record = False

                # Start new sequence
                if should_record and not recording:
                    recording = True
                    img_index = 0
                    image_frame_paths = []
                    last_capture_time = 0.0
                    last_video_frame_time = 0.0
                    api_triggered_recording = is_api_trigger
                    RECORDING_INPUT_TYPE = fetch_recording_mode()
                    output_dir = OUTPUT_DIR_API if api_triggered_recording else OUTPUT_DIR_AUTO
                    output_dir.mkdir(parents=True, exist_ok=True)
                    ts_name = timestamp_folder_name()
                    seq_dir = output_dir / ts_name
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    trigger_type = "API" if api_triggered_recording else "MOTION"
                    if RECORDING_INPUT_TYPE == 'image_sequence':
                        video_path = None
                        video_writer = None
                        print(f"Started image sequence recording: {seq_dir} (triggered by: {trigger_type})")
                    else:
                        video_path = seq_dir / "vid.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            str(video_path), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
                        print(f"Started video recording: {video_path} (triggered by: {trigger_type})")

                # Stop sequence
                if not should_record and recording:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print(f"Stopped recording. Saved {img_index} frames.")
                    if api_triggered_recording:
                        if RECORDING_INPUT_TYPE == 'image_sequence':
                            frames_snapshot = list(image_frame_paths)
                            seq_name = seq_dir.name if seq_dir else ts_name
                            threading.Thread(target=upload_and_analyse_images,
                                             args=(frames_snapshot, seq_name), daemon=True).start()
                        elif video_path is not None:
                            threading.Thread(target=upload_video,
                                             args=(video_path,), daemon=True).start()
                    api_triggered_recording = False
                    video_path = None
                    image_frame_paths = []
                    with control_lock:
                        if manual_record_command == "stop":
                            manual_record_command = None

                # Capture frame
                if recording:
                    if RECORDING_INPUT_TYPE == 'image_sequence':
                        if (now - last_video_frame_time) >= VIDEO_FRAME_INTERVAL:
                            small = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                            frame_path = seq_dir / f"frame_{img_index:04d}.jpg"
                            cv2.imwrite(str(frame_path), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            image_frame_paths.append(str(frame_path))
                            img_index += 1
                            last_video_frame_time = now
                    elif video_writer is not None:
                        if (now - last_video_frame_time) >= VIDEO_FRAME_INTERVAL:
                            video_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                            video_writer.write(video_frame)
                            img_index += 1
                            last_video_frame_time = now

                # Update shared status
                time_remaining = None
                if rec_start is not None and cmd == "start":
                    time_remaining = max(0, RECORDING_TIMEOUT - (now - rec_start))
                with control_lock:
                    current_status = {
                        "recording": recording,
                        "video_path": str(video_path) if video_path else None,
                        "frames_captured": img_index,
                        "last_motion_time": last_motion_time,
                        "time_remaining": time_remaining,
                        "api_triggered": api_triggered_recording,
                    }

                # Update HUD state
                hud_mode = "API" if cmd == "start" else ("STOP" if cmd == "stop" else "AUTO")
                if recording:
                    hud_state = "REC (API)" if api_triggered_recording else "REC (MOTION)"
                else:
                    hud_state = "IDLE"
                hud_time_remaining = time_remaining
                hud_has_motion = has_motion

                # Encode JPEG for MJPEG stream (throttled)
                now2 = time.time()
                if (now2 - last_stream_time) >= stream_interval:
                    ok_j, buf = cv2.imencode(".jpg", frame,
                                             [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok_j:
                        with jpeg_lock:
                            latest_jpeg = buf.tobytes()
                    last_stream_time = now2

                # Draw recording indicator (blinking, anti-aliased)
                if recording:
                    fh_rec, fw_rec = frame.shape[:2]
                    # Blink: use a sine wave so the dot fades smoothly in and out
                    blink_alpha = (math.sin(now * 4.0) + 1.0) / 2.0  # 0..1, ~2 Hz
                    if blink_alpha > 0.05:
                        center = (fw_rec - 50, 50)
                        radius = 12
                        # Draw anti-aliased circle via overlay with alpha
                        overlay = frame.copy()
                        cv2.circle(overlay, center, radius, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay, blink_alpha, frame, 1.0 - blink_alpha, 0, frame)

                # Upload to GL texture
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.flip(frame_rgb, 0)
                fh_gl, fw_gl = frame_rgb.shape[:2]
                glBindTexture(GL_TEXTURE_2D, tex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fw_gl, fh_gl, 0,
                             GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

        # Viewport
        fb_w, fb_h = glfw.get_framebuffer_size(win)
        if fb_w == 0 or fb_h == 0:
            glfw.poll_events()
            continue
        glViewport(0, 0, fb_w, fb_h)

        glClearColor(0.03, 0.03, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ── Render shader (always, even in intro states) ──
        glUseProgram(prog)

        # Projection (orthographic, aspect-correct)
        aspect = fb_w / max(fb_h, 1)
        z = zoom[0]
        if aspect > 1:
            ol, or_, ob, ot = -z * aspect, z * aspect, -z, z
        else:
            ol, or_, ob, ot = -z, z, -z / aspect, z / aspect
        mvp = make_ortho(ol, or_, ob, ot, -10.0, 10.0)

        # Set all uniforms
        glUniformMatrix4fv(u['u_modelViewProjection'], 1, GL_FALSE, mvp)
        glUniform1f(u['u_time'], time.time() - t0)
        glUniform1f(u['u_heightScale'], P['heightScale'])
        glUniform1f(u['u_dripSpeed'], P['dripSpeed'])
        glUniform1f(u['u_distortion'], P['distortion'])
        glUniform1f(u['u_ringCount'], P['ringCount'])
        glUniform3f(u['u_lightDir'], 0.3, 0.3, 0.9)
        glUniform1f(u['u_aspectRatio'], TARGET_ASPECT)

        # Decide whether webcam should be visible in the shader
        webcam_visible = (current_intro_state == "running"
                          and not current_show_chat
                          and not current_show_organism)

        # Drive rect1 elevation/blend based on webcam visibility
        if webcam_visible:
            P['rect1Elev'] = 0.3
            P['rect1Blend'] = 0.25
        else:
            P['rect1Elev'] = 0.0
            P['rect1Blend'] = 0.0

        glUniform2f(u['u_rect1Pos'], P['rect1X'], P['rect1Y'])
        glUniform2f(u['u_rect1Size'], P['rect1W'], P['rect1H'])
        glUniform1f(u['u_rect1Elevation'], P['rect1Elev'])
        glUniform1f(u['u_rect1Blend'], P['rect1Blend'])
        glUniform1f(u['u_rect1Hue'], P['rect1Hue'])
        glUniform1f(u['u_rect1Sat'], P['rect1Sat'])
        glUniform1f(u['u_rect1Bright'], P['rect1Bright'])
        glUniform1f(u['u_rect1Contrast'], P['rect1Contrast'])

        glUniform2f(u['u_rect3Pos'], P['rect3X'], P['rect3Y'])
        glUniform2f(u['u_rect3Size'], P['rect3W'], P['rect3H'])
        glUniform1f(u['u_rect3Elevation'], P['rect3Elev'])
        glUniform1f(u['u_rect3Blend'], P['rect3Blend'])
        glUniform1f(u['u_rect3Hue'], P['rect3Hue'])
        glUniform1f(u['u_rect3Sat'], P['rect3Sat'])
        glUniform1f(u['u_rect3Bright'], P['rect3Bright'])
        glUniform1f(u['u_rect3Contrast'], P['rect3Contrast'])

        glUniform2f(u['u_rect4Pos'], P['rect4X'], P['rect4Y'])
        glUniform2f(u['u_rect4Size'], P['rect4W'], P['rect4H'])
        glUniform1f(u['u_rect4Elevation'], P['rect4Elev'])
        glUniform1f(u['u_rect4Blend'], P['rect4Blend'])
        glUniform1f(u['u_rect4Hue'], P['rect4Hue'])
        glUniform1f(u['u_rect4Sat'], P['rect4Sat'])
        glUniform1f(u['u_rect4Bright'], P['rect4Bright'])
        glUniform1f(u['u_rect4Contrast'], P['rect4Contrast'])

        glUniform1f(u['u_terrainHue'], P['terrainHue'])
        glUniform1f(u['u_terrainSat'], P['terrainSat'])
        glUniform1f(u['u_terrainBright'], P['terrainBright'])
        glUniform1f(u['u_terrainContrast'], P['terrainContrast'])

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(u['u_webcamTex'], 0)

        # Bind vertex data and draw
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_uv)
        glEnableVertexAttribArray(uv_loc)
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)

        # ── Render intro overlays ──
        if current_intro_state == "logo" and logo_tex is not None and logo_vbo is not None:
            # Draw logo overlay (aspect-ratio-preserving)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)

            glUseProgram(hud_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, logo_tex)
            glUniform1i(hud_tex_loc, 0)

            glBindBuffer(GL_ARRAY_BUFFER, logo_vbo)
            glEnableVertexAttribArray(hud_pos_loc)
            glVertexAttribPointer(hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(hud_uv_loc)
            glVertexAttribPointer(hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glDisableVertexAttribArray(hud_pos_loc)
            glDisableVertexAttribArray(hud_uv_loc)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

        elif current_intro_state == "instructions":
            # Draw instructions overlay
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)

            glUseProgram(hud_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, overlay_tex)
            glUniform1i(hud_tex_loc, 0)

            glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo)
            glEnableVertexAttribArray(hud_pos_loc)
            glVertexAttribPointer(hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(hud_uv_loc)
            glVertexAttribPointer(hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glDisableVertexAttribArray(hud_pos_loc)
            glDisableVertexAttribArray(hud_uv_loc)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

        # ── Draw organism display (when stop is called) ──
        if current_show_organism:
            # Draw organism text overlay (fullscreen, semi-transparent)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)

            glUseProgram(hud_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, organism_text_tex)
            glUniform1i(hud_tex_loc, 0)

            glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo)
            glEnableVertexAttribArray(hud_pos_loc)
            glVertexAttribPointer(hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(hud_uv_loc)
            glVertexAttribPointer(hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glDisableVertexAttribArray(hud_pos_loc)
            glDisableVertexAttribArray(hud_uv_loc)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

        # ── Draw chat display (when /api/chat is called) ──
        if current_show_chat and current_chat_messages:
            # Update chat text texture with current messages
            chat_img = render_chat_messages(current_chat_messages, 1920, 1080)
            glBindTexture(GL_TEXTURE_2D, chat_text_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, chat_img)

            # Draw chat text overlay (fullscreen, semi-transparent)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)

            glUseProgram(hud_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, chat_text_tex)
            glUniform1i(hud_tex_loc, 0)

            glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo)
            glEnableVertexAttribArray(hud_pos_loc)
            glVertexAttribPointer(hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(hud_uv_loc)
            glVertexAttribPointer(hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glDisableVertexAttribArray(hud_pos_loc)
            glDisableVertexAttribArray(hud_uv_loc)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

        # ── Draw HUD overlay (only in running state) ──
        if current_intro_state == "running" and show_hud[0]:
            hud_img = render_hud_text(hud_mode, hud_state, img_index,
                                      hud_time_remaining, hud_has_motion)
            glBindTexture(GL_TEXTURE_2D, hud_tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, HUD_WIDTH, HUD_HEIGHT,
                            GL_RGBA, GL_UNSIGNED_BYTE, hud_img)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)

            glUseProgram(hud_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, hud_tex)
            glUniform1i(hud_tex_loc, 0)

            glBindBuffer(GL_ARRAY_BUFFER, hud_vbo)
            glEnableVertexAttribArray(hud_pos_loc)
            glVertexAttribPointer(hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(hud_uv_loc)
            glVertexAttribPointer(hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glDisableVertexAttribArray(hud_pos_loc)
            glDisableVertexAttribArray(hud_uv_loc)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

        glfw.swap_buffers(win)

    # Cleanup
    cap.release()
    glfw.terminate()
    print("Done.")


if __name__ == "__main__":
    main()