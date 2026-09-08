#!/usr/bin/env python3
"""Herma Webcam Shader — entry point.

The implementation lives in the ``herma`` package:

    herma.config       Static constants and TOML loading helpers
    herma.state        Shared mutable state (locks + globals)
    herma.overlays     PIL text rendering for HUD/instructions/organism/chat
    herma.gl_utils     Shader / mesh / texture helpers
    herma.recording    Recording state machine (motion + capture)
    herma.remote       HTTP/SSE pipeline to HermaSentiChat
    herma.render_loop  GLFW window + GL setup + main render loop
    herma.web          Flask routes (one blueprint per concern)

Requirements:
    pip install glfw PyOpenGL opencv-python numpy flask flask-cors requests Pillow
    (If Python < 3.11) pip install tomli

Controls:
    ESC    Quit
    Tab    Toggle fullscreen / windowed
    Space  Toggle HUD status bar
    R      Restart to logo state
    Scroll Zoom in/out
"""

import argparse
import os
import sys
import threading
from pathlib import Path

import glfw

from herma import config, state, recording, render_loop
from herma.web import run_web_server


def _parse_args(script_dir: Path):
    parser = argparse.ArgumentParser(description="Herma Webcam Shader")
    parser.add_argument("--config", type=str, default=str(script_dir / "config.toml"),
                        help="Path to config.toml")
    parser.add_argument("camera_index", nargs="?", default=None,
                        help="Optional camera index override (int)")
    return parser.parse_args()


def _resolve_cam_idx(cfg, args):
    cam_idx = int(config._deep_get(cfg, "video", "cam_index", default=config.CAM_INDEX))
    if args.camera_index is not None:
        try:
            cam_idx = int(args.camera_index)
        except ValueError:
            print(f"Usage: {sys.argv[0]} [--config path/to/config.toml] [camera_index]")
            sys.exit(1)
    return cam_idx


def main():
    script_dir = Path(__file__).resolve().parent

    args = _parse_args(script_dir)
    cfg = config.load_config_toml(Path(args.config))

    cam_idx = _resolve_cam_idx(cfg, args)
    display_cfg = config._deep_get(cfg, "display", default={}) or {}
    herma_host = str(config._deep_get(cfg, "herma_server", "host", default="10.142.77.6"))
    herma_port = int(config._deep_get(cfg, "herma_server", "port", default=5009))

    end_video_path = config._deep_get(cfg, "sentences", "video_path",
                                       default=config.END_VIDEO_PATH)
    state.END_VIDEO_PATH = str(Path(str(end_video_path)).expanduser())

    state.AWS_UPLOAD_URL = os.environ.get(
        "AWS_UPLOAD_URL",
        f"http://{herma_host}:{herma_port}/api/upload",
    )

    print(f"Config: {Path(args.config)}")
    print(f"Camera index: {cam_idx}")
    print(f"Herma server: {herma_host}:{herma_port}")
    print(f"Upload URL: {state.AWS_UPLOAD_URL}")
    print(f"Sentences background video: {state.END_VIDEO_PATH}")

    win, cap, frame, cam_w, cam_h = render_loop.bootstrap(display_cfg, cam_idx)
    machine = recording.RecordingMachine()

    threading.Thread(target=run_web_server, daemon=True).start()
    print(f"Web server: http://{config.WEB_HOST}:{config.WEB_PORT}/")
    print(f"Recording timeout: {config.RECORDING_TIMEOUT}s, "
          f"Capture interval: {config.CAPTURE_INTERVAL}s")

    try:
        render_loop.run(win, cap, frame, cam_w, cam_h, machine)
    finally:
        cap.release()
        glfw.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
