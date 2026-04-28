"""Per-frame recording state machine.

``RecordingMachine`` owns all per-recording transient state: motion
detection background, the active video writer or image-sequence
directory, frame counters, and HUD output. The render loop calls
``step(frame, now)`` once per webcam frame; the machine takes care of
running motion detection, advancing the start/stop state machine,
writing frames to disk, and (on stop) kicking off the upload thread.
"""

import math
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config, remote, state


def _timestamp_folder_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class RecordingMachine:
    def __init__(self):
        # Motion detection
        self.bg = None

        # Recording state
        self.recording = False
        self.seq_dir = None
        self.video_writer = None
        self.video_path = None
        self.image_frame_paths = []
        self.api_triggered_recording = False
        self.last_cmd = None

        # Timers / counters
        self.last_motion_time = 0.0
        self.last_capture_time = 0.0
        self.last_video_frame_time = 0.0
        self.img_index = 0
        self.last_stream_time = 0.0
        self.stream_interval = 1.0 / float(config.STREAM_FPS)

        # Output for HUD
        self.hud_mode = "AUTO"
        self.hud_state = "IDLE"
        self.hud_time_remaining = None
        self.hud_has_motion = False

        # Output dirs
        config.OUTPUT_DIR_API.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR_AUTO.mkdir(parents=True, exist_ok=True)

    def reset(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.video_path = None
        self.bg = None
        self.recording = False
        self.seq_dir = None
        self.image_frame_paths = []
        self.last_motion_time = 0.0
        self.last_capture_time = 0.0
        self.last_video_frame_time = 0.0
        self.img_index = 0
        self.last_stream_time = 0.0
        self.api_triggered_recording = False
        self.last_cmd = None
        self.hud_mode = "AUTO"
        self.hud_state = "IDLE"
        self.hud_time_remaining = None
        self.hud_has_motion = False

    # ── Per-frame entrypoint ────────────────────────────────────────────────

    def step(self, frame, now):
        """Advance the recording state machine for one webcam frame.

        Returns the (possibly modified) BGR frame to upload to the GL texture.
        Side effects: writes to ``state.latest_jpeg`` and ``state.current_status``.
        """
        frame = self._downscale(frame)
        has_motion = self._detect_motion(frame)
        self._advance_state_machine(now)
        if self.recording:
            self._capture_frame(frame, now)
        self._update_status(now)
        self._update_hud(has_motion)
        self._encode_stream_jpeg(frame, now)
        if self.recording:
            frame = self._draw_recording_indicator(frame, now)
        return frame

    # ── Internals ───────────────────────────────────────────────────────────

    def _downscale(self, frame):
        if config.DOWNSCALE_WIDTH is None:
            return frame
        fh, fw = frame.shape[:2]
        if fw <= config.DOWNSCALE_WIDTH:
            return frame
        scale = config.DOWNSCALE_WIDTH / float(fw)
        return cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                          interpolation=cv2.INTER_AREA)

    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.bg is None:
            self.bg = gray.astype("float")
            return False

        cv2.accumulateWeighted(gray, self.bg, 0.02)
        bg_uint8 = cv2.convertScaleAbs(self.bg)
        delta = cv2.absdiff(gray, bg_uint8)
        thresh = cv2.threshold(delta, config.MOTION_THRESHOLD, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= config.MIN_MOTION_AREA:
                return True
        return False

    def _advance_state_machine(self, now):
        with state.control_lock:
            cmd = state.manual_record_command
            rec_start = state.recording_start_time

        api_start_just_called = (cmd == "start" and self.last_cmd != "start")
        self.last_cmd = cmd

        if api_start_just_called and self.recording and not self.api_triggered_recording:
            print(f"Stopping motion recording for API recording. "
                  f"Saved {self.img_index} images to {self.seq_dir}")
            self.recording = False

        # Timeout
        if cmd == "start" and rec_start is not None:
            if (now - rec_start) >= config.RECORDING_TIMEOUT:
                print(f"Recording timeout after {config.RECORDING_TIMEOUT}s")
                with state.control_lock:
                    state.manual_record_command = "stop"
                    state.recording_start_time = None
                cmd = "stop"

        should_record = False
        is_api_trigger = False
        if cmd == "start":
            should_record = True
            is_api_trigger = True
            self.last_motion_time = now

        # Start
        if should_record and not self.recording:
            self._start_recording(is_api_trigger)

        # Stop
        if not should_record and self.recording:
            self._stop_recording()

        # Cache the rec_start so _update_status can compute time_remaining
        self._cmd = cmd
        self._rec_start = rec_start

    def _start_recording(self, is_api_trigger):
        self.recording = True
        self.img_index = 0
        self.image_frame_paths = []
        self.last_capture_time = 0.0
        self.last_video_frame_time = 0.0
        self.api_triggered_recording = is_api_trigger
        state.RECORDING_INPUT_TYPE = remote.fetch_recording_mode()

        output_dir = config.OUTPUT_DIR_API if self.api_triggered_recording else config.OUTPUT_DIR_AUTO
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_name = _timestamp_folder_name()
        self.seq_dir = output_dir / ts_name
        self.seq_dir.mkdir(parents=True, exist_ok=True)

        trigger_type = "API" if self.api_triggered_recording else "MOTION"
        if state.RECORDING_INPUT_TYPE == 'image_sequence':
            self.video_path = None
            self.video_writer = None
            print(f"Started image sequence recording: {self.seq_dir} (triggered by: {trigger_type})")
        else:
            self.video_path = self.seq_dir / "vid.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.video_path), fourcc, config.VIDEO_FPS,
                (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            print(f"Started video recording: {self.video_path} (triggered by: {trigger_type})")

    def _stop_recording(self):
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        print(f"Stopped recording. Saved {self.img_index} frames.")

        if self.api_triggered_recording:
            if state.RECORDING_INPUT_TYPE == 'image_sequence':
                frames_snapshot = list(self.image_frame_paths)
                seq_name = self.seq_dir.name if self.seq_dir else _timestamp_folder_name()
                threading.Thread(
                    target=remote.upload_and_analyse_images,
                    args=(frames_snapshot, seq_name),
                    daemon=True,
                ).start()
            elif self.video_path is not None:
                threading.Thread(
                    target=remote.upload_video,
                    args=(self.video_path,),
                    daemon=True,
                ).start()

        self.api_triggered_recording = False
        self.video_path = None
        self.image_frame_paths = []
        with state.control_lock:
            if state.manual_record_command == "stop":
                state.manual_record_command = None

    def _capture_frame(self, frame, now):
        if state.RECORDING_INPUT_TYPE == 'image_sequence':
            if (now - self.last_video_frame_time) >= config.VIDEO_FRAME_INTERVAL:
                small = cv2.resize(frame, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
                frame_path = self.seq_dir / f"frame_{self.img_index:04d}.jpg"
                cv2.imwrite(str(frame_path), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                self.image_frame_paths.append(str(frame_path))
                self.img_index += 1
                self.last_video_frame_time = now
        elif self.video_writer is not None:
            if (now - self.last_video_frame_time) >= config.VIDEO_FRAME_INTERVAL:
                video_frame = cv2.resize(frame, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
                self.video_writer.write(video_frame)
                self.img_index += 1
                self.last_video_frame_time = now

    def _update_status(self, now):
        time_remaining = None
        if self._rec_start is not None and self._cmd == "start":
            time_remaining = max(0, config.RECORDING_TIMEOUT - (now - self._rec_start))

        with state.control_lock:
            state.current_status = {
                "recording": self.recording,
                "video_path": str(self.video_path) if self.video_path else None,
                "frames_captured": self.img_index,
                "last_motion_time": self.last_motion_time,
                "time_remaining": time_remaining,
                "api_triggered": self.api_triggered_recording,
            }

        self.hud_time_remaining = time_remaining

    def _update_hud(self, has_motion):
        cmd = self._cmd
        if cmd == "start":
            self.hud_mode = "API"
        elif cmd == "stop":
            self.hud_mode = "STOP"
        else:
            self.hud_mode = "AUTO"

        if self.recording:
            self.hud_state = "REC (API)" if self.api_triggered_recording else "REC (MOTION)"
        else:
            self.hud_state = "IDLE"

        self.hud_has_motion = has_motion

    def _encode_stream_jpeg(self, frame, now):
        if (now - self.last_stream_time) < self.stream_interval:
            return
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if ok:
            with state.jpeg_lock:
                state.latest_jpeg = buf.tobytes()
        self.last_stream_time = now

    def _draw_recording_indicator(self, frame, now):
        # Sine-wave fade so the dot pulses ~2 Hz.
        blink_alpha = (math.sin(now * 4.0) + 1.0) / 2.0
        if blink_alpha <= 0.05:
            return frame
        fh, fw = frame.shape[:2]
        center = (fw - 50, 50)
        radius = 12
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, blink_alpha, frame, 1.0 - blink_alpha, 0, frame)
        return frame
