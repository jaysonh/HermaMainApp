"""Per-frame capture state machine.

``RecordingMachine`` owns all per-capture transient state: motion
detection background, the output directory for the current run, and HUD
output. The render loop calls ``step(frame, now)`` once per webcam frame;
the machine takes care of running motion detection and advancing the
start/stop state machine.

Nothing is written while the capture window is open — the experience is a
single still. When the window closes (the [COMPLETE] button hitting
``/api/stop``, or the timeout) the most recent frame is saved as one JPEG
and handed to the upload thread.
"""

import math
import threading
from datetime import datetime

import cv2

from . import config, remote, state


def _timestamp_folder_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class RecordingMachine:
    def __init__(self):
        # Motion detection
        self.bg = None

        # Capture state
        self.recording = False
        self.seq_dir = None
        self.capture_path = None
        self.last_frame = None          # most recent frame, saved on stop
        self.api_triggered_recording = False
        self.last_cmd = None

        # Timers / counters
        self.last_motion_time = 0.0
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
        self.capture_path = None
        self.last_frame = None
        self.bg = None
        self.recording = False
        self.seq_dir = None
        self.last_motion_time = 0.0
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
        The frame is stashed as ``self.last_frame`` so ``_stop_recording`` can
        save it — the state machine is advanced *after* the stash, so a stop
        arriving on this frame captures this frame.
        """
        frame = self._downscale(frame)
        self.last_frame = frame
        has_motion = self._detect_motion(frame)
        self._advance_state_machine(now)
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
            print(f"Abandoning motion capture window {self.seq_dir} for an API one")
            self.recording = False

        # Timeout
        if cmd == "start" and rec_start is not None:
            if (now - rec_start) >= state.RECORDING_TIMEOUT:
                print(f"Recording timeout after {state.RECORDING_TIMEOUT}s")
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
        self.capture_path = None
        self.api_triggered_recording = is_api_trigger

        output_dir = config.OUTPUT_DIR_API if self.api_triggered_recording else config.OUTPUT_DIR_AUTO
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_name = _timestamp_folder_name()
        self.seq_dir = output_dir / ts_name
        self.seq_dir.mkdir(parents=True, exist_ok=True)

        trigger_type = "API" if self.api_triggered_recording else "MOTION"
        print(f"Capture window open: {self.seq_dir} (triggered by: {trigger_type})")

    def _stop_recording(self):
        self.recording = False
        self.capture_path = self._save_capture()
        if self.capture_path is None:
            print("Capture window closed, but no webcam frame was available")
        else:
            print(f"Captured still: {self.capture_path}")

        # Publish it for the sentences page to show beside the text.
        state.set_last_recording(str(self.capture_path) if self.capture_path else None)

        if self.api_triggered_recording and self.capture_path is not None:
            threading.Thread(
                target=remote.upload_and_analyse_capture,
                args=(self.capture_path,),
                daemon=True,
            ).start()

        self.api_triggered_recording = False
        with state.control_lock:
            if state.manual_record_command == "stop":
                state.manual_record_command = None

    def _save_capture(self):
        """Write the most recent webcam frame to ``<seq_dir>/capture.jpg``."""
        if self.last_frame is None or self.seq_dir is None:
            return None
        frame = self.last_frame
        fh, fw = frame.shape[:2]
        if fw > config.CAPTURE_MAX_WIDTH:
            scale = config.CAPTURE_MAX_WIDTH / float(fw)
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                               interpolation=cv2.INTER_AREA)
        path = self.seq_dir / "capture.jpg"
        if not cv2.imwrite(str(path), frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), config.CAPTURE_JPEG_QUALITY]):
            print(f"Failed to write capture: {path}")
            return None
        self.img_index = 1
        return path

    def _update_status(self, now):
        time_remaining = None
        if self._rec_start is not None and self._cmd == "start":
            time_remaining = max(0, state.RECORDING_TIMEOUT - (now - self._rec_start))

        with state.control_lock:
            state.current_status = {
                "recording": self.recording,
                "capture_path": str(self.capture_path) if self.capture_path else None,
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
            self.hud_state = "CAP (API)" if self.api_triggered_recording else "CAP (MOTION)"
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
