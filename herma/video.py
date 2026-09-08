"""Sources that feed a GL texture on the sentences page.

``BackgroundVideo`` plays the looping clip that replaces the terrain shader
behind the text. Decoding happens on the render thread, one frame at a time,
paced by the file's own frame rate — a full-file preload would cost hundreds
of MB for a 1080p clip.

``StillImage`` wears the same interface but holds a single frame: it is what
the preview area shows, since the run captures one screenshot rather than a
video.
"""

from pathlib import Path

import cv2


class BackgroundVideo:
    """A looping video source that hands the render loop one frame at a time.

    ``poll(now)`` returns an RGB frame only when the next one is due, and None
    otherwise, so the caller keeps re-using the texture it already uploaded.
    """

    def __init__(self, path, default_fps=30.0):
        # expanduser so a config.toml can use "~/herma-assets/clip.mp4" — the
        # shell isn't involved in reading that value, so nothing else expands it.
        # A path may also be a printf-style pattern for an image sequence.
        self.path = Path(path).expanduser()
        self.cap = None
        self.is_sequence = "%" in self.path.name
        self.default_fps = float(default_fps)
        self.fps = self.default_fps
        self.aspect = None   # width / height, known once the file is open
        self._next_due = 0.0

    @property
    def active(self):
        return self.cap is not None

    def start(self, now):
        if self.cap is not None:
            return True
        if not self.is_sequence and not self.path.exists():
            print(f"Background video not found: {self.path}")
            return False

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            print(f"Could not open background video: {self.path}")
            cap.release()
            return False

        # An image sequence has no frame rate of its own — OpenCV invents one
        # (25) rather than reporting 0 — and some containers report nonsense.
        # Either way, use the rate the caller knows it was captured at so
        # playback runs at real time instead of racing.
        fps = cap.get(cv2.CAP_PROP_FPS)
        if self.is_sequence or not fps or fps <= 1.0:
            self.fps = self.default_fps
        else:
            self.fps = fps
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspect = (w / h) if w and h else None
        self.cap = cap
        self._next_due = now
        print(f"Background video playing: {self.path.name} ({self.fps:.0f} fps)")
        return True

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Background video stopped")

    def poll(self, now):
        """Return the next RGB frame if one is due, else None."""
        if self.cap is None or now < self._next_due:
            return None

        ok, frame = self.cap.read()
        if not ok:
            # End of file — rewind and keep going.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                print("Background video could not be rewound — stopping")
                self.stop()
                return None

        # Never let a slow decode build up a backlog of overdue frames.
        self._next_due = max(now, self._next_due + 1.0 / self.fps)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.flip(frame, 0)  # GL quads sample with v=0 at the bottom


class StillImage:
    """A single image behind the same interface as ``BackgroundVideo``.

    ``poll`` hands the frame over once and returns None thereafter — the render
    loop keeps re-using the texture it already uploaded, so the still just sits
    there.
    """

    def __init__(self, path, default_fps=None):
        self.path = Path(path).expanduser()
        self.frame = None
        self.aspect = None
        self._delivered = False

    @property
    def active(self):
        return self.frame is not None

    def start(self, now):
        if self.frame is not None:
            return True
        if not self.path.exists():
            print(f"Capture image not found: {self.path}")
            return False

        img = cv2.imread(str(self.path))
        if img is None:
            print(f"Could not read capture image: {self.path}")
            return False

        h, w = img.shape[:2]
        self.aspect = (w / h) if h else None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame = cv2.flip(img, 0)  # GL quads sample with v=0 at the bottom
        self._delivered = False
        print(f"Capture image shown: {self.path.name} ({w}x{h})")
        return True

    def stop(self):
        if self.frame is not None:
            self.frame = None
            self._delivered = False
            print("Capture image cleared")

    def poll(self, now):
        """Return the frame the first time it is asked for, then None."""
        if self.frame is None or self._delivered:
            return None
        self._delivered = True
        return self.frame
