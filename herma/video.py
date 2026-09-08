"""Background video playback for the sentences page.

While the generated sentences are on screen the terrain shader is hidden and
this plays a video behind the text instead. Decoding happens on the render
thread, one frame at a time, paced by the file's own frame rate — a full-file
preload would cost hundreds of MB for a 1080p clip.
"""

from pathlib import Path

import cv2


class BackgroundVideo:
    """A looping video source that hands the render loop one frame at a time.

    ``poll(now)`` returns an RGB frame only when the next one is due, and None
    otherwise, so the caller keeps re-using the texture it already uploaded.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.cap = None
        self.fps = 30.0
        self._next_due = 0.0

    @property
    def active(self):
        return self.cap is not None

    def start(self, now):
        if self.cap is not None:
            return True
        if not self.path.exists():
            print(f"Background video not found: {self.path}")
            return False

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            print(f"Could not open background video: {self.path}")
            cap.release()
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1.0 else 30.0
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
