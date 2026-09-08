"""Static configuration: numeric constants, default shader parameters,
asset paths, and TOML loading helpers. Anything mutated at runtime lives
in ``herma.state`` instead.
"""

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli

import glfw

# ─── Window / Render ────────────────────────────────────────────────────────

CAM_INDEX = 1
WINDOW_W = 1280
WINDOW_H = 720
GRID_RES = 400
TARGET_ASPECT = 16.0 / 9.0

# ─── Default Shader Parameters ──────────────────────────────────────────────

# rect3 is the top green square, rect4 is the bottom purple square.
# rect1 is where the webcam texture is composited.
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

# ─── Recording / Motion ─────────────────────────────────────────────────────

MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 2500

CAPTURE_INTERVAL = 0.5
RECORDING_TIMEOUT = 30.0
STILL_SECONDS_TO_STOP = 5.0

DOWNSCALE_WIDTH = 1920
OUTPUT_DIR_API = Path("recorded")
OUTPUT_DIR_AUTO = Path("tmp")

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 10
VIDEO_FRAME_INTERVAL = 1.0 / VIDEO_FPS

# ─── Web Server ─────────────────────────────────────────────────────────────

WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
STREAM_FPS = 10
JPEG_QUALITY = 80

# ─── HUD ────────────────────────────────────────────────────────────────────

HUD_WIDTH = 800
HUD_HEIGHT = 36

# ─── Chat ───────────────────────────────────────────────────────────────────

MAX_CHAT_MESSAGES = 10

# ─── Onboarding / Organism Text ─────────────────────────────────────────────

ONBOARDING_TEXT = (
    "The objects in front of you are precise replicas of human internal organs."
    "You are invited to make your own arrangement, using as many or as few as you wish."
    "Pick up the first organ to start creating a new anatomy."
)
ORGANISM_TEXT = ""

# ─── Asset Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONT_PATH = str(ASSETS_DIR / "sylfaen.ttf")
# Every on-screen overlay — instructions, organism (sentences, thank-you
# message, organism name + description) and chat bubbles — uses Cascadia Code.
# Only the HUD status bar still uses FONT_PATH.
OVERLAY_FONT_PATH = str(ASSETS_DIR / "CascadiaCode-Regular.ttf")
# Those overlays draw black text with a white outline.
TEXT_FILL = (0, 0, 0, 255)
TEXT_OUTLINE = (255, 255, 255, 255)
LOGO_PATH = ASSETS_DIR / "LogoV2.png"


# ─── TOML Loading ───────────────────────────────────────────────────────────

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
    """Pick a GLFW monitor based on [display] settings.

    Uses ``display_primary_monitor`` (bool) and ``monitor_index`` (0-based int).
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
