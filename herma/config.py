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

# ─── Capture / Motion ───────────────────────────────────────────────────────

MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 2500

# How long the capture window stays open before the still is taken anyway.
# Default only — overridden from config.toml [recording].timeout_seconds; read
# it as state.RECORDING_TIMEOUT, never from here.
RECORDING_TIMEOUT_DEFAULT = 30.0

DOWNSCALE_WIDTH = 1920
OUTPUT_DIR_API = Path("recorded")
OUTPUT_DIR_AUTO = Path("tmp")

# The still captured when [COMPLETE] is pressed. It is the only input the
# organ analysis gets, so it is kept large — downscaled just enough to keep
# the upload quick.
CAPTURE_MAX_WIDTH = 1280
CAPTURE_JPEG_QUALITY = 90

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
    "The objects in front of you are replicas of human internal organs.\n"
    "\n"
    "Create your own anatomy by choosing and arranging the organs on the "
    "table. You may use as many or as few as you wish.\n"
    "\n"
    "Press [NEW ANATOMY] to begin.\n"
    "\n"
    "When you are satisfied with your creation press [COMPLETE]"
)
ORGANISM_TEXT = ""

# Shown from /api/stop until the analysis comes back. Drawn large and centred,
# sized to nearly fill the screen width — see overlays.render_organism_overlay.
LOADING_TEXT = "LOADING ORGANISM"
LOADING_WIDTH_FRACTION = 0.88
# Outline width in pixels for that one screen. The shared rule scales the
# stroke with the font size, which is far too heavy at ~150px.
LOADING_OUTLINE_WIDTH = 3

# ─── Asset Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONT_PATH = str(ASSETS_DIR / "sylfaen.ttf")
# Every on-screen overlay — instructions, organism (sentences, thank-you
# message, organism name + description) and chat bubbles — uses Cascadia Code.
# Only the HUD status bar still uses FONT_PATH.
OVERLAY_FONT_PATH = str(ASSETS_DIR / "CascadiaCode-Regular.ttf")
# The instructions screen is set in the bold face instead.
OVERLAY_FONT_BOLD_PATH = str(ASSETS_DIR / "CascadiaCode-Bold.ttf")
# Those overlays draw black text with a white outline.
TEXT_FILL = (0, 0, 0, 255)
TEXT_OUTLINE = (255, 255, 255, 255)
LOGO_PATH = ASSETS_DIR / "LogoV2.png"

# Played behind the sentences page instead of the terrain shader. Kept as a
# path relative to the project root so it resolves on Windows (D:\herma\...)
# and WSL (/mnt/d/herma/...) alike; override with [sentences].video_path.
END_VIDEO_PATH = str(PROJECT_ROOT.parent / "PlatesBlackButtons" / "Screen" / "BlankEndVideo.mp4")

# How long the finished page stays up before the thank-you message.
SENTENCES_PAGE_HOLD = 8.0

# The recording plays in the top-right of the sentences page, with the text
# laid out around it. Fractions of the 1920x1080 page; the height follows from
# the recording's own aspect ratio.
INSET_WIDTH_FRACTION = 0.44
INSET_RIGHT_MARGIN_FRACTION = 0.04
INSET_TOP_FRACTION = 0.05
INSET_GUTTER = 40          # px of clear space between the text and the inset
INSET_FALLBACK_ASPECT = 4 / 3
# How far past rect1 the shader panel is framed, as a fraction of its size.
# Room is needed outside rect1 for the edge to fade out in, so this is positive
# again — the terrain that would show there is what the feather removes.
INSET_SHADER_MARGIN = 0.05
# Width of that fade, in uv units of the shader grid.
INSET_SHADER_FEATHER = 0.05

# Typed out the same way as the sentences, once the page is done.
THANK_YOU_TEXT = "Thank you for your experience"
THANK_YOU_HOLD = 10.0

# Typing speed of the instructions screen, in characters per second.
INSTRUCTIONS_CPS = 25.0
# Its outline width in pixels. The shared rule scales the stroke with the font
# size, which leaves this screen thin now that it auto-fits smaller.
INSTRUCTIONS_OUTLINE_WIDTH = 3


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
