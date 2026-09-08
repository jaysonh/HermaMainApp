"""PIL-based overlay rendering: HUD, instructions, organism, chat.

Each function returns a numpy array (uint8 RGBA) that has already been
flipped vertically so it can be uploaded directly to a GL texture.
"""

import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config


def _load_font(size, path=None):
    path = path or config.FONT_PATH
    try:
        return ImageFont.truetype(path, size)
    except (IOError, OSError):
        print(f"Warning: Could not load {path}, falling back to default font")
        return ImageFont.load_default()


def _draw_outlined_text(draw, xy, text, font, anchor=None):
    """Draw black text with a white outline — the look shared by every Cascadia overlay.

    The stroke is scaled off the font size so it stays proportional from the
    22px chat bubbles up to the 64px instructions screen.
    """
    draw.text(
        xy, text, font=font,
        fill=config.TEXT_FILL,
        stroke_width=max(1, round(font.size / 20)),
        stroke_fill=config.TEXT_OUTLINE,
        anchor=anchor,
    )


def _animated_dots():
    """Return 1-3 dots cycling based on current time."""
    return "." * (int(time.time() * 2) % 3 + 1)


def render_hud_text(mode_str, rec_state, img_index, time_remaining, has_motion):
    pil_img = Image.new("RGBA", (config.HUD_WIDTH, config.HUD_HEIGHT), (0, 0, 0, 0))
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
    img = cv2.flip(img, 0)
    return img


# Font size for the loading screen depends only on the text and the screen
# width, but the overlay is re-rendered every frame to animate the dots — so it
# is worked out once and kept.
_loading_font_cache = {}


def _fit_font_to_width(text, target_w, draw):
    """Largest font size whose rendered ``text`` still fits ``target_w``."""
    size = 24
    best = size
    while size <= 400:
        font = _load_font(size, config.OVERLAY_FONT_PATH)
        if draw.textlength(text, font=font) > target_w:
            break
        best = size
        size += 4
    return best


def _render_loading_overlay(draw, width, height):
    """Draw the loading message big and centred, animated dots trailing it.

    The size is fitted to the message *plus* three dots, and the whole block is
    centred on that width, so the dots always fit on screen and the message
    holds still instead of shuffling as they cycle.
    """
    base = config.LOADING_TEXT
    key = (base, width)
    cached = _loading_font_cache.get(key)
    if cached is None:
        target_w = width * config.LOADING_WIDTH_FRACTION
        font = _load_font(_fit_font_to_width(base + "...", target_w, draw),
                          config.OVERLAY_FONT_PATH)
        base_w = draw.textlength(base, font=font)
        full_w = draw.textlength(base + "...", font=font)
        _loading_font_cache.clear()
        _loading_font_cache[key] = cached = (font, base_w, full_w)
    font, base_w, full_w = cached

    bbox = draw.textbbox((0, 0), base, font=font)
    x = (width - full_w) / 2
    y = (height - (bbox[3] + bbox[1])) / 2

    _draw_outlined_text(draw, (x, y), base, font)
    _draw_outlined_text(draw, (x + base_w, y), _animated_dots(), font)


def render_organism_overlay(text, width, height):
    """Render organism name + description with a grey translucent panel sized to fit the text."""
    pil_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)

    if text.strip() == config.LOADING_TEXT:
        _render_loading_overlay(draw, width, height)
        return cv2.flip(np.array(pil_img, dtype=np.uint8), 0)

    title_font = _load_font(44, config.OVERLAY_FONT_PATH)
    body_font = _load_font(26, config.OVERLAY_FONT_PATH)

    panel_padding = 30
    line_height = 38
    title_line_height = 56
    title_bottom_gap = 20

    max_text_area_w = width - panel_padding * 4

    parts = text.strip().split('\n', 1)
    title = parts[0].strip() if parts else ""
    description = parts[1].strip() if len(parts) > 1 else ""

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

    content_h = 0
    if title_lines:
        content_h += len(title_lines) * title_line_height + title_bottom_gap
    content_h += len(body_lines) * line_height

    panel_h = content_h + panel_padding * 2
    panel_w = max_text_area_w + panel_padding * 2

    # No panel is drawn any more — the white outline carries the text over
    # whatever is behind it — but its box still positions the block.
    panel_x = (width - panel_w) // 2
    panel_y = (height - panel_h) // 2

    text_x = panel_x + panel_padding
    text_y = panel_y + panel_padding

    for tl in title_lines:
        _draw_outlined_text(draw, (text_x, text_y), tl, title_font)
        text_y += title_line_height
    if title_lines:
        text_y += title_bottom_gap

    for bl in body_lines:
        if bl:
            _draw_outlined_text(draw, (text_x, text_y), bl, body_font)
        text_y += line_height

    img = np.array(pil_img, dtype=np.uint8)
    img = cv2.flip(img, 0)
    return img


# ─── Sentences page (typewriter) ────────────────────────────────────────────

# Laying the page out means wrapping the text at up to ten candidate sizes, so
# the result is cached — the render loop asks for a new frame ~20 times a second
# and only the visible character count changes between them.
_page_layout_cache = {}

_PAGE_SIZES = (64, 56, 48, 44, 40, 36, 32, 28, 24, 20, 18)


def _layout_page(draw, text, width, height, align, font_size,
                 line_height, margin_x, margin_y):
    """Wrap a page of text and work out where every line goes.

    With ``font_size=None`` the largest size that still fits vertically is
    chosen. Returns ``(font, lines, line_height, xs, y_start)`` where ``xs`` is
    the x position of each line — for centred text that is the *finished*
    line's position, so a half-typed line doesn't drift as it fills in.
    """
    key = (text, width, height, align, font_size, line_height, margin_x, margin_y)
    cached = _page_layout_cache.get(key)
    if cached is not None:
        return cached

    margin_x = width // 10 if margin_x is None else margin_x
    margin_y = height // 12 if margin_y is None else margin_y
    max_w = width - margin_x * 2
    max_h = height - margin_y * 2
    sizes = (font_size,) if font_size else _PAGE_SIZES

    font = None
    lines = []
    lh = 0
    for size in sizes:
        font = _load_font(size, config.OVERLAY_FONT_PATH)
        lh = round(size * 1.5) if line_height is None else line_height
        lines = []
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            current = ""
            for word in paragraph.split():
                test = current + " " + word if current else word
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] <= max_w:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
        if len(lines) * lh <= max_h:
            break

    if align == "center":
        xs = [(width - draw.textlength(line, font=font)) // 2 for line in lines]
    else:
        xs = [margin_x] * len(lines)

    y_start = (height - len(lines) * lh) // 2
    result = (font, lines, lh, xs, y_start)

    _page_layout_cache.clear()  # at most a couple of pages exist at a time
    _page_layout_cache[key] = result
    return result


class TypewriterPage:
    """Types a page of text out onto a persistent canvas.

    Redrawing all of the text every tick costs ~70ms — PIL rasterises the
    outline stroke separately for every call — which would drop the render loop
    below the background video's frame rate. The page only ever *gains*
    characters, so the canvas is kept between frames and each update draws just
    the newly revealed run of characters.
    """

    def __init__(self, text, width, height, align="left", font_size=None,
                 line_height=None, margin_x=None, margin_y=None):
        self.text = text
        self.width = width
        self.height = height
        self.align = align
        self.total_chars = sentences_page_total_chars(text)

        self.img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.img)
        # GL-ready mirror of the canvas, kept flipped. Converting the whole
        # 1920x1080 image costs ~11ms, so only changed rows are copied across.
        self._buf = np.zeros((height, width, 4), dtype=np.uint8)
        (self.font, self.lines, self.line_height,
         self.xs, self.y_start) = _layout_page(
            self.draw, text, width, height, align, font_size,
            line_height, margin_x, margin_y)

        # Where each wrapped line starts in the character stream. The line break
        # itself counts as one character, matching sentences_page_total_chars.
        self.line_starts = []
        cursor = 0
        for line in self.lines:
            self.line_starts.append(cursor)
            cursor += len(line) + 1

        self.drawn = 0

    def set_visible(self, visible_chars):
        """Draw any characters revealed since the last call.

        Returns True if the canvas changed and needs re-uploading.
        """
        visible_chars = max(0, min(int(visible_chars), self.total_chars))
        if visible_chars == self.drawn:
            return False

        if visible_chars < self.drawn:  # rewound — start the page over
            self.img.paste((0, 0, 0, 0), (0, 0, self.width, self.height))
            self._buf[:] = 0
            self.drawn = 0

        dirty_top = self.height
        dirty_bottom = 0
        for i, line in enumerate(self.lines):
            if not line:
                continue
            start = self.line_starts[i]
            already = max(0, min(self.drawn - start, len(line)))
            now = max(0, min(visible_chars - start, len(line)))
            if now <= already:
                continue
            x = self.xs[i]
            if already:
                x += self.draw.textlength(line[:already], font=self.font)
            line_y = self.y_start + i * self.line_height
            _draw_outlined_text(self.draw, (x, line_y), line[already:now], self.font)
            dirty_top = min(dirty_top, line_y - self.line_height)
            dirty_bottom = max(dirty_bottom, line_y + self.line_height * 2)

        self.drawn = visible_chars
        if dirty_bottom <= dirty_top:
            return False

        y0 = max(0, dirty_top)
        y1 = min(self.height, dirty_bottom)
        band = np.array(self.img.crop((0, y0, self.width, y1)), dtype=np.uint8)
        self._buf[self.height - y1:self.height - y0] = band[::-1]
        return True

    def frame(self):
        """The canvas as a GL-ready (flipped) uint8 RGBA array."""
        return self._buf


def sentences_page_total_chars(text):
    """Characters the typewriter has to get through — matches the reveal above."""
    return len(text) + text.count("\n")


def render_chat_messages(messages, width, height):
    """Render chat messages in bubble style — organism on left, user on right."""
    pil_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(22, config.OVERLAY_FONT_PATH)

    if not messages:
        return np.array(pil_img, dtype=np.uint8)

    padding = 20
    bubble_padding = 15
    message_spacing = 15
    line_height = 35
    max_bubble_width = int(width * 0.4)
    margin = 50

    bubble_data = []
    for msg in messages:
        sender = msg.get("sender", "user")
        text = msg.get("message", "")

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

        max_line_width = max(
            draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
            for line in lines
        )
        bubble_width = max_line_width + (bubble_padding * 2)
        bubble_height = len(lines) * line_height + (bubble_padding * 2)

        bubble_data.append({
            'sender': sender,
            'lines': lines,
            'bubble_width': bubble_width,
            'bubble_height': bubble_height,
        })

    dots_indicator_height = line_height + (bubble_padding * 2)
    total_height = (sum(b['bubble_height'] for b in bubble_data)
                    + message_spacing * len(bubble_data)
                    + dots_indicator_height)

    # Anchor newest message at the bottom of the image; older above.
    # After cv2.flip this puts newest at the bottom of the GL screen.
    start_y = min(margin, (height - margin) - total_height)
    y_position = start_y

    for data in bubble_data:
        bh = data['bubble_height']

        if y_position + bh < 0:
            y_position += bh + message_spacing
            continue

        bw = data['bubble_width']
        sender = data['sender']
        lines = data['lines']

        if sender == "organism":
            bubble_x = padding
            bubble_color = (0, 0, 0, 220)
        else:
            bubble_x = width - bw - padding
            bubble_color = (255, 255, 255, 220)

        bubble_y = y_position
        draw.rectangle(
            [(bubble_x, bubble_y), (bubble_x + bw, bubble_y + bh)],
            fill=bubble_color)

        text_y = bubble_y + bubble_padding
        for line in lines:
            _draw_outlined_text(draw, (bubble_x + bubble_padding, text_y), line, font)
            text_y += line_height

        y_position = bubble_y + bh + message_spacing

    # Animated typing indicator on the opposite side of the last message.
    last_sender = messages[-1].get("sender", "user")
    dots_sender = "organism" if last_sender == "user" else "user"
    dot_count = int(time.time() * 2) % 3 + 1
    dots_text = "." * dot_count

    dots_bbox = draw.textbbox((0, 0), "...", font=font)
    min_dots_width = dots_bbox[2] - dots_bbox[0]
    dots_bw = min_dots_width + (bubble_padding * 2)
    dots_bh = line_height + (bubble_padding * 2)

    if dots_sender == "organism":
        dots_x = padding
        dots_bubble_color = (0, 0, 0, 220)
    else:
        dots_x = width - dots_bw - padding
        dots_bubble_color = (255, 255, 255, 220)

    if y_position + dots_bh >= 0:
        draw.rectangle(
            [(dots_x, y_position), (dots_x + dots_bw, y_position + dots_bh)],
            fill=dots_bubble_color)
        _draw_outlined_text(draw, (dots_x + bubble_padding, y_position + bubble_padding),
                            dots_text, font)

    img = np.array(pil_img, dtype=np.uint8)
    img = cv2.flip(img, 0)
    return img
