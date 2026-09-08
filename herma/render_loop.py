"""GLFW window, OpenGL resource setup, and the main render loop.

Public surface:
    bootstrap(display_cfg, cam_idx) -> (win, cap, frame, cam_w, cam_h)
        Initialise GLFW, open the window, open the camera, read one frame.

    run(win, cap, frame, cam_w, cam_h)
        Compile shaders, allocate GL resources, enter the render loop.
        Returns when the window is closed.
"""

import ctypes
import sys
import time
from types import SimpleNamespace

import cv2
import glfw
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_BLEND, GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT,
    GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA, GL_RGB, GL_RGBA, GL_SRC_ALPHA,
    GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_FAN, GL_TRIANGLES, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT,
    glActiveTexture, glBindBuffer, glBindTexture, glBlendFunc, glBufferData,
    glClear, glClearColor, glDisable, glDisableVertexAttribArray, glDrawArrays,
    glDrawElements, glEnable, glEnableVertexAttribArray, glGenBuffers,
    glGenTextures, glGetAttribLocation, glGetUniformLocation, glTexImage2D,
    glTexParameteri, glTexSubImage2D, glUniform1f, glUniform1i, glUniform2f,
    glUniform3f, glUniformMatrix4fv, glUseProgram, glVertexAttribPointer,
    glViewport,
)

from . import config, gl_utils, overlays, state, video
from shaders import VERT_SRC, FRAG_SRC, HUD_VERT_SRC, HUD_FRAG_SRC


# ─── Bootstrap (GLFW window + webcam) ───────────────────────────────────────

def bootstrap(display_cfg, cam_idx):
    """Initialise GLFW, create the window, open the camera, return everything."""
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)

    monitor = config.pick_monitor_from_config(display_cfg)
    mode = glfw.get_video_mode(monitor)
    win = glfw.create_window(mode.size.width, mode.size.height,
                             "Herma Webcam Shader", monitor, None)
    if not win:
        glfw.terminate()
        print("Failed to create window")
        sys.exit(1)

    glfw.make_context_current(win)
    glfw.swap_interval(1)

    cap = cv2.VideoCapture(cam_idx)
    state.webcam_cap = cap
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

    if config.DOWNSCALE_WIDTH is not None:
        fh, fw = frame.shape[:2]
        if fw > config.DOWNSCALE_WIDTH:
            scale = config.DOWNSCALE_WIDTH / float(fw)
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                               interpolation=cv2.INTER_AREA)

    cam_h, cam_w = frame.shape[:2]
    print(f"Camera opened: {cam_w}x{cam_h} (index {cam_idx})")

    # Adjust rect1 aspect ratio to match camera
    cam_aspect = cam_w / cam_h
    area = config.P['rect1W'] * config.P['rect1H']
    config.P['rect1H'] = (area / cam_aspect) ** 0.5
    config.P['rect1W'] = config.P['rect1H'] * cam_aspect

    return win, cap, frame, cam_w, cam_h


# ─── GL resource setup ──────────────────────────────────────────────────────

_TERRAIN_UNIFORM_NAMES = [
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


def _setup_gl_resources(win, frame, cam_w, cam_h):
    """Compile shaders, build the grid mesh, allocate textures and VBOs.

    Returns a SimpleNamespace bundling every GL handle the render loop needs.
    """
    gl = SimpleNamespace()

    # Terrain shader program
    gl.prog = gl_utils.link_program(VERT_SRC, FRAG_SRC)
    gl.u = {name: glGetUniformLocation(gl.prog, name) for name in _TERRAIN_UNIFORM_NAMES}
    gl.pos_loc = glGetAttribLocation(gl.prog, 'a_position')
    gl.uv_loc = glGetAttribLocation(gl.prog, 'a_uv')

    # Grid mesh
    print(f"Building {config.GRID_RES}x{config.GRID_RES} grid mesh "
          f"(aspect {config.TARGET_ASPECT:.2f})...")
    positions, uvs, indices = gl_utils.make_grid(config.GRID_RES, config.TARGET_ASPECT)
    gl.num_indices = len(indices)

    gl.vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, gl.vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)

    gl.vbo_uv = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, gl.vbo_uv)
    glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)

    gl.ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Webcam texture
    gl.cam_w = cam_w
    gl.cam_h = cam_h
    gl.tex = _make_texture()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    # Logo texture + quad
    gl.logo_tex, logo_w, logo_h = gl_utils.load_image_texture(config.LOGO_PATH)
    gl.logo_vbo = None
    if gl.logo_tex is not None and logo_w > 0 and logo_h > 0:
        init_fb_w, init_fb_h = glfw.get_framebuffer_size(win)
        logo_aspect = logo_w / logo_h
        screen_aspect = init_fb_w / init_fb_h
        ndc_h = (screen_aspect / logo_aspect) * 2.0
        ndc_h = min(ndc_h, 2.0)
        half_h = ndc_h / 2.0
        logo_quad = np.array([
            -1.0, -half_h,  0.0, 0.0,
             1.0, -half_h,  1.0, 0.0,
             1.0,  half_h,  1.0, 1.0,
            -1.0,  half_h,  0.0, 1.0,
        ], dtype=np.float32)
        gl.logo_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, gl.logo_vbo)
        glBufferData(GL_ARRAY_BUFFER, logo_quad.nbytes, logo_quad, GL_STATIC_DRAW)

    # HUD program + bottom-strip quad
    gl.hud_prog = gl_utils.link_program(HUD_VERT_SRC, HUD_FRAG_SRC)
    gl.hud_pos_loc = glGetAttribLocation(gl.hud_prog, 'a_pos')
    gl.hud_uv_loc = glGetAttribLocation(gl.hud_prog, 'a_uv')
    gl.hud_tex_loc = glGetUniformLocation(gl.hud_prog, 'u_tex')

    hud_quad = np.array([
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0, -0.92, 1.0, 1.0,
        -1.0, -0.92, 0.0, 1.0,
    ], dtype=np.float32)
    gl.hud_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, gl.hud_vbo)
    glBufferData(GL_ARRAY_BUFFER, hud_quad.nbytes, hud_quad, GL_STATIC_DRAW)

    gl.hud_tex = _make_texture()
    blank_hud = np.zeros((config.HUD_HEIGHT, config.HUD_WIDTH, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, config.HUD_WIDTH, config.HUD_HEIGHT, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank_hud)

    # Fullscreen overlay quad (logo is sized differently — see above)
    overlay_quad = np.array([
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0,  1.0,  1.0, 1.0,
        -1.0,  1.0,  0.0, 1.0,
    ], dtype=np.float32)
    gl.overlay_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, gl.overlay_vbo)
    glBufferData(GL_ARRAY_BUFFER, overlay_quad.nbytes, overlay_quad, GL_STATIC_DRAW)

    # Instructions overlay (typed out on entry — see the render loop)
    gl.overlay_tex = _make_texture()
    blank_overlay = np.zeros((1080, 1920, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank_overlay)

    # Organism overlay (re-rendered when text changes)
    gl.organism_text_tex = _make_texture()
    organism_text_img = overlays.render_organism_overlay(config.ORGANISM_TEXT, 1920, 1080)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, organism_text_img)

    # Sentences page (re-rendered as the typewriter reveals more characters)
    gl.sentences_tex = _make_texture()
    blank_page = np.zeros((1080, 1920, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank_page)

    # Background video shown instead of the terrain while the page is up
    gl.video_tex = _make_texture()
    gl.video_w = 0
    gl.video_h = 0

    # Chat overlay (re-rendered every frame while visible)
    gl.chat_text_tex = _make_texture()
    blank_chat = np.zeros((1080, 1920, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank_chat)

    return gl


def _make_texture():
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return tex


# ─── Input callbacks ────────────────────────────────────────────────────────

def _install_callbacks(win, ctx):
    def on_scroll(_win, _xoff, yoff):
        ctx.zoom = max(0.5, min(3.0, ctx.zoom + yoff * 0.1))

    def on_key(_win, key, _sc, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(win, True)
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            mon = glfw.get_primary_monitor()
            vid = glfw.get_video_mode(mon)
            if ctx.is_fullscreen:
                glfw.set_window_monitor(win, None,
                                        ctx.windowed_pos[0], ctx.windowed_pos[1],
                                        ctx.windowed_size[0], ctx.windowed_size[1], 0)
            else:
                ctx.windowed_pos = list(glfw.get_window_pos(win))
                ctx.windowed_size = list(glfw.get_window_size(win))
                glfw.set_window_monitor(win, mon, 0, 0,
                                        vid.size.width, vid.size.height,
                                        vid.refresh_rate)
            ctx.is_fullscreen = not ctx.is_fullscreen
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            ctx.show_hud = not ctx.show_hud
        if key == glfw.KEY_R and action == glfw.PRESS:
            print("Key: Reset to initial state")
            state.request_restart()

    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_key_callback(win, on_key)


# ─── Render helpers ─────────────────────────────────────────────────────────

def _upload_webcam_texture(gl, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, 0)
    fh, fw = frame_rgb.shape[:2]
    glBindTexture(GL_TEXTURE_2D, gl.tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fw, fh, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)


def _clear_webcam_texture(gl):
    """Upload a black frame to the webcam texture (used while overlays show)."""
    black = np.zeros((gl.cam_h, gl.cam_w, 3), dtype=np.uint8)
    black = cv2.flip(black, 0)
    glBindTexture(GL_TEXTURE_2D, gl.tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, gl.cam_w, gl.cam_h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, black)


def _upload_video_texture(gl, frame):
    """Upload a background-video frame, reallocating if its size changed."""
    fh, fw = frame.shape[:2]
    glBindTexture(GL_TEXTURE_2D, gl.video_tex)
    if (fw, fh) != (gl.video_w, gl.video_h):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fw, fh, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, frame)
        gl.video_w, gl.video_h = fw, fh
    else:
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fw, fh,
                        GL_RGB, GL_UNSIGNED_BYTE, frame)


def _render_terrain(gl, fb_w, fb_h, ctx, t0, webcam_visible):
    glViewport(0, 0, fb_w, fb_h)
    glClearColor(0.03, 0.03, 0.05, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(gl.prog)

    aspect = fb_w / max(fb_h, 1)
    z = ctx.zoom
    if aspect > 1:
        ol, or_, ob, ot = -z * aspect, z * aspect, -z, z
    else:
        ol, or_, ob, ot = -z, z, -z / aspect, z / aspect
    mvp = gl_utils.make_ortho(ol, or_, ob, ot, -10.0, 10.0)

    P = config.P
    u = gl.u

    # Drive rect1 elevation/blend based on webcam visibility
    if webcam_visible:
        P['rect1Elev'] = 0.3
        P['rect1Blend'] = 0.25
    else:
        P['rect1Elev'] = 0.0
        P['rect1Blend'] = 0.0

    glUniformMatrix4fv(u['u_modelViewProjection'], 1, GL_FALSE, mvp)
    glUniform1f(u['u_time'], time.time() - t0)
    glUniform1f(u['u_heightScale'], P['heightScale'])
    glUniform1f(u['u_dripSpeed'], P['dripSpeed'])
    glUniform1f(u['u_distortion'], P['distortion'])
    glUniform1f(u['u_ringCount'], P['ringCount'])
    glUniform3f(u['u_lightDir'], 0.3, 0.3, 0.9)
    glUniform1f(u['u_aspectRatio'], config.TARGET_ASPECT)

    for r in ('rect1', 'rect3', 'rect4'):
        glUniform2f(u[f'u_{r}Pos'], P[f'{r}X'], P[f'{r}Y'])
        glUniform2f(u[f'u_{r}Size'], P[f'{r}W'], P[f'{r}H'])
        glUniform1f(u[f'u_{r}Elevation'], P[f'{r}Elev'])
        glUniform1f(u[f'u_{r}Blend'], P[f'{r}Blend'])
        glUniform1f(u[f'u_{r}Hue'], P[f'{r}Hue'])
        glUniform1f(u[f'u_{r}Sat'], P[f'{r}Sat'])
        glUniform1f(u[f'u_{r}Bright'], P[f'{r}Bright'])
        glUniform1f(u[f'u_{r}Contrast'], P[f'{r}Contrast'])

    glUniform1f(u['u_terrainHue'], P['terrainHue'])
    glUniform1f(u['u_terrainSat'], P['terrainSat'])
    glUniform1f(u['u_terrainBright'], P['terrainBright'])
    glUniform1f(u['u_terrainContrast'], P['terrainContrast'])

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, gl.tex)
    glUniform1i(u['u_webcamTex'], 0)

    glBindBuffer(GL_ARRAY_BUFFER, gl.vbo_pos)
    glEnableVertexAttribArray(gl.pos_loc)
    glVertexAttribPointer(gl.pos_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, gl.vbo_uv)
    glEnableVertexAttribArray(gl.uv_loc)
    glVertexAttribPointer(gl.uv_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.ebo)
    glDrawElements(GL_TRIANGLES, gl.num_indices, GL_UNSIGNED_INT, None)


def _draw_textured_quad(gl, tex, vbo):
    """Common draw path for fullscreen overlays: bind tex+vbo, draw a quad."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_DEPTH_TEST)

    glUseProgram(gl.hud_prog)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glUniform1i(gl.hud_tex_loc, 0)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableVertexAttribArray(gl.hud_pos_loc)
    glVertexAttribPointer(gl.hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(gl.hud_uv_loc)
    glVertexAttribPointer(gl.hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

    glDisableVertexAttribArray(gl.hud_pos_loc)
    glDisableVertexAttribArray(gl.hud_uv_loc)

    glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)


# ─── Main loop ──────────────────────────────────────────────────────────────

def run(win, cap, frame, cam_w, cam_h, recording_machine):
    gl = _setup_gl_resources(win, frame, cam_w, cam_h)

    ctx = SimpleNamespace(
        zoom=1.0,
        show_hud=False,
        is_fullscreen=True,
        windowed_pos=[100, 100],
        windowed_size=[config.WINDOW_W, config.WINDOW_H],
    )
    _install_callbacks(win, ctx)

    end_video = video.BackgroundVideo(state.END_VIDEO_PATH)
    page_renderer = None          # overlays.TypewriterPage while a page is up
    instructions_renderer = None  # ditto, for the instructions screen
    instructions_start = None     # when the instructions typing began

    glEnable(GL_DEPTH_TEST)
    t0 = time.time()
    last_intro_state = None

    print("Intro state: logo (waiting for /api/begin)")
    print("Running. Press ESC to quit. Tab to toggle fullscreen. Scroll to zoom.")

    while not glfw.window_should_close(win):
        glfw.poll_events()

        with state.intro_lock:
            current_intro_state = state.intro_state
        if current_intro_state != last_intro_state:
            print(f"Render loop: intro_state → {current_intro_state}")
            last_intro_state = current_intro_state

        with state.chat_lock:
            current_show_chat = state.show_chat
            current_chat_messages = list(state.chat_messages)

        with state.organism_lock:
            current_show_organism = state.show_organism
            current_organism_text_dirty = state.organism_text_dirty
            current_organism_text = state.organism_overlay_text
            state.organism_text_dirty = False

        (page_text, page_start, page_cps, page_align,
         show_end_video) = state.get_sentences_page()

        # ── Restart ──
        if state.consume_restart_request():
            print("Restarting to initial state...")
            state.reset_to_initial_state()
            recording_machine.reset()
            _clear_webcam_texture(gl)
            end_video.stop()
            page_renderer = None
            instructions_renderer = None
            instructions_start = None
            continue

        # ── Background video follows the sentences sequence ──
        if show_end_video and not end_video.active:
            end_video.start(time.time())
        elif not show_end_video and end_video.active:
            end_video.stop()

        # ── Re-render organism text when needed (also every frame while loading, to animate dots) ──
        organism_is_loading = (current_show_organism
                               and current_organism_text.strip() == config.LOADING_TEXT)
        if current_organism_text_dirty or organism_is_loading:
            organism_text_img = overlays.render_organism_overlay(current_organism_text, 1920, 1080)
            glBindTexture(GL_TEXTURE_2D, gl.organism_text_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, organism_text_img)

        # ── Type out the sentences page ──
        page_visible = bool(page_text) and page_start is not None
        if page_visible:
            if (page_renderer is None or page_renderer.text != page_text
                    or page_renderer.align != page_align):
                page_renderer = overlays.TypewriterPage(page_text, 1920, 1080,
                                                        align=page_align)
            typed = (time.monotonic() - page_start) * page_cps
            if page_renderer.set_visible(typed):
                glBindTexture(GL_TEXTURE_2D, gl.sentences_tex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, page_renderer.frame())
        else:
            page_renderer = None

        # ── Type out the instructions screen ──
        if current_intro_state == "instructions":
            if instructions_renderer is None:
                instructions_renderer = overlays.TypewriterPage(
                    config.ONBOARDING_TEXT, 1920, 1080,
                    align="center", font_size=64, line_height=90, margin_x=80)
                instructions_start = time.monotonic()
            typed = (time.monotonic() - instructions_start) * config.INSTRUCTIONS_CPS
            if instructions_renderer.set_visible(typed):
                glBindTexture(GL_TEXTURE_2D, gl.overlay_tex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, instructions_renderer.frame())
        else:
            instructions_renderer = None
            instructions_start = None

        # ── Hide stale webcam frame while overlays show ──
        if current_show_chat or current_show_organism:
            _clear_webcam_texture(gl)

        # ── Webcam pipeline (only in running state, no chat) ──
        if current_intro_state == "running" and not current_show_chat:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: cap.read() failed - webcam not delivering frames")
            else:
                now = time.time()
                frame = recording_machine.step(frame, now)
                _upload_webcam_texture(gl, frame)

        # ── Viewport ──
        fb_w, fb_h = glfw.get_framebuffer_size(win)
        if fb_w == 0 or fb_h == 0:
            glfw.poll_events()
            continue

        webcam_visible = (current_intro_state == "running"
                          and not current_show_chat
                          and not current_show_organism)

        if show_end_video:
            # Terrain shader hidden — the video (or black, if it failed to
            # open) is the whole background.
            glViewport(0, 0, fb_w, fb_h)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            video_frame = end_video.poll(time.time())
            if video_frame is not None:
                _upload_video_texture(gl, video_frame)
            if gl.video_w:
                _draw_textured_quad(gl, gl.video_tex, gl.overlay_vbo)
        else:
            _render_terrain(gl, fb_w, fb_h, ctx, t0, webcam_visible)

        # ── Intro overlays ──
        if current_intro_state == "logo" and gl.logo_tex is not None and gl.logo_vbo is not None:
            _draw_textured_quad(gl, gl.logo_tex, gl.logo_vbo)
        elif current_intro_state == "instructions":
            _draw_textured_quad(gl, gl.overlay_tex, gl.overlay_vbo)

        # ── Sentences page / organism overlay ──
        if page_visible:
            _draw_textured_quad(gl, gl.sentences_tex, gl.overlay_vbo)
        elif current_show_organism:
            _draw_textured_quad(gl, gl.organism_text_tex, gl.overlay_vbo)

        # ── Chat overlay (re-rendered every frame to animate typing dots) ──
        if current_show_chat and current_chat_messages:
            chat_img = overlays.render_chat_messages(current_chat_messages, 1920, 1080)
            glBindTexture(GL_TEXTURE_2D, gl.chat_text_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, chat_img)
            _draw_textured_quad(gl, gl.chat_text_tex, gl.overlay_vbo)

        # ── HUD strip (only in running state) ──
        if current_intro_state == "running" and ctx.show_hud:
            hud_img = overlays.render_hud_text(
                recording_machine.hud_mode,
                recording_machine.hud_state,
                recording_machine.img_index,
                recording_machine.hud_time_remaining,
                recording_machine.hud_has_motion,
            )
            glBindTexture(GL_TEXTURE_2D, gl.hud_tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, config.HUD_WIDTH, config.HUD_HEIGHT,
                            GL_RGBA, GL_UNSIGNED_BYTE, hud_img)
            _draw_textured_quad(gl, gl.hud_tex, gl.hud_vbo)

        glfw.swap_buffers(win)
