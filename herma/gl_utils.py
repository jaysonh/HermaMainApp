"""OpenGL helpers: shader compilation, mesh generation, image-texture loading."""

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_CLAMP_TO_EDGE, GL_COMPILE_STATUS, GL_FRAGMENT_SHADER,
    GL_LINEAR, GL_LINK_STATUS, GL_RGBA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
    glAttachShader, glBindTexture, glCompileShader, glCreateProgram,
    glCreateShader, glDeleteShader, glGenTextures, glGetProgramInfoLog,
    glGetProgramiv, glGetShaderInfoLog, glGetShaderiv, glLinkProgram,
    glShaderSource, glTexImage2D, glTexParameteri,
)
from PIL import Image


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
    """Load an image from disk and create an OpenGL texture for it."""
    if not image_path.exists():
        print(f"Warning: Image file not found: {image_path}")
        return None, 0, 0

    try:
        img = Image.open(image_path).convert("RGBA")
        img_data = np.array(img, dtype=np.uint8)
        img_data = np.flipud(img_data)

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
