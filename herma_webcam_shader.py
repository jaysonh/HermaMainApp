#!/usr/bin/env python3
"""Herma Webcam Shader

Displays webcam feed embedded in a melting terrain shader.

Requirements:
    pip install glfw PyOpenGL opencv-python numpy flask flask-cors requests

Controls:
    ESC    - Quit
    Tab    - Toggle fullscreen/windowed
    Space  - Toggle HUD status bar
    Scroll - Zoom in/out
"""

import sys
import time
import threading
import ctypes
from datetime import datetime
from pathlib import Path
import os
import argparse

import numpy as np
import cv2
import requests
from flask import Flask, Response, jsonify, request as flask_request
from flask_cors import CORS

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli

try:
    import glfw
    from OpenGL.GL import *
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install glfw PyOpenGL opencv-python numpy flask flask-cors requests")
    sys.exit(1)

# ─── Configuration ──────────────────────────────────────────────────────────

CAM_INDEX = 1
WINDOW_W = 1280
WINDOW_H = 720
GRID_RES = 200
TARGET_ASPECT = 16.0 / 9.0

# Default shader parameters (from herma-record.html)
P = dict(
    heightScale=0.3, dripSpeed=0.4, distortion=0.5, ringCount=8.0,
    terrainHue=-0.35, terrainSat=1.0, terrainBright=1.0, terrainContrast=1.0,
    rect1X=0.5, rect1Y=0.42, rect1W=0.21, rect1H=0.2,
    rect1Elev=0.3, rect1Blend=0.25,
    rect1Hue=0.0, rect1Sat=1.0, rect1Bright=1.0, rect1Contrast=1.0,
    rect3X=0.5, rect3Y=0.86, rect3W=0.22, rect3H=0.06,
    rect3Elev=0.4, rect3Blend=0.15,
    rect3Hue=0.2, rect3Sat=1.0, rect3Bright=1.1, rect3Contrast=1.0,
    rect4X=0.36, rect4Y=0.14, rect4W=0.11, rect4H=0.05,
    rect4Elev=0.3, rect4Blend=0.15,
    rect4Hue=0.2, rect4Sat=1.3, rect4Bright=1.0, rect4Contrast=1.0,
)

# ─── Recording / Web Server Settings ────────────────────────────────────────

MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 2500

CAPTURE_INTERVAL = 0.5
RECORDING_TIMEOUT = 30.0
STILL_SECONDS_TO_STOP = 5.0

DOWNSCALE_WIDTH = 960
OUTPUT_DIR_API = Path("recorded")
OUTPUT_DIR_AUTO = Path("tmp")

WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
STREAM_FPS = 10
JPEG_QUALITY = 80

API_KEY = os.environ.get("MONITOR_API_KEY", "jayson")
AWS_UPLOAD_URL = os.environ.get("AWS_UPLOAD_URL", "http://10.142.77.6:5009/api/upload")
AWS_UPLOAD_KEY = os.environ.get("AWS_UPLOAD_KEY", "jayson")

# ─── Flask App & Shared State ───────────────────────────────────────────────

flask_app = Flask(__name__)
CORS(flask_app)

latest_jpeg = None
jpeg_lock = threading.Lock()

control_lock = threading.Lock()
manual_record_command = None
recording_start_time = None
current_status = {
    "recording": False,
    "sequence_dir": None,
    "images_captured": 0,
    "last_motion_time": None,
    "time_remaining": None,
}


# ─── Vertex Shader ──────────────────────────────────────────────────────────

VERT_SRC = """
#version 120

attribute vec3 a_position;
attribute vec2 a_uv;

uniform mat4 u_modelViewProjection;
uniform float u_time;
uniform float u_heightScale;
uniform float u_dripSpeed;
uniform float u_distortion;
uniform float u_ringCount;
uniform float u_aspectRatio;

uniform vec2 u_rect1Pos;
uniform vec2 u_rect1Size;
uniform float u_rect1Elevation;
uniform float u_rect1Blend;

uniform vec2 u_rect3Pos;
uniform vec2 u_rect3Size;
uniform float u_rect3Elevation;
uniform float u_rect3Blend;

uniform vec2 u_rect4Pos;
uniform vec2 u_rect4Size;
uniform float u_rect4Elevation;
uniform float u_rect4Blend;

varying vec2 v_uv;
varying float v_height;
varying vec3 v_normal;
varying vec3 v_position;
varying float v_rect1Blend;
varying float v_rect3Blend;
varying float v_rect4Blend;
varying float v_distanceToRect1Center;
varying float v_distanceToRect3Center;
varying float v_distanceToRect4Center;
varying float v_terrainHeight;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289v2(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289v2(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amplitude * snoise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

float squareDistance(vec2 uv, vec2 center) {
    vec2 d = abs(uv - center);
    return max(d.x, d.y);
}

float rectSDF(vec2 uv, vec2 rectPos, vec2 rectSize, float edgeBlend, float time) {
    vec2 d = abs(uv - rectPos) - rectSize;
    float edgeNoise = fbm(uv * 15.0 + time * 0.2) * 0.03 * edgeBlend;
    edgeNoise += snoise(uv * 30.0 + time * 0.3) * 0.015 * edgeBlend;
    float dripNoise = snoise(vec2(uv.x * 20.0, time * u_dripSpeed)) * 0.02;
    dripNoise *= smoothstep(rectPos.y - rectSize.y, rectPos.y - rectSize.y - 0.1, uv.y);
    dripNoise *= edgeBlend;
    float outside = length(max(d, 0.0));
    float inside = min(max(d.x, d.y), 0.0);
    return outside + inside + edgeNoise + dripNoise;
}

float getRectBlend(vec2 uv, vec2 rectPos, vec2 rectSize, float edgeBlend, float time) {
    float dist = rectSDF(uv, rectPos, rectSize, edgeBlend, time);
    float blendWidth = 0.02 + edgeBlend * 0.05;
    return 1.0 - smoothstep(-blendWidth, blendWidth * 0.5, dist);
}

float getTerrainHeight(vec2 uv, float time) {
    vec2 warpedUV = uv;
    float n1 = fbm(uv * 2.0 + vec2(time * 0.1, 0.0));
    float n2 = fbm(uv * 2.0 + vec2(0.0, time * 0.15));
    warpedUV += vec2(n1, n2) * u_distortion * 0.3;

    float dripNoise = snoise(vec2(uv.x * 8.0, 0.0)) * 0.5 + 0.5;
    float drips = snoise(vec2(uv.x * 15.0, uv.y * 2.0 - time * u_dripSpeed - dripNoise * 3.0));
    drips = smoothstep(0.3, 0.7, drips) * (1.0 - uv.y) * 0.3;
    warpedUV.y += drips * u_distortion;

    vec2 center = vec2(0.5);
    float dist = squareDistance(warpedUV, center);
    float noiseVal = fbm(warpedUV * 3.0 + time * 0.05) * 0.08;
    dist += noiseVal * u_distortion;
    float ringPattern = sin(dist * u_ringCount * 6.28318) * 0.5 + 0.5;

    float height = ringPattern * 0.7;
    height -= drips * 0.4;

    float streaks = snoise(vec2(uv.x * 40.0, uv.y * 2.0 - time * u_dripSpeed));
    streaks = smoothstep(0.6, 0.9, streaks);
    height -= streaks * 0.2 * (1.0 - uv.y);

    height += fbm(warpedUV * 8.0) * 0.1;

    vec2 edgeD = abs(uv - center);
    edgeD.x /= u_aspectRatio;
    float edgeDist = max(edgeD.x, edgeD.y);
    float falloff = 1.0 - smoothstep(0.35, 0.5, edgeDist);
    height *= falloff;

    return height * u_heightScale;
}

float getRectElevation(vec2 uv, vec2 rectPos, vec2 rectSize, float rectElevation, float edgeBlend, float time) {
    float baseHeight = rectElevation;
    vec2 normalizedDist = (uv - rectPos) / rectSize;
    float distToCenter = length(normalizedDist);
    float edgeFactor = smoothstep(0.3, 1.0, distToCenter) * edgeBlend;

    float surfaceNoise = fbm(uv * 20.0 + time * 0.3) * 0.03 * edgeFactor;
    surfaceNoise += snoise(uv * 40.0 + time * 0.5) * 0.015 * edgeFactor;

    float edgeDrip = snoise(vec2(uv.x * 25.0, time * u_dripSpeed * 0.5));
    edgeDrip = smoothstep(0.5, 0.8, edgeDrip) * 0.02 * edgeFactor;

    return baseHeight + surfaceNoise - edgeDrip;
}

float getHeight(vec2 uv, float time) {
    float terrainH = getTerrainHeight(uv, time);

    float rect1H = getRectElevation(uv, u_rect1Pos, u_rect1Size, u_rect1Elevation, u_rect1Blend, time);
    float blend1 = getRectBlend(uv, u_rect1Pos, u_rect1Size, u_rect1Blend, time);
    float elevationFade1 = 1.0 - exp(-u_rect1Elevation * 30.0);
    blend1 *= elevationFade1;

    float rect3H = getRectElevation(uv, u_rect3Pos, u_rect3Size, u_rect3Elevation, u_rect3Blend, time);
    float blend3 = getRectBlend(uv, u_rect3Pos, u_rect3Size, u_rect3Blend, time);
    float elevationFade3 = 1.0 - exp(-u_rect3Elevation * 30.0);
    blend3 *= elevationFade3;

    float rect4H = getRectElevation(uv, u_rect4Pos, u_rect4Size, u_rect4Elevation, u_rect4Blend, time);
    float blend4 = getRectBlend(uv, u_rect4Pos, u_rect4Size, u_rect4Blend, time);
    float elevationFade4 = 1.0 - exp(-u_rect4Elevation * 30.0);
    blend4 *= elevationFade4;

    float edgeTerrainBleed1 = terrainH * 0.3 * (1.0 - blend1) * u_rect1Blend;
    float edgeTerrainBleed3 = terrainH * 0.3 * (1.0 - blend3) * u_rect3Blend;
    float edgeTerrainBleed4 = terrainH * 0.3 * (1.0 - blend4) * u_rect4Blend;

    float height = terrainH;
    height = mix(height, rect1H + edgeTerrainBleed1 * blend1, blend1);
    height = mix(height, rect3H + edgeTerrainBleed3 * blend3, blend3);
    height = mix(height, rect4H + edgeTerrainBleed4 * blend4, blend4);

    return height;
}

void main() {
    v_uv = a_uv;
    float time = u_time;
    float height = getHeight(a_uv, time);
    v_height = height;
    v_terrainHeight = getTerrainHeight(a_uv, time);

    float rawBlend1 = getRectBlend(a_uv, u_rect1Pos, u_rect1Size, u_rect1Blend, time);
    float elevationFade1 = 1.0 - exp(-u_rect1Elevation * 30.0);
    v_rect1Blend = rawBlend1 * elevationFade1;

    float rawBlend3 = getRectBlend(a_uv, u_rect3Pos, u_rect3Size, u_rect3Blend, time);
    float elevationFade3 = 1.0 - exp(-u_rect3Elevation * 30.0);
    v_rect3Blend = rawBlend3 * elevationFade3;

    float rawBlend4 = getRectBlend(a_uv, u_rect4Pos, u_rect4Size, u_rect4Blend, time);
    float elevationFade4 = 1.0 - exp(-u_rect4Elevation * 30.0);
    v_rect4Blend = rawBlend4 * elevationFade4;

    vec2 nd1 = (a_uv - u_rect1Pos) / u_rect1Size;
    v_distanceToRect1Center = length(nd1);
    vec2 nd3 = (a_uv - u_rect3Pos) / u_rect3Size;
    v_distanceToRect3Center = length(nd3);
    vec2 nd4 = (a_uv - u_rect4Pos) / u_rect4Size;
    v_distanceToRect4Center = length(nd4);

    vec3 pos = a_position;
    pos.z = height;

    float eps = 0.004;
    float hL = getHeight(a_uv - vec2(eps, 0.0), time);
    float hR = getHeight(a_uv + vec2(eps, 0.0), time);
    float hD = getHeight(a_uv - vec2(0.0, eps), time);
    float hU = getHeight(a_uv + vec2(0.0, eps), time);

    v_normal = normalize(vec3(hL - hR, hD - hU, eps * 4.0));
    v_position = pos;
    gl_Position = u_modelViewProjection * vec4(pos, 1.0);
}
"""


# ─── Fragment Shader ────────────────────────────────────────────────────────

FRAG_SRC = """
#version 120

varying vec2 v_uv;
varying float v_height;
varying vec3 v_normal;
varying vec3 v_position;
varying float v_rect1Blend;
varying float v_rect3Blend;
varying float v_rect4Blend;
varying float v_distanceToRect1Center;
varying float v_distanceToRect3Center;
varying float v_distanceToRect4Center;
varying float v_terrainHeight;

uniform float u_time;
uniform float u_heightScale;
uniform float u_rect1Elevation;
uniform float u_rect1Blend;
uniform float u_rect3Elevation;
uniform float u_rect3Blend;
uniform float u_rect4Elevation;
uniform float u_rect4Blend;
uniform vec3 u_lightDir;

uniform float u_terrainHue;
uniform float u_terrainSat;
uniform float u_terrainBright;
uniform float u_terrainContrast;

uniform float u_rect1Hue;
uniform float u_rect1Sat;
uniform float u_rect1Bright;
uniform float u_rect1Contrast;

uniform float u_rect3Hue;
uniform float u_rect3Sat;
uniform float u_rect3Bright;
uniform float u_rect3Contrast;

uniform float u_rect4Hue;
uniform float u_rect4Sat;
uniform float u_rect4Bright;
uniform float u_rect4Contrast;

// Webcam & aspect
uniform sampler2D u_webcamTex;
uniform vec2 u_rect1Pos;
uniform vec2 u_rect1Size;
uniform float u_aspectRatio;

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289v2(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289v2(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 4; i++) {
        value += amplitude * snoise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

float squareDistance(vec2 uv, vec2 center) {
    vec2 d = abs(uv - center);
    return max(d.x, d.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 adjustColor(vec3 color, float hueShift, float satMult, float brightMult, float contrast) {
    vec3 hsv = rgb2hsv(color);
    hsv.x = fract(hsv.x + hueShift);
    hsv.y = clamp(hsv.y * satMult, 0.0, 1.0);
    hsv.z = clamp(hsv.z * brightMult, 0.0, 1.0);
    vec3 rgb = hsv2rgb(hsv);
    rgb = (rgb - 0.5) * contrast + 0.5;
    return clamp(rgb, 0.0, 1.0);
}

void main() {
    vec3 normal = normalize(v_normal);
    vec3 lightDir = normalize(u_lightDir);

    float h = v_height / max(u_heightScale, 0.01);
    h = clamp(h * 1.2, 0.0, 1.0);

    // Terrain colors
    vec3 deepBlue = vec3(0.05, 0.08, 0.25);
    vec3 blue = vec3(0.15, 0.3, 0.75);
    vec3 lightBlue = vec3(0.4, 0.55, 0.9);
    vec3 silver = vec3(0.65, 0.7, 0.75);
    vec3 gold = vec3(0.8, 0.7, 0.45);
    vec3 white = vec3(0.95, 0.95, 1.0);

    vec3 terrainColor;
    if (h < 0.15) {
        terrainColor = mix(deepBlue, blue, h / 0.15);
    } else if (h < 0.4) {
        terrainColor = mix(blue, lightBlue, (h - 0.15) / 0.25);
    } else if (h < 0.65) {
        terrainColor = mix(lightBlue, silver, (h - 0.4) / 0.25);
    } else if (h < 0.85) {
        terrainColor = mix(silver, gold, (h - 0.65) / 0.2);
    } else {
        terrainColor = mix(gold, white, (h - 0.85) / 0.15);
    }
    terrainColor = adjustColor(terrainColor, u_terrainHue, u_terrainSat, u_terrainBright, u_terrainContrast);

    // Rectangle 1 - procedural base color
    vec3 rect1Deep = vec3(0.04, 0.06, 0.2);
    vec3 rect1Blue = vec3(0.12, 0.25, 0.65);
    vec3 rect1LightBlue = vec3(0.3, 0.45, 0.8);
    vec3 rect1Silver = vec3(0.5, 0.55, 0.65);
    vec3 rect1Gold = vec3(0.6, 0.55, 0.4);

    vec3 rect1Color;
    if (h < 0.2) {
        rect1Color = mix(rect1Deep, rect1Blue, h / 0.2);
    } else if (h < 0.5) {
        rect1Color = mix(rect1Blue, rect1LightBlue, (h - 0.2) / 0.3);
    } else if (h < 0.75) {
        rect1Color = mix(rect1LightBlue, rect1Silver, (h - 0.5) / 0.25);
    } else {
        rect1Color = mix(rect1Silver, rect1Gold, (h - 0.75) / 0.25);
    }
    rect1Color = adjustColor(rect1Color, u_rect1Hue, u_rect1Sat, u_rect1Bright, u_rect1Contrast);

    // ── Webcam texture replaces rect1 interior ──
    vec2 webcamUV = (v_uv - (u_rect1Pos - u_rect1Size)) / (u_rect1Size * 2.0);
    vec3 webcamSample = texture2D(u_webcamTex, clamp(webcamUV, 0.0, 1.0)).rgb;
    float inRect = step(0.0, webcamUV.x) * step(webcamUV.x, 1.0)
                 * step(0.0, webcamUV.y) * step(webcamUV.y, 1.0);
    rect1Color = mix(rect1Color, webcamSample, inRect);

    // Rectangle 3 colors
    vec3 rect3Deep = vec3(0.15, 0.06, 0.02);
    vec3 rect3Mid = vec3(0.4, 0.2, 0.08);
    vec3 rect3Light = vec3(0.6, 0.4, 0.15);
    vec3 rect3Silver = vec3(0.65, 0.55, 0.4);
    vec3 rect3Gold = vec3(0.75, 0.6, 0.35);

    vec3 rect3Color;
    if (h < 0.2) {
        rect3Color = mix(rect3Deep, rect3Mid, h / 0.2);
    } else if (h < 0.5) {
        rect3Color = mix(rect3Mid, rect3Light, (h - 0.2) / 0.3);
    } else if (h < 0.75) {
        rect3Color = mix(rect3Light, rect3Silver, (h - 0.5) / 0.25);
    } else {
        rect3Color = mix(rect3Silver, rect3Gold, (h - 0.75) / 0.25);
    }
    rect3Color = adjustColor(rect3Color, u_rect3Hue, u_rect3Sat, u_rect3Bright, u_rect3Contrast);

    // Rectangle 4 colors
    vec3 rect4Deep = vec3(0.03, 0.06, 0.2);
    vec3 rect4Mid = vec3(0.1, 0.2, 0.5);
    vec3 rect4Light = vec3(0.25, 0.4, 0.7);
    vec3 rect4Silver = vec3(0.45, 0.55, 0.7);
    vec3 rect4Gold = vec3(0.5, 0.6, 0.7);

    vec3 rect4Color;
    if (h < 0.2) {
        rect4Color = mix(rect4Deep, rect4Mid, h / 0.2);
    } else if (h < 0.5) {
        rect4Color = mix(rect4Mid, rect4Light, (h - 0.2) / 0.3);
    } else if (h < 0.75) {
        rect4Color = mix(rect4Light, rect4Silver, (h - 0.5) / 0.25);
    } else {
        rect4Color = mix(rect4Silver, rect4Gold, (h - 0.75) / 0.25);
    }
    rect4Color = adjustColor(rect4Color, u_rect4Hue, u_rect4Sat, u_rect4Bright, u_rect4Contrast);

    // Elevation fades
    float elevationFade1 = 1.0 - exp(-u_rect1Elevation * 30.0);
    float effectiveRect1Blend = v_rect1Blend * elevationFade1;

    float elevationFade3 = 1.0 - exp(-u_rect3Elevation * 30.0);
    float effectiveRect3Blend = v_rect3Blend * elevationFade3;

    float elevationFade4 = 1.0 - exp(-u_rect4Elevation * 30.0);
    float effectiveRect4Blend = v_rect4Blend * elevationFade4;

    // Grid patterns
    float centerFade1 = 1.0 - smoothstep(0.0, 0.8, v_distanceToRect1Center);
    float gridPattern1 = 0.0;
    if (effectiveRect1Blend > 0.5) {
        vec2 gridUV = v_uv * 50.0;
        gridPattern1 = step(0.92, fract(gridUV.x)) + step(0.92, fract(gridUV.y));
        gridPattern1 = min(gridPattern1, 1.0) * centerFade1 * 0.1 * elevationFade1;
    }
    rect1Color += rect1Color * gridPattern1;

    float centerFade3 = 1.0 - smoothstep(0.0, 0.8, v_distanceToRect3Center);
    float gridPattern3 = 0.0;
    if (effectiveRect3Blend > 0.5) {
        vec2 gridUV = v_uv * 45.0;
        gridPattern3 = step(0.91, fract(gridUV.x)) + step(0.91, fract(gridUV.y));
        gridPattern3 = min(gridPattern3, 1.0) * centerFade3 * 0.1 * elevationFade3;
    }
    rect3Color += rect3Color * gridPattern3;

    float centerFade4 = 1.0 - smoothstep(0.0, 0.8, v_distanceToRect4Center);
    float gridPattern4 = 0.0;
    if (effectiveRect4Blend > 0.5) {
        vec2 gridUV = v_uv * 50.0;
        gridPattern4 = step(0.92, fract(gridUV.x)) + step(0.92, fract(gridUV.y));
        gridPattern4 = min(gridPattern4, 1.0) * centerFade4 * 0.1 * elevationFade4;
    }
    rect4Color += rect4Color * gridPattern4;

    // Edge blending
    float edgeZone1 = smoothstep(0.6, 1.0, v_distanceToRect1Center) * effectiveRect1Blend;
    float terrainBleed1 = fbm(v_uv * 25.0 + u_time * 0.1) * 0.5 + 0.5;
    terrainBleed1 *= edgeZone1 * u_rect1Blend;
    rect1Color = mix(rect1Color, terrainColor, terrainBleed1);

    float edgeZone3 = smoothstep(0.6, 1.0, v_distanceToRect3Center) * effectiveRect3Blend;
    float terrainBleed3 = fbm(v_uv * 25.0 + u_time * 0.14) * 0.5 + 0.5;
    terrainBleed3 *= edgeZone3 * u_rect3Blend;
    rect3Color = mix(rect3Color, terrainColor, terrainBleed3);

    float edgeZone4 = smoothstep(0.6, 1.0, v_distanceToRect4Center) * effectiveRect4Blend;
    float terrainBleed4 = fbm(v_uv * 25.0 + u_time * 0.16) * 0.5 + 0.5;
    terrainBleed4 *= edgeZone4 * u_rect4Blend;
    rect4Color = mix(rect4Color, terrainColor, terrainBleed4);

    // Combine
    vec3 color = terrainColor;
    color = mix(color, rect1Color, effectiveRect1Blend);
    color = mix(color, rect3Color, effectiveRect3Blend);
    color = mix(color, rect4Color, effectiveRect4Blend);

    // Bubble effects
    float bubbleNoise = snoise(v_uv * 60.0 + u_time * 0.4);
    bubbleNoise = smoothstep(0.3, 0.7, bubbleNoise);

    float bubbleZone1 = edgeZone1 * (1.0 - edgeZone1) * 4.0;
    vec3 bubbleColor1 = mix(terrainColor, rect1Color, 0.5) * 1.2;
    color += bubbleColor1 * bubbleNoise * bubbleZone1 * u_rect1Blend * 0.25 * elevationFade1;

    float bubbleZone3 = edgeZone3 * (1.0 - edgeZone3) * 4.0;
    vec3 bubbleColor3 = mix(terrainColor, rect3Color, 0.5) * 1.2;
    color += bubbleColor3 * bubbleNoise * bubbleZone3 * u_rect3Blend * 0.25 * elevationFade3;

    float bubbleZone4 = edgeZone4 * (1.0 - edgeZone4) * 4.0;
    vec3 bubbleColor4 = mix(terrainColor, rect4Color, 0.5) * 1.2;
    color += bubbleColor4 * bubbleNoise * bubbleZone4 * u_rect4Blend * 0.25 * elevationFade4;

    // Lighting
    float diff = max(dot(normal, lightDir), 0.0);
    diff = diff * 0.5 + 0.5;

    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));
    vec3 halfDir = normalize(lightDir + viewDir);
    float maxRectBlend = max(effectiveRect1Blend * centerFade1,
                             max(effectiveRect3Blend * centerFade3, effectiveRect4Blend * centerFade4));
    float specPower = mix(24.0, 32.0, maxRectBlend);
    float spec = pow(max(dot(normal, halfDir), 0.0), specPower);
    vec3 specColor = vec3(0.7, 0.75, 0.9) * spec * 0.4;

    float rim = 1.0 - max(dot(viewDir, normal), 0.0);
    rim = pow(rim, 2.5) * 0.3;
    vec3 rimColor = color * rim;

    color = color * diff + specColor + rimColor;

    // Noise grain
    float noise = snoise(v_uv * 80.0) * 0.02;
    color += noise;

    // Edge fade
    float terrainEdgeFade = 1.0;
    float maxEffectiveBlend = max(effectiveRect1Blend, max(effectiveRect3Blend, effectiveRect4Blend));
    if (maxEffectiveBlend < 0.5) {
        vec2 center = vec2(0.5);
        vec2 ed = abs(v_uv - center);
        ed.x /= u_aspectRatio;
        float edgeDist = max(ed.x, ed.y);
        terrainEdgeFade = 1.0 - smoothstep(0.38, 0.48, edgeDist);
    }

    vec3 bgColor = vec3(0.03, 0.03, 0.05);
    color = mix(bgColor, color, max(terrainEdgeFade, maxEffectiveBlend));

    gl_FragColor = vec4(color, 1.0);
}
"""


# ─── HUD Overlay Shaders ───────────────────────────────────────────────────

HUD_VERT_SRC = """
#version 120
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

HUD_FRAG_SRC = """
#version 120
uniform sampler2D u_tex;
varying vec2 v_uv;
void main() {
    gl_FragColor = texture2D(u_tex, v_uv);
}
"""

HUD_WIDTH = 800
HUD_HEIGHT = 36


def render_hud_text(mode_str, rec_state, img_index, time_remaining, has_motion):
    img = np.zeros((HUD_HEIGHT, HUD_WIDTH, 4), dtype=np.uint8)
    img[:, :, 3] = 160  # semi-transparent dark background

    parts = [mode_str, rec_state]
    if rec_state != "IDLE":
        parts.append(f"imgs={img_index}")
    if time_remaining is not None:
        parts.append(f"timeout={time_remaining:.1f}s")
    if has_motion:
        parts.append("MOTION")
    text = "  |  ".join(parts)

    cv2.putText(img, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255, 255), 1, cv2.LINE_AA)
    # flip vertically for GL texture origin
    img = cv2.flip(img, 0)
    return img


# ─── Flask Routes & Recording Helpers ───────────────────────────────────────

def localhost_or_api_key(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if flask_request.remote_addr in ("127.0.0.1", "::1"):
            return f(*args, **kwargs)
        key = (flask_request.headers.get("X-API-Key")
               or flask_request.args.get("api_key"))
        if key != API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


def timestamp_folder_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def upload_sequence(seq_dir: Path, image_count: int):
    if image_count == 0:
        print("No images to upload.")
        return False
    image_files = sorted(seq_dir.glob("img_*.jpg"))
    actual_image_count = len(image_files)
    if actual_image_count == 0:
        print(f"No images found in {seq_dir}")
        return False
    file_handles = []
    try:
        print(f"Uploading {actual_image_count} images from {seq_dir.name}...")
        files = []
        for img_path in image_files:
            fh = open(img_path, "rb")
            file_handles.append(fh)
            files.append(("files", (img_path.name, fh, "image/jpeg")))
        response = requests.post(
            AWS_UPLOAD_URL,
            headers={"X-API-Key": AWS_UPLOAD_KEY},
            files=files,
            data={
                "sequence_name": seq_dir.name,
                "image_count": actual_image_count,
                "timestamp": datetime.now().isoformat(),
            },
            timeout=120,
        )
        if response.ok:
            print(f"Upload successful: {response.json()}")
            return True
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Upload error: {e}")
        return False
    finally:
        for fh in file_handles:
            fh.close()


def mjpeg_generator():
    while True:
        with jpeg_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
               + frame_bytes + b"\r\n")
        time.sleep(0.001)


@flask_app.get("/")
def index():
    return f"""<!doctype html><html><head><title>Herma Monitor</title>
<style>body{{font-family:system-ui;margin:24px}}img{{max-width:100%}}</style></head>
<body><h2>Herma Webcam Stream</h2>
<p>Recording timeout: {RECORDING_TIMEOUT}s | Capture interval: {CAPTURE_INTERVAL}s</p>
<img src="/stream"/></body></html>"""


@flask_app.get("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.post("/api/start")
@localhost_or_api_key
def api_start():
    global manual_record_command, recording_start_time
    with control_lock:
        manual_record_command = "start"
        recording_start_time = time.time()
    print(f"API: Start recording (timeout in {RECORDING_TIMEOUT}s)")
    return jsonify({"status": "ok", "action": "start_recording",
                    "timeout_seconds": RECORDING_TIMEOUT,
                    "output_dir": str(OUTPUT_DIR_API)})


@flask_app.post("/api/stop")
@localhost_or_api_key
def api_stop():
    global manual_record_command, recording_start_time
    with control_lock:
        manual_record_command = "stop"
        recording_start_time = None
    print("API: Stop recording requested")
    return jsonify({"status": "ok", "action": "stop_recording"})


@flask_app.post("/api/auto")
@localhost_or_api_key
def api_auto():
    global manual_record_command, recording_start_time
    with control_lock:
        manual_record_command = None
        recording_start_time = None
    return jsonify({"status": "ok", "action": "auto_mode"})


@flask_app.get("/api/status")
def api_status():
    with control_lock:
        mode = ("manual_record" if manual_record_command == "start" else
                "manual_stop" if manual_record_command == "stop" else "auto")
        time_remaining = None
        if recording_start_time is not None and manual_record_command == "start":
            time_remaining = max(0, RECORDING_TIMEOUT - (time.time() - recording_start_time))
        return jsonify({**current_status, "mode": mode,
                        "time_remaining": time_remaining})


def run_web_server():
    flask_app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)


# ─── Helper Functions ───────────────────────────────────────────────────────

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


# ─── Main ───────────────────────────────────────────────────────────────────
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
    """
    Uses:
      [display].display_primary_monitor (bool)
      [display].monitor_index (int, 0-based)
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
    
    
def main():
    # Parse optional camera index from command line
    cam_idx = CAM_INDEX
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [camera_index]")
            sys.exit(1)

    # Init GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)

    # Start in fullscreen on the primary monitor
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    win = glfw.create_window(mode.size.width, mode.size.height,
                             "Herma Webcam Shader", monitor, None)
    if not win:
        glfw.terminate()
        print("Failed to create window")
        sys.exit(1)

    glfw.make_context_current(win)
    glfw.swap_interval(1)

    # Track fullscreen state and saved windowed geometry for restore
    is_fullscreen = [True]
    windowed_pos = [100, 100]
    windowed_size = [WINDOW_W, WINDOW_H]

    # Open webcam
    cap = cv2.VideoCapture(cam_idx)
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

    # Apply same downscale that the render loop will use
    if DOWNSCALE_WIDTH is not None:
        fh, fw = frame.shape[:2]
        if fw > DOWNSCALE_WIDTH:
            scale = DOWNSCALE_WIDTH / float(fw)
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                               interpolation=cv2.INTER_AREA)

    cam_h, cam_w = frame.shape[:2]
    print(f"Camera opened: {cam_w}x{cam_h} (index {cam_idx})")

    # Adjust rect1 aspect ratio to match camera
    cam_aspect = cam_w / cam_h
    area = P['rect1W'] * P['rect1H']
    P['rect1H'] = (area / cam_aspect) ** 0.5
    P['rect1W'] = P['rect1H'] * cam_aspect

    # Compile & link shaders
    prog = link_program(VERT_SRC, FRAG_SRC)

    # Cache uniform locations
    uniform_names = [
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
    u = {}
    for name in uniform_names:
        u[name] = glGetUniformLocation(prog, name)

    # Attribute locations
    pos_loc = glGetAttribLocation(prog, 'a_position')
    uv_loc = glGetAttribLocation(prog, 'a_uv')

    # Create grid mesh
    print(f"Building {GRID_RES}x{GRID_RES} grid mesh (aspect {TARGET_ASPECT:.2f})...")
    positions, uvs, indices = make_grid(GRID_RES, TARGET_ASPECT)
    num_indices = len(indices)

    vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)

    vbo_uv = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uv)
    glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Create webcam texture
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Upload initial frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    # ── HUD overlay resources ──
    hud_prog = link_program(HUD_VERT_SRC, HUD_FRAG_SRC)
    hud_pos_loc = glGetAttribLocation(hud_prog, 'a_pos')
    hud_uv_loc = glGetAttribLocation(hud_prog, 'a_uv')
    hud_tex_loc = glGetUniformLocation(hud_prog, 'u_tex')

    # Quad: full-width strip at bottom of screen (NDC)
    hud_quad = np.array([
        # x     y     u    v
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0, -0.92, 1.0, 1.0,
        -1.0, -0.92, 0.0, 1.0,
    ], dtype=np.float32)
    hud_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, hud_vbo)
    glBufferData(GL_ARRAY_BUFFER, hud_quad.nbytes, hud_quad, GL_STATIC_DRAW)

    hud_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, hud_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    # Allocate initial texture
    blank = np.zeros((HUD_HEIGHT, HUD_WIDTH, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, HUD_WIDTH, HUD_HEIGHT, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, blank)

    # Zoom & HUD state
    zoom = [1.0]
    show_hud = [True]

    def on_scroll(_win, _xoff, yoff):
        zoom[0] = max(0.5, min(3.0, zoom[0] + yoff * 0.1))

    def on_key(_win, key, _sc, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(win, True)
        if key == glfw.KEY_TAB and action == glfw.PRESS:
            mon = glfw.get_primary_monitor()
            vid = glfw.get_video_mode(mon)
            if is_fullscreen[0]:
                # Switch to windowed, restore saved size/position
                glfw.set_window_monitor(win, None,
                                        windowed_pos[0], windowed_pos[1],
                                        windowed_size[0], windowed_size[1], 0)
            else:
                # Save current windowed geometry before going fullscreen
                windowed_pos[0], windowed_pos[1] = glfw.get_window_pos(win)
                windowed_size[0], windowed_size[1] = glfw.get_window_size(win)
                glfw.set_window_monitor(win, mon, 0, 0,
                                        vid.size.width, vid.size.height,
                                        vid.refresh_rate)
            is_fullscreen[0] = not is_fullscreen[0]
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            show_hud[0] = not show_hud[0]

    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_key_callback(win, on_key)

    glEnable(GL_DEPTH_TEST)
    t0 = time.time()

    # ── Recording / motion state ──
    global latest_jpeg, manual_record_command, current_status, recording_start_time
    OUTPUT_DIR_API.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_AUTO.mkdir(parents=True, exist_ok=True)

    bg = None
    recording = False
    seq_dir = None
    last_motion_time = 0.0
    last_capture_time = 0.0
    img_index = 0
    last_stream_time = 0.0
    stream_interval = 1.0 / float(STREAM_FPS)
    api_triggered_recording = False
    last_cmd = None
    hud_mode = "AUTO"
    hud_state = "IDLE"
    hud_time_remaining = None
    hud_has_motion = False

    # Start Flask web server in background
    threading.Thread(target=run_web_server, daemon=True).start()

    print(f"Web server: http://{WEB_HOST}:{WEB_PORT}/")
    print(f"API endpoints: /api/start, /api/stop, /api/auto, /api/status")
    print(f"Upload URL: {AWS_UPLOAD_URL}")
    print(f"Recording timeout: {RECORDING_TIMEOUT}s, Capture interval: {CAPTURE_INTERVAL}s")
    print("Running. Press ESC to quit. Tab to toggle fullscreen. Scroll to zoom.")

    # ── Render loop ──
    while not glfw.window_should_close(win):
        glfw.poll_events()

        # Read webcam frame
        ret, frame = cap.read()
        if ret:
            # Downscale if needed
            if DOWNSCALE_WIDTH is not None:
                fh, fw = frame.shape[:2]
                if fw > DOWNSCALE_WIDTH:
                    scale = DOWNSCALE_WIDTH / float(fw)
                    frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                       interpolation=cv2.INTER_AREA)

            # ── Motion detection ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            has_motion = False
            if bg is None:
                bg = gray.astype("float")
            else:
                cv2.accumulateWeighted(gray, bg, 0.02)
                bg_uint8 = cv2.convertScaleAbs(bg)
                delta = cv2.absdiff(gray, bg_uint8)
                thresh = cv2.threshold(delta, MOTION_THRESHOLD, 255,
                                       cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) >= MIN_MOTION_AREA:
                        has_motion = True
                        break

            # ── Recording state machine ──
            now = time.time()

            with control_lock:
                cmd = manual_record_command
                rec_start = recording_start_time

            api_start_just_called = (cmd == "start" and last_cmd != "start")
            last_cmd = cmd

            if api_start_just_called and recording and not api_triggered_recording:
                print(f"Stopping motion recording for API recording. "
                      f"Saved {img_index} images to {seq_dir}")
                recording = False

            if cmd == "start" and rec_start is not None:
                if (now - rec_start) >= RECORDING_TIMEOUT:
                    print(f"Recording timeout after {RECORDING_TIMEOUT}s")
                    with control_lock:
                        manual_record_command = "stop"
                        recording_start_time = None
                    cmd = "stop"

            should_record = False
            is_api_trigger = False

            if cmd == "start":
                should_record = True
                is_api_trigger = True
                last_motion_time = now
            elif cmd == "stop":
                should_record = False
            else:
                if has_motion:
                    should_record = True
                    last_motion_time = now
                elif recording and (now - last_motion_time) < STILL_SECONDS_TO_STOP:
                    should_record = True

            # Start new sequence
            if should_record and not recording:
                recording = True
                img_index = 0
                last_capture_time = 0.0
                api_triggered_recording = is_api_trigger
                output_dir = OUTPUT_DIR_API if api_triggered_recording else OUTPUT_DIR_AUTO
                seq_dir = output_dir / timestamp_folder_name()
                seq_dir.mkdir(parents=True, exist_ok=True)
                trigger_type = "API" if api_triggered_recording else "MOTION"
                print(f"Started sequence: {seq_dir} (triggered by: {trigger_type})")

            # Stop sequence
            if not should_record and recording:
                recording = False
                print(f"Stopped sequence. Saved {img_index} images to {seq_dir}")
                if api_triggered_recording and seq_dir is not None:
                    threading.Thread(target=upload_sequence,
                                     args=(seq_dir, img_index), daemon=True).start()
                api_triggered_recording = False
                with control_lock:
                    if manual_record_command == "stop":
                        manual_record_command = None

            # Capture frame to disk
            if recording and seq_dir is not None:
                if (now - last_capture_time) >= CAPTURE_INTERVAL:
                    out_path = seq_dir / f"img_{img_index:05d}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    img_index += 1
                    last_capture_time = now

            # Update shared status
            time_remaining = None
            if rec_start is not None and cmd == "start":
                time_remaining = max(0, RECORDING_TIMEOUT - (now - rec_start))
            with control_lock:
                current_status = {
                    "recording": recording,
                    "sequence_dir": str(seq_dir) if seq_dir else None,
                    "images_captured": img_index,
                    "last_motion_time": last_motion_time,
                    "time_remaining": time_remaining,
                    "api_triggered": api_triggered_recording,
                }

            # Update HUD state
            hud_mode = "API" if cmd == "start" else ("STOP" if cmd == "stop" else "AUTO")
            if recording:
                hud_state = "REC (API)" if api_triggered_recording else "REC (MOTION)"
            else:
                hud_state = "IDLE"
            hud_time_remaining = time_remaining
            hud_has_motion = has_motion

            # Encode JPEG for MJPEG stream (throttled)
            now2 = time.time()
            if (now2 - last_stream_time) >= stream_interval:
                ok_j, buf = cv2.imencode(".jpg", frame,
                                         [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok_j:
                    with jpeg_lock:
                        latest_jpeg = buf.tobytes()
                last_stream_time = now2

            # Upload to GL texture
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 0)
            fh_gl, fw_gl = frame_rgb.shape[:2]
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fw_gl, fh_gl, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

        # Viewport
        fb_w, fb_h = glfw.get_framebuffer_size(win)
        if fb_w == 0 or fb_h == 0:
            glfw.poll_events()
            continue
        glViewport(0, 0, fb_w, fb_h)

        glClearColor(0.03, 0.03, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(prog)

        # Projection (orthographic, aspect-correct)
        aspect = fb_w / max(fb_h, 1)
        z = zoom[0]
        if aspect > 1:
            ol, or_, ob, ot = -z * aspect, z * aspect, -z, z
        else:
            ol, or_, ob, ot = -z, z, -z / aspect, z / aspect
        mvp = make_ortho(ol, or_, ob, ot, -10.0, 10.0)

        # Set all uniforms
        glUniformMatrix4fv(u['u_modelViewProjection'], 1, GL_FALSE, mvp)
        glUniform1f(u['u_time'], time.time() - t0)
        glUniform1f(u['u_heightScale'], P['heightScale'])
        glUniform1f(u['u_dripSpeed'], P['dripSpeed'])
        glUniform1f(u['u_distortion'], P['distortion'])
        glUniform1f(u['u_ringCount'], P['ringCount'])
        glUniform3f(u['u_lightDir'], 0.3, 0.3, 0.9)
        glUniform1f(u['u_aspectRatio'], TARGET_ASPECT)

        glUniform2f(u['u_rect1Pos'], P['rect1X'], P['rect1Y'])
        glUniform2f(u['u_rect1Size'], P['rect1W'], P['rect1H'])
        glUniform1f(u['u_rect1Elevation'], P['rect1Elev'])
        glUniform1f(u['u_rect1Blend'], P['rect1Blend'])
        glUniform1f(u['u_rect1Hue'], P['rect1Hue'])
        glUniform1f(u['u_rect1Sat'], P['rect1Sat'])
        glUniform1f(u['u_rect1Bright'], P['rect1Bright'])
        glUniform1f(u['u_rect1Contrast'], P['rect1Contrast'])

        glUniform2f(u['u_rect3Pos'], P['rect3X'], P['rect3Y'])
        glUniform2f(u['u_rect3Size'], P['rect3W'], P['rect3H'])
        glUniform1f(u['u_rect3Elevation'], P['rect3Elev'])
        glUniform1f(u['u_rect3Blend'], P['rect3Blend'])
        glUniform1f(u['u_rect3Hue'], P['rect3Hue'])
        glUniform1f(u['u_rect3Sat'], P['rect3Sat'])
        glUniform1f(u['u_rect3Bright'], P['rect3Bright'])
        glUniform1f(u['u_rect3Contrast'], P['rect3Contrast'])

        glUniform2f(u['u_rect4Pos'], P['rect4X'], P['rect4Y'])
        glUniform2f(u['u_rect4Size'], P['rect4W'], P['rect4H'])
        glUniform1f(u['u_rect4Elevation'], P['rect4Elev'])
        glUniform1f(u['u_rect4Blend'], P['rect4Blend'])
        glUniform1f(u['u_rect4Hue'], P['rect4Hue'])
        glUniform1f(u['u_rect4Sat'], P['rect4Sat'])
        glUniform1f(u['u_rect4Bright'], P['rect4Bright'])
        glUniform1f(u['u_rect4Contrast'], P['rect4Contrast'])

        glUniform1f(u['u_terrainHue'], P['terrainHue'])
        glUniform1f(u['u_terrainSat'], P['terrainSat'])
        glUniform1f(u['u_terrainBright'], P['terrainBright'])
        glUniform1f(u['u_terrainContrast'], P['terrainContrast'])

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(u['u_webcamTex'], 0)

        # Bind vertex data and draw
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_uv)
        glEnableVertexAttribArray(uv_loc)
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)

        # ── Draw HUD overlay ──
        if show_hud[0]:
            hud_img = render_hud_text(hud_mode, hud_state, img_index,
                                      hud_time_remaining, hud_has_motion)
            glBindTexture(GL_TEXTURE_2D, hud_tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, HUD_WIDTH, HUD_HEIGHT,
                            GL_RGBA, GL_UNSIGNED_BYTE, hud_img)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST)

            glUseProgram(hud_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, hud_tex)
            glUniform1i(hud_tex_loc, 0)

            glBindBuffer(GL_ARRAY_BUFFER, hud_vbo)
            glEnableVertexAttribArray(hud_pos_loc)
            glVertexAttribPointer(hud_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            glEnableVertexAttribArray(hud_uv_loc)
            glVertexAttribPointer(hud_uv_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

            glDisableVertexAttribArray(hud_pos_loc)
            glDisableVertexAttribArray(hud_uv_loc)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

        glfw.swap_buffers(win)

    # Cleanup
    cap.release()
    glfw.terminate()
    print("Done.")

if __name__ == "__main__":
    main()
