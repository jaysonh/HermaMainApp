"""GLSL shader sources for the Herma webcam shader."""

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

    // Shrink webcam image but keep it centered
    float webcamScale = 0.9;
    webcamUV = (webcamUV - 0.5) / webcamScale + 0.5;

    vec3 webcamSample = texture2D(u_webcamTex, clamp(webcamUV, 0.0, 1.0)).rgb;

    //float inRect = step(0.0, webcamUV.x) * step(webcamUV.x, 1.0)
    //             * step(0.0, webcamUV.y) * step(webcamUV.y, 1.0);

    // ---- Feathered mask ----
    // Distance to nearest edge in UV space (0 at edge, 0.5 at center)
    float edgeDist = min(min(webcamUV.x, 1.0 - webcamUV.x),
                         min(webcamUV.y, 1.0 - webcamUV.y));

    // Feather width in UV units (tweak this)
    float feather = 0.04;
    float inRect = smoothstep(0.0, feather, edgeDist);

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
