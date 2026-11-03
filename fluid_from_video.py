"""
Video-driven WebGL Fluid Simulation - Python port
Original: Copyright (c) 2017 Pavel Dobryakov (MIT License)
Ported to Python/moderngl, extended to be driven by an input video.

動画の「黒い部分」を流体の領域として切り抜き、
「動いている境界（エッジ）」が流体を押す／かき混ぜる。
  - 黒   : 流体が存在する領域
  - 白   : WHITE_MODE='solid'    -> 流体が入れない壁（動く壁として流体を押す）。
                                    表示は真っ黒。ただし周囲の流体の発光(bloom)だけ壁に染み出す
           WHITE_MODE='boundary' -> 壁にはならず、境界の動きだけが流体に作用
  - グレー: 反応が弱くなる（GRAY_POWER で調整）
出力の解像度は入力動画と同じ。FPS は OUTPUT_FPS / --fps で指定。

Usage:
    python fluid_from_video.py input.mp4 -o out.mp4 --fps 30 --white-mode solid

Requirements:
    pip install moderngl numpy imageio imageio-ffmpeg
"""
import os
import math
import argparse
import numpy as np
import moderngl
import imageio.v2 as imageio

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CONFIG = dict(
    # ---- 入出力 -------------------------------------------------------------
    INPUT_FILE=r"input.mp4",
    OUTPUT_FILE="fluid_output.mp4",
    OUTPUT_FPS=60,            # None = 元動画と同じFPS
    OUTPUT_QUALITY=9,           # 0-10 (imageio-ffmpeg)。高いほど高画質

    # ---- 動画連動の設定 -----------------------------------------------------
    WHITE_MODE="boundary",         # 'solid'    : 白い部分は流体が入れない壁 (表示は黒、光だけ染み出す)
                                # 'boundary' : 白い部分は壁にならず、境界の動きにだけ反応
    SOLID_THRESHOLD=0.5,        # 輝度がこれ以上なら「白（壁）」とみなす (solid時)
    GRAY_POWER=2.0,             # 大きいほどグレー(低コントラスト)の反応が弱くなる。0で白黒と同じ
    NOISE_FLOOR=0.005,           # 動画ノイズを動きとみなさないための輝度差の下限
    MAX_DISPLACEMENT=15.0,       # 1フレームあたりの境界移動量の上限 (シム解像度でのテクセル)
    FORCE_SCALE=5.0,            # 境界の動きが流体速度に与える強さ(押し出しの強さ)
    FORCE_RATE=60.0,            # 境界の動きに流体速度が追従する速さ (1/秒)
    EMIT_DYE=True,              # 動く境界から色を出す
    EMIT_RATE=150.0,             # 色の出る量 (旧4.0では暗すぎて見えなかった)
    EMIT_SPEED_REF=1.0,        # この速度(テクセル/秒)で色の出る量が最大になる
    EMIT_MAX=0.8,               # 染料の明るさの上限 (溜まりすぎ防止)
    WALL_GLOW=1.0,              # solid時: 白い部分へ染み出す光(bloom)の強さ。0で完全な黒
    CLIP_TO_BLACK=True,         # True: 黒い部分にだけ流体を表示 (白は切り抜かれる)
    BACKGROUND="video",         # boundaryモード時の切り抜かれた部分: 'video'=元動画 / 'color'=BG_COLOR
                                # (solidモードでは常に黒)
    BG_COLOR=(255, 255, 255),
    INITIAL_SPLATS=8,           # 最初に黒い部分へ散らす色の数
    SUBSTEPS=1,                 # 1フレームあたりのシミュレーション分割数 (低FPS時に増やすと安定)

    # ---- 流体シミュレーション (元コードと同じ) -------------------------------
    SIM_RESOLUTION=256,         # 境界を追従させるため元(128)より高め
    DYE_RESOLUTION=None,        # None = 元動画の解像度に合わせる
    DENSITY_DISSIPATION=1.2,
    VELOCITY_DISSIPATION=0.15,
    PRESSURE=0.8,
    PRESSURE_ITERATIONS=20,
    CURL=30,
    SPLAT_RADIUS=0.2,
    SPLAT_FORCE=6000,
    SHADING=True,
    COLORFUL=True,
    COLOR_UPDATE_SPEED=10,
    BLOOM=True,
    BLOOM_ITERATIONS=8,
    BLOOM_RESOLUTION=256,
    BLOOM_INTENSITY=0.8,
    BLOOM_THRESHOLD=0.6,
    BLOOM_SOFT_KNEE=0.7,
    SUNRAYS=True,
    SUNRAYS_RESOLUTION=196,
    SUNRAYS_WEIGHT=1.0,
    BACK_COLOR=(0, 0, 0),       # 流体の背面色 (0-255)
)

# ----------------------------------------------------------------------------
# GLSL shaders
# ----------------------------------------------------------------------------
BASE_VERT = """
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;
void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
"""

BLUR_VERT = """
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
uniform vec2 texelSize;
void main () {
    vUv = aPosition * 0.5 + 0.5;
    float offset = 1.33333333;
    vL = vUv - texelSize * offset;
    vR = vUv + texelSize * offset;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
"""

BLUR_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
uniform sampler2D uTexture;
void main () {
    vec4 sum = texture2D(uTexture, vUv) * 0.29411764;
    sum += texture2D(uTexture, vL) * 0.35294117;
    sum += texture2D(uTexture, vR) * 0.35294117;
    gl_FragColor = sum;
}
"""

CLEAR_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () { gl_FragColor = value * texture2D(uTexture, vUv); }
"""

BLOOM_PREFILTER_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform vec3 curve;
uniform float threshold;
void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;
    float br = max(c.r, max(c.g, c.b));
    float rq = clamp(br - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;
    c *= max(rq, br - threshold) / max(br, 0.0001);
    gl_FragColor = vec4(c, 0.0);
}
"""

BLOOM_BLUR_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
void main () {
    vec4 sum = vec4(0.0);
    sum += texture2D(uTexture, vL);
    sum += texture2D(uTexture, vR);
    sum += texture2D(uTexture, vT);
    sum += texture2D(uTexture, vB);
    sum *= 0.25;
    gl_FragColor = sum;
}
"""

BLOOM_FINAL_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
uniform float intensity;
void main () {
    vec4 sum = vec4(0.0);
    sum += texture2D(uTexture, vL);
    sum += texture2D(uTexture, vR);
    sum += texture2D(uTexture, vT);
    sum += texture2D(uTexture, vB);
    sum *= 0.25;
    gl_FragColor = sum * intensity;
}
"""

SUNRAYS_MASK_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
void main () {
    vec4 c = texture2D(uTexture, vUv);
    float br = max(c.r, max(c.g, c.b));
    c.a = 1.0 - min(max(br * 20.0, 0.0), 0.8);
    gl_FragColor = c;
}
"""

SUNRAYS_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float weight;
#define ITERATIONS 16
void main () {
    float Density = 0.3;
    float Decay = 0.95;
    float Exposure = 0.7;
    vec2 coord = vUv;
    vec2 dir = vUv - 0.5;
    dir *= 1.0 / float(ITERATIONS) * Density;
    float illuminationDecay = 1.0;
    float color = texture2D(uTexture, vUv).a;
    for (int i = 0; i < ITERATIONS; i++) {
        coord -= dir;
        float col = texture2D(uTexture, coord).a;
        color += col * illuminationDecay * weight;
        illuminationDecay *= Decay;
    }
    gl_FragColor = vec4(color * Exposure, 0.0, 0.0, 1.0);
}
"""

SPLAT_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
void main () {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    gl_FragColor = vec4(base + splat, 1.0);
}
"""

ADVECTION_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    vec4 result = texture2D(uSource, coord);
    float decay = 1.0 + dissipation * dt;
    gl_FragColor = result / decay;
}
"""

DIVERGENCE_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
    if (vL.x < 0.0) { L = -C.x; }
    if (vR.x > 1.0) { R = -C.x; }
    if (vT.y > 1.0) { T = -C.y; }
    if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}
"""

CURL_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}
"""

VORTICITY_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity += force * dt;
    velocity = min(max(velocity, -1000.0), 1000.0);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}
"""

# 壁(solid)に隣接するセルは Neumann 条件 (壁側の圧力 = 自セルの圧力) として扱う
PRESSURE_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform sampler2D uMask;
void main () {
    float C = texture2D(uPressure, vUv).x;
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    if (texture2D(uMask, vL).y > 0.5) { L = C; }
    if (texture2D(uMask, vR).y > 0.5) { R = C; }
    if (texture2D(uMask, vT).y > 0.5) { T = C; }
    if (texture2D(uMask, vB).y > 0.5) { B = C; }
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}
"""

GRADIENT_SUBTRACT_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform sampler2D uMask;
void main () {
    float C = texture2D(uPressure, vUv).x;
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    if (texture2D(uMask, vL).y > 0.5) { L = C; }
    if (texture2D(uMask, vR).y > 0.5) { R = C; }
    if (texture2D(uMask, vT).y > 0.5) { T = C; }
    if (texture2D(uMask, vB).y > 0.5) { B = C; }
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}
"""

# ---- 動画連動用シェーダ -----------------------------------------------------

# 現フレームの輝度(R)と、壁判定(G)をシム解像度で作る。ミップマップで縮小するのでエイリアスが出にくい
MASK_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uFrame;
uniform float solidMode;
uniform float threshold;
void main () {
    vec3 c = texture2D(uFrame, vUv).rgb;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    float s = 0.0;
    if (solidMode > 0.5) { s = step(threshold, l); }
    gl_FragColor = vec4(l, s, 0.0, 1.0);
}
"""

# 前フレームとの輝度差と空間勾配から、エッジの法線方向の動き(normal flow)を推定する。
#   rg : 速度 (シム解像度のテクセル/秒)
#   b  : 反応の強さ 0..1  (エッジのコントラスト^GRAY_POWER × 動きのゲート)
MOTION_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uCurr;
uniform sampler2D uPrev;
uniform float invDt;
uniform float noiseFloor;
uniform float grayPower;
uniform float maxDisp;
void main () {
    float curC = texture2D(uCurr, vUv).x;
    float dL = curC - texture2D(uPrev, vUv).x;
    float C = 0.5 * (curC + texture2D(uPrev, vUv).x);
    float L = 0.5 * (texture2D(uCurr, vL).x + texture2D(uPrev, vL).x);
    float R = 0.5 * (texture2D(uCurr, vR).x + texture2D(uPrev, vR).x);
    float T = 0.5 * (texture2D(uCurr, vT).x + texture2D(uPrev, vT).x);
    float B = 0.5 * (texture2D(uCurr, vB).x + texture2D(uPrev, vB).x);
    float gx = 0.5 * (R - L);
    float gy = 0.5 * (T - B);
    float g2 = gx * gx + gy * gy;
    vec2 v = -dL * vec2(gx, gy) / (g2 + 0.0005);
    float len = length(v);
    if (len > maxDisp) { v *= maxDisp / len; }
    float mx = max(max(max(C, L), max(R, T)), B);
    float mn = min(min(min(C, L), min(R, T)), B);
    float range = clamp(mx - mn, 0.0001, 1.0);
    float gate = smoothstep(noiseFloor, noiseFloor * 2.0, abs(dL));
    float w = pow(range, grayPower) * gate;
    gl_FragColor = vec4(v * invDt, w, 1.0);
}
"""

# 動く境界に速度を追従させる。solidMode時は壁の内部では壁の速度に固定する。
# rate=0 で呼べば「壁の再適用」だけを行う (圧力投影後に使用)
FORCE_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uMotion;
uniform sampler2D uMask;
uniform float dt;
uniform float rate;
uniform float scale;
uniform float solidMode;
void main () {
    vec2 v = texture2D(uVelocity, vUv).xy;
    vec3 m = texture2D(uMotion, vUv).xyz;
    float k = 1.0 - exp(-rate * dt * m.z);
    v = mix(v, m.xy * scale, k);
    if (solidMode > 0.5 && texture2D(uMask, vUv).y > 0.5) {
        v = m.xy * m.z * scale;
    }
    gl_FragColor = vec4(v, 0.0, 1.0);
}
"""

# 動く境界から染料を出し、solidMode時は壁の中の染料を消す (染料解像度で処理)
DYE_UPDATE_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uSource;
uniform sampler2D uMotion;
uniform sampler2D uFrame;
uniform vec3 color;
uniform float emit;
uniform float emitMax;
uniform float speedRef;
uniform float solidMode;
uniform float threshold;
void main () {
    vec3 d = texture2D(uSource, vUv).rgb;
    vec3 m = texture2D(uMotion, vUv).xyz;
    float sp = clamp(length(m.xy) / speedRef, 0.0, 1.0);
    d += color * emit * m.z * sp;
    d = min(d, vec3(emitMax));
    if (solidMode > 0.5) {
        vec3 f = texture2D(uFrame, vUv).rgb;
        float l = dot(f, vec3(0.299, 0.587, 0.114));
        d *= 1.0 - smoothstep(threshold - 0.05, threshold + 0.05, l);
    }
    gl_FragColor = vec4(d, 1.0);
}
"""

DISPLAY_FRAG_TEMPLATE = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
uniform sampler2D uBloom;
uniform sampler2D uSunrays;
uniform sampler2D uDithering;
uniform sampler2D uFrame;
uniform vec2 ditherScale;
uniform vec2 texelSize;
uniform float clipMix;
uniform float bgMix;
uniform float solidMode;
uniform float wallGlow;
uniform vec3 bgColor;
uniform vec3 backColor;

vec3 linearToGamma (vec3 color) {
    color = max(color, vec3(0));
    return max(1.055 * pow(color, vec3(0.416666667)) - 0.055, vec3(0));
}

void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;
    vec3 glow = vec3(0.0);

{SHADING}

{BLOOM_DECL}

{SUNRAYS}

{BLOOM_COMPOSITE}

    // 流体を黒い部分にだけ表示 (切り抜き)
    float a = clamp(max(c.r, max(c.g, c.b)), 0.0, 1.0);
    vec3 fluid = c + backColor * (1.0 - a);
    vec3 src = texture2D(uFrame, vUv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));
    float vis = mix(1.0, clamp(1.0 - luma, 0.0, 1.0), clipMix);
    vec3 bg = mix(bgColor, src, bgMix);
    // solid時は白い部分(壁)を真っ黒にする
    bg = mix(bg, vec3(0.0), solidMode);
    vec3 outc = mix(bg, fluid, vis);
    // 壁の部分には、周囲の発光(bloom)だけを貫通させる
    outc += glow * wallGlow * (1.0 - vis) * solidMode;
    gl_FragColor = vec4(outc, 1.0);
}
"""

SHADING_BLOCK = """
    vec3 lc = texture2D(uTexture, vL).rgb;
    vec3 rc = texture2D(uTexture, vR).rgb;
    vec3 tc = texture2D(uTexture, vT).rgb;
    vec3 bc = texture2D(uTexture, vB).rgb;
    float dx = length(rc) - length(lc);
    float dy = length(tc) - length(bc);
    vec3 n = normalize(vec3(dx, dy, length(texelSize)));
    vec3 l = vec3(0.0, 0.0, 1.0);
    float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
    c *= diffuse;
"""

BLOOM_BLOCK = """
    vec3 bloom = texture2D(uBloom, vUv).rgb;
"""

SUNRAYS_BLOCK = """
    float sunrays = texture2D(uSunrays, vUv).r;
    c *= sunrays;
"""

BLOOM_AFTER_SUNRAYS = """
    bloom *= sunrays;
"""

BLOOM_COMPOSITE = """
    float noise = texture2D(uDithering, vUv * ditherScale).r;
    noise = noise * 2.0 - 1.0;
    bloom += noise / 255.0;
    bloom = linearToGamma(bloom);
    c += bloom;
    glow = bloom;
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def hsv_to_rgb(h, s, v):
    i = int(math.floor(h * 6))
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    if i == 0:   r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return np.array([r, g, b], dtype=np.float32)


def generate_color():
    return hsv_to_rgb(np.random.random(), 1.0, 1.0) * 0.15


def get_resolution(resolution, aspect_ratio):
    landscape = aspect_ratio >= 1
    ar = aspect_ratio if landscape else 1.0 / aspect_ratio
    mn = round(resolution)
    mx = round(resolution * ar)
    return (mx, mn) if landscape else (mn, mx)


def create_fbo(ctx, width, height, filtering=True):
    texture = ctx.texture((width, height), 4, dtype='f2')
    f = moderngl.LINEAR if filtering else moderngl.NEAREST
    texture.filter = (f, f)
    texture.repeat_x = False
    texture.repeat_y = False
    fbo = ctx.framebuffer(color_attachments=[texture])
    fbo.use()
    fbo.clear(0, 0, 0, 1)
    return {'texture': texture, 'fbo': fbo, 'width': width, 'height': height,
            'texel_size': (1.0 / width, 1.0 / height)}


def create_double_fbo(ctx, w, h, filtering=True):
    a = create_fbo(ctx, w, h, filtering)
    b = create_fbo(ctx, w, h, filtering)
    return {'read': a, 'write': b, 'width': w, 'height': h,
            'texel_size': a['texel_size']}


def swap(d):
    d['read'], d['write'] = d['write'], d['read']


def fvec(v):
    return tuple(float(x) for x in v)


# ----------------------------------------------------------------------------
# Main simulation class
# ----------------------------------------------------------------------------
class FluidSimulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.aspect = width / height
        self.solid_mode = 1.0 if CONFIG['WHITE_MODE'] == 'solid' else 0.0
        try:
            self.ctx = moderngl.create_standalone_context()
        except Exception:                       # ディスプレイが無い環境(サーバ等)向け
            self.ctx = moderngl.create_standalone_context(backend='egl')
        self.ctx.viewport = (0, 0, width, height)

        # 出力用のオフスクリーン
        self.screen_tex = self.ctx.texture((width, height), 4, dtype='f1')
        self.screen_fbo = self.ctx.framebuffer(color_attachments=[self.screen_tex])

        quad = np.array([-1, -1, -1, 1, 1, 1, 1, -1], dtype='f4')
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())

        self.programs = {}
        self._build_programs()

        sim_w, sim_h = get_resolution(CONFIG['SIM_RESOLUTION'], self.aspect)
        if CONFIG['DYE_RESOLUTION']:
            dye_w, dye_h = get_resolution(CONFIG['DYE_RESOLUTION'], self.aspect)
        else:
            dye_w, dye_h = width, height          # 元動画の解像度に合わせる

        self.velocity = create_double_fbo(self.ctx, sim_w, sim_h)
        self.dye = create_double_fbo(self.ctx, dye_w, dye_h)
        self.divergence = create_fbo(self.ctx, sim_w, sim_h, filtering=False)
        self.curl = create_fbo(self.ctx, sim_w, sim_h, filtering=False)
        self.pressure = create_double_fbo(self.ctx, sim_w, sim_h, filtering=False)

        # 動画由来のバッファ
        self.frame_tex = self.ctx.texture((width, height), 3, dtype='f1')
        self.frame_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.frame_tex.repeat_x = False
        self.frame_tex.repeat_y = False
        self.mask_curr = create_fbo(self.ctx, sim_w, sim_h, filtering=False)  # R=輝度 G=壁
        self.mask_prev = create_fbo(self.ctx, sim_w, sim_h, filtering=False)
        self.motion = create_fbo(self.ctx, sim_w, sim_h, filtering=True)      # RG=速度 B=強さ

        self._init_bloom()
        self._init_sunrays()

        # ディザ用テクスチャ (ファイルが無ければ中立グレー = ノイズなし)
        dither_path = 'LDR_LLL1_0.png'
        if os.path.exists(dither_path):
            img = imageio.imread(dither_path).astype(np.float32) / 255.0
            self.dither = self.ctx.texture((img.shape[1], img.shape[0]), 3,
                                           img[..., :3].tobytes(), dtype='f4')
        else:
            self.dither = self.ctx.texture((1, 1), 3, b'\x80\x80\x80', dtype='f1')
        self.dither.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.dither.repeat_x = True
        self.dither.repeat_y = True

        self.color_update_timer = 0.0
        self.pointer_color = generate_color()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_glsl330(src, is_frag):
        """WebGL(GLSL ES 1.0)風のソースを #version 330 に変換 (ドライバ差異を避ける)"""
        if is_frag:
            src = src.replace('gl_FragColor', 'fragColor').replace('texture2D(', 'texture(')
            head = "#version 330\n#define varying in\nout vec4 fragColor;\n"
        else:
            head = "#version 330\n#define attribute in\n#define varying out\n"
        return head + src

    def _make_program(self, frag, vert=BASE_VERT):
        prog = self.ctx.program(vertex_shader=self._to_glsl330(vert, False),
                                fragment_shader=self._to_glsl330(frag, True))
        vao = self.ctx.vertex_array(prog, [(self.vbo, '2f', 'aPosition')], self.ibo)
        return {'program': prog, 'vao': vao}

    def _build_programs(self):
        display_src = DISPLAY_FRAG_TEMPLATE
        display_src = display_src.replace('{SHADING}', SHADING_BLOCK if CONFIG['SHADING'] else '')
        display_src = display_src.replace('{BLOOM_DECL}', BLOOM_BLOCK if CONFIG['BLOOM'] else '')
        sr = SUNRAYS_BLOCK if CONFIG['SUNRAYS'] else ''
        if CONFIG['SUNRAYS'] and CONFIG['BLOOM']:
            sr += BLOOM_AFTER_SUNRAYS
        display_src = display_src.replace('{SUNRAYS}', sr)
        display_src = display_src.replace('{BLOOM_COMPOSITE}', BLOOM_COMPOSITE if CONFIG['BLOOM'] else '')

        mk = self._make_program
        self.programs = {
            'blur':       mk(BLUR_FRAG, vert=BLUR_VERT),
            'clear':      mk(CLEAR_FRAG),
            'bloom_pre':  mk(BLOOM_PREFILTER_FRAG),
            'bloom_blur': mk(BLOOM_BLUR_FRAG),
            'bloom_fin':  mk(BLOOM_FINAL_FRAG),
            'sun_mask':   mk(SUNRAYS_MASK_FRAG),
            'sun':        mk(SUNRAYS_FRAG),
            'splat':      mk(SPLAT_FRAG),
            'advection':  mk(ADVECTION_FRAG),
            'divergence': mk(DIVERGENCE_FRAG),
            'curl':       mk(CURL_FRAG),
            'vorticity':  mk(VORTICITY_FRAG),
            'pressure':   mk(PRESSURE_FRAG),
            'grad_sub':   mk(GRADIENT_SUBTRACT_FRAG),
            'mask':       mk(MASK_FRAG),
            'motion':     mk(MOTION_FRAG),
            'force':      mk(FORCE_FRAG),
            'dye_update': mk(DYE_UPDATE_FRAG),
            'display':    mk(display_src),
        }

    def _init_bloom(self):
        w, h = get_resolution(CONFIG['BLOOM_RESOLUTION'], self.aspect)
        self.bloom = create_fbo(self.ctx, w, h)
        self.bloom_fbos = []
        for i in range(CONFIG['BLOOM_ITERATIONS']):
            bw, bh = w >> (i + 1), h >> (i + 1)
            if bw < 2 or bh < 2:
                break
            self.bloom_fbos.append(create_fbo(self.ctx, bw, bh))

    def _init_sunrays(self):
        w, h = get_resolution(CONFIG['SUNRAYS_RESOLUTION'], self.aspect)
        self.sunrays = create_fbo(self.ctx, w, h)
        self.sunrays_temp = create_fbo(self.ctx, w, h)

    # ------------------------------------------------------------------
    def _bind_target(self, target):
        if target is None:
            self.screen_fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
        else:
            target['fbo'].use()
            self.ctx.viewport = (0, 0, target['width'], target['height'])

    def _run(self, key, target, tex=None, **uniforms):
        """テクスチャを unit 0.. に順に bind し、uniform を設定してフルスクリーン描画"""
        p = self.programs[key]
        prog = p['program']
        for unit, (name, t) in enumerate((tex or {}).items()):
            t.use(unit)
            if name in prog:
                prog[name].value = unit
        for name, val in uniforms.items():
            if name in prog:
                prog[name].value = val
        self._bind_target(target)
        p['vao'].render(moderngl.TRIANGLES, vertices=6)

    # ------------------------------------------------------------------
    # 動画フレームの取り込み
    # ------------------------------------------------------------------
    def set_frame(self, frame, dt_src, first=False):
        """frame: (H, W, 3) uint8 (上が先頭行)。dt_src: 前回取り込んだフレームとの時間差[秒]"""
        # GLの座標系(下が原点)に合わせて上下反転して転送
        self.frame_tex.write(np.ascontiguousarray(frame[::-1]).tobytes())
        self.frame_tex.build_mipmaps()

        mask_uniforms = dict(solidMode=self.solid_mode, threshold=float(CONFIG['SOLID_THRESHOLD']))
        tex = {'uFrame': self.frame_tex}
        if first:
            self._run('mask', self.mask_prev, tex, **mask_uniforms)
            self._run('mask', self.mask_curr, tex, **mask_uniforms)
            return
        self.mask_prev, self.mask_curr = self.mask_curr, self.mask_prev
        self._run('mask', self.mask_curr, tex, **mask_uniforms)
        self._run('motion', self.motion,
                  {'uCurr': self.mask_curr['texture'], 'uPrev': self.mask_prev['texture']},
                  texelSize=self.velocity['texel_size'],
                  invDt=1.0 / max(dt_src, 1e-6),
                  noiseFloor=float(CONFIG['NOISE_FLOOR']),
                  grayPower=float(CONFIG['GRAY_POWER']),
                  maxDisp=float(CONFIG['MAX_DISPLACEMENT']))

    # ------------------------------------------------------------------
    def splat(self, x, y, dx, dy, color):
        radius = CONFIG['SPLAT_RADIUS'] / 100.0
        if self.aspect > 1:
            radius *= self.aspect
        common = dict(aspectRatio=self.aspect, point=(float(x), float(y)), radius=float(radius))
        self._run('splat', self.velocity['write'], {'uTarget': self.velocity['read']['texture']},
                  color=(float(dx), float(dy), 0.0), **common)
        swap(self.velocity)
        self._run('splat', self.dye['write'], {'uTarget': self.dye['read']['texture']},
                  color=fvec(color), **common)
        swap(self.dye)

    def multiple_splats(self, amount, gray=None):
        """gray: (H, W) 0..1 の輝度。指定があれば暗い(黒い)場所にだけ色を置く"""
        xs = ys = np.array([])
        if gray is not None:
            ys, xs = np.nonzero(gray < 0.3)
        h, w = (gray.shape if gray is not None else (1, 1))
        for _ in range(amount):
            if len(xs):
                j = np.random.randint(len(xs))
                x, y = (xs[j] + 0.5) / w, 1.0 - (ys[j] + 0.5) / h
            else:
                x, y = np.random.random(), np.random.random()
            self.splat(x, y,
                       1000 * (np.random.random() - 0.5),
                       1000 * (np.random.random() - 0.5),
                       generate_color() * 10.0)

    # ------------------------------------------------------------------
    def _force(self, dt, rate):
        v = self.velocity
        self._run('force', v['write'],
                  {'uVelocity': v['read']['texture'],
                   'uMotion': self.motion['texture'],
                   'uMask': self.mask_curr['texture']},
                  dt=float(dt), rate=float(rate),
                  scale=float(CONFIG['FORCE_SCALE']), solidMode=self.solid_mode)
        swap(v)

    def step(self, dt):
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        v = self.velocity
        ts = v['texel_size']
        mask_tex = self.mask_curr['texture']

        # curl / vorticity
        self._run('curl', self.curl, {'uVelocity': v['read']['texture']}, texelSize=ts)
        self._run('vorticity', v['write'],
                  {'uVelocity': v['read']['texture'], 'uCurl': self.curl['texture']},
                  texelSize=ts, curl=float(CONFIG['CURL']), dt=float(dt))
        swap(v)

        # 動く境界(壁)の速度を流体に反映
        self._force(dt, CONFIG['FORCE_RATE'])

        # divergence
        self._run('divergence', self.divergence, {'uVelocity': v['read']['texture']}, texelSize=ts)

        # pressure
        pr = self.pressure
        self._run('clear', pr['write'], {'uTexture': pr['read']['texture']},
                  value=float(CONFIG['PRESSURE']))
        swap(pr)
        for _ in range(CONFIG['PRESSURE_ITERATIONS']):
            self._run('pressure', pr['write'],
                      {'uDivergence': self.divergence['texture'],
                       'uPressure': pr['read']['texture'],
                       'uMask': mask_tex},
                      texelSize=ts)
            swap(pr)

        # gradient subtract
        self._run('grad_sub', v['write'],
                  {'uPressure': pr['read']['texture'],
                   'uVelocity': v['read']['texture'],
                   'uMask': mask_tex},
                  texelSize=ts)
        swap(v)

        # 壁の内部の速度を再適用 (rate=0 なので壁の固定だけ)
        if self.solid_mode > 0.5:
            self._force(dt, 0.0)

        # advect velocity
        self._run('advection', v['write'],
                  {'uVelocity': v['read']['texture'], 'uSource': v['read']['texture']},
                  texelSize=ts, dt=float(dt), dissipation=float(CONFIG['VELOCITY_DISSIPATION']))
        swap(v)

        # advect dye
        d = self.dye
        self._run('advection', d['write'],
                  {'uVelocity': v['read']['texture'], 'uSource': d['read']['texture']},
                  texelSize=ts, dt=float(dt), dissipation=float(CONFIG['DENSITY_DISSIPATION']))
        swap(d)

        # 動く境界から色を出す + 壁の中の染料を消す
        emit = CONFIG['EMIT_RATE'] * dt if CONFIG['EMIT_DYE'] else 0.0
        self._run('dye_update', d['write'],
                  {'uSource': d['read']['texture'],
                   'uMotion': self.motion['texture'],
                   'uFrame': self.frame_tex},
                  color=fvec(self.pointer_color), emit=float(emit),
                  emitMax=float(CONFIG['EMIT_MAX']),
                  speedRef=float(CONFIG['EMIT_SPEED_REF']),
                  solidMode=self.solid_mode, threshold=float(CONFIG['SOLID_THRESHOLD']))
        swap(d)

    # ------------------------------------------------------------------
    def _blur(self, target, temp, iterations=1):
        for _ in range(iterations):
            self._run('blur', temp, {'uTexture': target['texture']},
                      texelSize=(target['texel_size'][0], 0.0))
            self._run('blur', target, {'uTexture': temp['texture']},
                      texelSize=(0.0, target['texel_size'][1]))

    def _apply_bloom(self, source, destination):
        if len(self.bloom_fbos) < 2:
            return
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        knee = CONFIG['BLOOM_THRESHOLD'] * CONFIG['BLOOM_SOFT_KNEE'] + 0.0001
        self._run('bloom_pre', destination, {'uTexture': source['texture']},
                  curve=(CONFIG['BLOOM_THRESHOLD'] - knee, knee * 2, 0.25 / knee),
                  threshold=float(CONFIG['BLOOM_THRESHOLD']))
        last = destination
        for fb in self.bloom_fbos:
            self._run('bloom_blur', fb, {'uTexture': last['texture']}, texelSize=last['texel_size'])
            last = fb

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.bloom_fbos) - 2, -1, -1):
            base = self.bloom_fbos[i]
            self._run('bloom_blur', base, {'uTexture': last['texture']}, texelSize=last['texel_size'])
            last = base
        ctx.disable(moderngl.BLEND)

        self._run('bloom_fin', destination, {'uTexture': last['texture']},
                  texelSize=last['texel_size'], intensity=float(CONFIG['BLOOM_INTENSITY']))

    def _apply_sunrays(self, source, mask, destination):
        self.ctx.disable(moderngl.BLEND)
        self._run('sun_mask', mask, {'uTexture': source['texture']})
        self._run('sun', destination, {'uTexture': mask['texture']},
                  weight=float(CONFIG['SUNRAYS_WEIGHT']))

    def render_to_screen(self):
        self.ctx.disable(moderngl.BLEND)
        if CONFIG['BLOOM']:
            self._apply_bloom(self.dye['read'], self.bloom)
        if CONFIG['SUNRAYS']:
            self._apply_sunrays(self.dye['read'], self.dye['write'], self.sunrays)
            self._blur(self.sunrays, self.sunrays_temp, 1)

        solid = self.solid_mode > 0.5
        tex = {'uTexture': self.dye['read']['texture'], 'uFrame': self.frame_tex}
        uniforms = dict(
            texelSize=(1.0 / self.width, 1.0 / self.height),
            clipMix=1.0 if solid else 0.0,
            bgMix=1.0 if CONFIG['BACKGROUND'] == 'video' else 0.0,
            solidMode=self.solid_mode,
            wallGlow=float(CONFIG['WALL_GLOW']),
            bgColor=fvec(c / 255.0 for c in CONFIG['BG_COLOR']),
            backColor=fvec(c / 255.0 for c in CONFIG['BACK_COLOR']),
        )
        if CONFIG['BLOOM']:
            tex['uBloom'] = self.bloom['texture']
            tex['uDithering'] = self.dither
            uniforms['ditherScale'] = (self.width / self.dither.width,
                                       self.height / self.dither.height)
        if CONFIG['SUNRAYS']:
            tex['uSunrays'] = self.sunrays['texture']
        self._run('display', None, tex, **uniforms)

    def read_pixels(self):
        data = self.screen_fbo.read(components=3, dtype='f1')
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.flipud(img)

    # ------------------------------------------------------------------
    def update_colors(self, dt):
        if not CONFIG['COLORFUL']:
            return
        self.color_update_timer += dt * CONFIG['COLOR_UPDATE_SPEED']
        if self.color_update_timer >= 1:
            self.color_update_timer -= 1
            self.pointer_color = generate_color()


# ----------------------------------------------------------------------------
# Video helpers / entry point
# ----------------------------------------------------------------------------
def iter_frames(reader, first, src_fps, out_fps, n_out):
    """出力フレーム i に対応する元動画フレームを順に返す (FPSが違っても時間で対応づける)"""
    cur, cur_idx, i = first, 0, 0
    while n_out is None or i < n_out:
        target = int(i / out_fps * src_fps + 1e-6)
        while cur_idx < target:
            try:
                cur = reader.get_next_data()
            except Exception:
                return
            cur_idx += 1
        yield i, cur_idx, cur
        i += 1


def parse_args():
    ap = argparse.ArgumentParser(description="Video-driven fluid simulation")
    ap.add_argument('input', nargs='?', default=CONFIG['INPUT_FILE'])
    ap.add_argument('-o', '--output', default=CONFIG['OUTPUT_FILE'])
    ap.add_argument('--fps', type=float, default=CONFIG['OUTPUT_FPS'],
                    help='出力FPS (省略時は元動画と同じ)')
    ap.add_argument('--white-mode', choices=['solid', 'boundary'], default=CONFIG['WHITE_MODE'],
                    help="solid: 白は流体が入れない壁 / boundary: 境界の動きにだけ反応")
    return ap.parse_args()


def main():
    args = parse_args()
    CONFIG['WHITE_MODE'] = args.white_mode
    np.random.seed(42)

    reader = imageio.get_reader(args.input, 'ffmpeg')
    meta = reader.get_meta_data()
    src_fps = float(meta['fps'])
    out_fps = float(args.fps or src_fps)

    first = reader.get_next_data()
    h0, w0 = first.shape[:2]
    W, H = w0 - w0 % 2, h0 - h0 % 2            # H.264 は偶数サイズが必要 (奇数なら1px切り落とし)
    crop = lambda f: f[:H, :W, :3]

    duration = meta.get('duration')
    n_out = None
    if duration and np.isfinite(duration):
        n_out = int(duration * out_fps + 1e-6)

    print(f"input : {args.input}  {W}x{H} @ {src_fps:.3f}fps")
    print(f"output: {args.output}  {W}x{H} @ {out_fps:.3f}fps  mode={CONFIG['WHITE_MODE']}")

    sim = FluidSimulation(W, H)
    f0 = crop(first)
    sim.set_frame(f0, 1.0 / src_fps, first=True)
    gray = f0.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32) / 255.0
    sim.multiple_splats(CONFIG['INITIAL_SPLATS'], gray)

    writer = imageio.get_writer(args.output, fps=out_fps, codec='libx264',
                                quality=CONFIG['OUTPUT_QUALITY'], macro_block_size=1,
                                pixelformat='yuv420p')
    dt = 1.0 / out_fps
    sub = max(1, int(CONFIG['SUBSTEPS']))
    last_src = 0
    for i, src_idx, frame in iter_frames(reader, first, src_fps, out_fps, n_out):
        if src_idx != last_src:
            sim.set_frame(crop(frame), (src_idx - last_src) / src_fps)
            last_src = src_idx
        for _ in range(sub):
            sim.update_colors(dt / sub)
            sim.step(dt / sub)
        sim.render_to_screen()
        writer.append_data(sim.read_pixels())
        if i % 30 == 0:
            print(f"frame {i}/{n_out if n_out else '?'}")

    writer.close()
    reader.close()
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
