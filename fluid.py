"""
WebGL Fluid Simulation - Python port
Original: Copyright (c) 2017 Pavel Dobryakov (MIT License)
Ported to Python/moderngl with MP4 output.
"""
import os
import math
import numpy as np
import moderngl
import imageio.v2 as imageio

# ----------------------------------------------------------------------------
# Configuration (mirrors the JS `config` object)
# ----------------------------------------------------------------------------
CONFIG = dict(
    SIM_RESOLUTION=128,
    DYE_RESOLUTION=1024,
    DENSITY_DISSIPATION=1.0,
    VELOCITY_DISSIPATION=0.2,
    PRESSURE=0.8,
    PRESSURE_ITERATIONS=20,
    CURL=30,
    SPLAT_RADIUS=0.25,
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
    BACK_COLOR=(0, 0, 0),
)

# Video output settings
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
FPS = 60
DURATION_SECONDS = 10
OUTPUT_FILE = "fluid_simulation.mp4"

# ----------------------------------------------------------------------------
# GLSL shaders (ported verbatim from the JS source)
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

COPY_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
uniform sampler2D uTexture;
void main () { gl_FragColor = texture2D(uTexture, vUv); }
"""

CLEAR_FRAG = """
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () { gl_FragColor = value * texture2D(uTexture, vUv); }
"""

COLOR_FRAG = """
precision mediump float;
uniform vec4 color;
void main () { gl_FragColor = color; }
"""

CHECKERBOARD_FRAG = """
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float aspectRatio;
#define SCALE 25.0
void main () {
    vec2 uv = floor(vUv * SCALE * vec2(aspectRatio, 1.0));
    float v = mod(uv.x + uv.y, 2.0);
    v = v * 0.1 + 0.8;
    gl_FragColor = vec4(vec3(v), 1.0);
}
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
void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
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
void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
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
uniform vec2 ditherScale;
uniform vec2 texelSize;

vec3 linearToGamma (vec3 color) {
    color = max(color, vec3(0));
    return max(1.055 * pow(color, vec3(0.416666667)) - 0.055, vec3(0));
}

void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;

{SHADING}

{BLOOM_DECL}

{SUNRAYS}

{BLOOM_COMPOSITE}

    float a = max(c.r, max(c.g, c.b));
    gl_FragColor = vec4(c, a);
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
"""


# ----------------------------------------------------------------------------
# Helpers (ported from JS)
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
    c = hsv_to_rgb(np.random.random(), 1.0, 1.0)
    return c * 0.15


def get_resolution(resolution, aspect_ratio):
    landscape = aspect_ratio >= 1
    ar = aspect_ratio if landscape else 1.0 / aspect_ratio
    mn = round(resolution)
    mx = round(resolution * ar)
    return (mx, mn) if landscape else (mn, mx)


def create_fbo(ctx, width, height, filtering=True):
    texture = ctx.texture((width, height), 4, dtype='f2')
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR) if filtering else (moderngl.NEAREST, moderngl.NEAREST)
    texture.repeat_x = False
    texture.repeat_y = False
    fbo = ctx.framebuffer(color_attachments=[texture])
    fbo.use()
    fbo.clear(0, 0, 0, 1)
    return {
        'texture': texture, 'fbo': fbo, 'width': width, 'height': height,
        'texel_size': (1.0 / width, 1.0 / height),
    }


def create_double_fbo(ctx, w, h, filtering=True):
    f1 = create_fbo(ctx, w, h, filtering)
    f2 = create_fbo(ctx, w, h, filtering)
    return {'read': f1, 'write': f2, 'width': w, 'height': h,
            'texel_size': f1['texel_size'],
            'swap': lambda d: (d.__setitem__('read', d['write']),
                               d.__setitem__('write', f1 if d['read'] is not f1 else f2))}


# ----------------------------------------------------------------------------
# Main simulation class
# ----------------------------------------------------------------------------
class FluidSimulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.aspect = width / height
        self.ctx = moderngl.create_standalone_context()
        self.ctx.viewport = (0, 0, width, height)

        # Standalone contexts have no default framebuffer — make our own "screen"
        self.screen_tex = self.ctx.texture((width, height), 4, dtype='f1')
        self.screen_fbo = self.ctx.framebuffer(color_attachments=[self.screen_tex])

        # Fullscreen quad
        quad = np.array([-1, -1, -1, 1, 1, 1, 1, -1], dtype='f4')
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())

        self.programs = {}
        self._build_programs()

        sim_w, sim_h = get_resolution(CONFIG['SIM_RESOLUTION'], self.aspect)
        dye_w, dye_h = get_resolution(CONFIG['DYE_RESOLUTION'], self.aspect)

        self.velocity = create_double_fbo(self.ctx, sim_w, sim_h)
        self.dye = create_double_fbo(self.ctx, dye_w, dye_h)
        self.divergence = create_fbo(self.ctx, sim_w, sim_h, filtering=False)
        self.curl = create_fbo(self.ctx, sim_w, sim_h, filtering=False)
        self.pressure = create_double_fbo(self.ctx, sim_w, sim_h, filtering=False)

        self._init_bloom()
        self._init_sunrays()

        # Dithering texture (optional, defaults to white)
        dither_path = 'LDR_LLL1_0.png'
        if os.path.exists(dither_path):
            img = imageio.imread(dither_path).astype(np.float32) / 255.0
            self.dither = self.ctx.texture((img.shape[1], img.shape[0]), 3,
                                           img[..., :3].tobytes(), dtype='f4')
        else:
            self.dither = self.ctx.texture((1, 1), 3, b'\xff\xff\xff', dtype='f1')
        self.dither.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.dither.repeat_x = True
        self.dither.repeat_y = True

        self.color_update_timer = 0.0
        self.pointer_color = generate_color()

    def _make_program(self, frag, vert=BASE_VERT, defines=None):
        header = ''
        if defines:
            header = ''.join(f'#define {d}\n' for d in defines)
        prog = self.ctx.program(vertex_shader=vert, fragment_shader=header + frag)
        vao = self.ctx.vertex_array(prog, [(self.vbo, '2f', 'aPosition')], self.ibo)
        return {'program': prog, 'vao': vao}

    def _build_programs(self):
        shading = CONFIG['SHADING']
        bloom = CONFIG['BLOOM']
        sunrays = CONFIG['SUNRAYS']

        display_src = DISPLAY_FRAG_TEMPLATE
        display_src = display_src.replace('{SHADING}', SHADING_BLOCK if shading else '')

        # (1) declare bloom before sunrays
        display_src = display_src.replace('{BLOOM_DECL}', BLOOM_BLOCK if bloom else '')

        # (2) sunrays block; if bloom is on, also multiply it inside sunrays
        sr = SUNRAYS_BLOCK if sunrays else ''
        if sunrays and bloom:
            sr += BLOOM_AFTER_SUNRAYS
        display_src = display_src.replace('{SUNRAYS}', sr)

        # (3) composite bloom at the end
        display_src = display_src.replace('{BLOOM_COMPOSITE}', BLOOM_COMPOSITE if bloom else '')

        self.programs = {
            'blur':      self._make_program(BLUR_FRAG, vert=BLUR_VERT),
            'copy':      self._make_program(COPY_FRAG),
            'clear':     self._make_program(CLEAR_FRAG),
            'color':     self._make_program(COLOR_FRAG),
            'checker':   self._make_program(CHECKERBOARD_FRAG),
            'bloom_pre': self._make_program(BLOOM_PREFILTER_FRAG),
            'bloom_blur':self._make_program(BLOOM_BLUR_FRAG),
            'bloom_fin': self._make_program(BLOOM_FINAL_FRAG),
            'sun_mask':  self._make_program(SUNRAYS_MASK_FRAG),
            'sun':       self._make_program(SUNRAYS_FRAG),
            'splat':     self._make_program(SPLAT_FRAG),
            'advection': self._make_program(ADVECTION_FRAG),
            'divergence':self._make_program(DIVERGENCE_FRAG),
            'curl':      self._make_program(CURL_FRAG),
            'vorticity': self._make_program(VORTICITY_FRAG),
            'pressure':  self._make_program(PRESSURE_FRAG),
            'grad_sub':  self._make_program(GRADIENT_SUBTRACT_FRAG),
            'display':   self._make_program(display_src),
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

    def _blit(self, target, clear=False):
        if target is None:
            self.screen_fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.screen_fbo.clear(0, 0, 0, 1)
        else:
            target['fbo'].use()
            self.ctx.viewport = (0, 0, target['width'], target['height'])
            if clear:
                target['fbo'].clear(0, 0, 0, 1)

    def _draw(self, prog_dict, target, **uniforms):
        prog = prog_dict['program']
        for name, val in uniforms.items():
            if name in prog:
                prog[name].value = val
        self._blit(target)
        prog_dict['vao'].render(moderngl.TRIANGLES, vertices=6)

    # ------------------------------------------------------------------
    def splat(self, x, y, dx, dy, color):
        radius = CONFIG['SPLAT_RADIUS'] / 100.0
        if self.aspect > 1:
            radius *= self.aspect
        # velocity splat
        p = self.programs['splat']
        p['program']['aspectRatio'].value = self.aspect
        p['program']['point'].value = (x, y)
        p['program']['color'].value = (dx, dy, 0.0)
        p['program']['radius'].value = radius   # ← 変更
        self.velocity['read']['texture'].use(0)
        self._draw(p, self.velocity['write'])
        self.velocity['read'], self.velocity['write'] = self.velocity['write'], self.velocity['read']

        # dye splat
        p['program']['uTarget'].value = 0
        self.dye['read']['texture'].use(0)
        p['program']['color'].value = tuple(color)
        self._draw(p, self.dye['write'])
        self.dye['read'], self.dye['write'] = self.dye['write'], self.dye['read']

    def multiple_splats(self, amount):
        for _ in range(amount):
            color = generate_color() * 10.0
            self.splat(np.random.random(), np.random.random(),
                       1000 * (np.random.random() - 0.5),
                       1000 * (np.random.random() - 0.5), color)

    # ------------------------------------------------------------------
    def step(self, dt):
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)

        # curl
        p = self.programs['curl']
        p['program']['texelSize'].value = self.velocity['texel_size']
        p['program']['uVelocity'].value = 0
        self.velocity['read']['texture'].use(0)
        self._draw(p, self.curl)

        # vorticity
        p = self.programs['vorticity']
        p['program']['texelSize'].value = self.velocity['texel_size']
        p['program']['uVelocity'].value = 0
        p['program']['uCurl'].value = 1
        p['program']['curl'].value = CONFIG['CURL']
        p['program']['dt'].value = dt
        self.velocity['read']['texture'].use(0)
        self.curl['texture'].use(1)
        self._draw(p, self.velocity['write'])
        self.velocity['read'], self.velocity['write'] = self.velocity['write'], self.velocity['read']

        # divergence
        p = self.programs['divergence']
        p['program']['texelSize'].value = self.velocity['texel_size']
        p['program']['uVelocity'].value = 0
        self.velocity['read']['texture'].use(0)
        self._draw(p, self.divergence)

        # clear pressure
        p = self.programs['clear']
        p['program']['uTexture'].value = 0
        p['program']['value'].value = CONFIG['PRESSURE']
        self.pressure['read']['texture'].use(0)
        self._draw(p, self.pressure['write'])
        self.pressure['read'], self.pressure['write'] = self.pressure['write'], self.pressure['read']

        # pressure iterations
        p = self.programs['pressure']
        p['program']['texelSize'].value = self.velocity['texel_size']
        p['program']['uDivergence'].value = 0
        self.divergence['texture'].use(0)
        for _ in range(CONFIG['PRESSURE_ITERATIONS']):
            p['program']['uPressure'].value = 1
            self.pressure['read']['texture'].use(1)
            self._draw(p, self.pressure['write'])
            self.pressure['read'], self.pressure['write'] = self.pressure['write'], self.pressure['read']

        # gradient subtract
        p = self.programs['grad_sub']
        p['program']['texelSize'].value = self.velocity['texel_size']
        p['program']['uPressure'].value = 0
        p['program']['uVelocity'].value = 1
        self.pressure['read']['texture'].use(0)
        self.velocity['read']['texture'].use(1)
        self._draw(p, self.velocity['write'])
        self.velocity['read'], self.velocity['write'] = self.velocity['write'], self.velocity['read']

        # advect velocity
        p = self.programs['advection']
        p['program']['texelSize'].value = self.velocity['texel_size']
        p['program']['dt'].value = dt
        p['program']['dissipation'].value = CONFIG['VELOCITY_DISSIPATION']
        p['program']['uVelocity'].value = 0
        p['program']['uSource'].value = 0
        self.velocity['read']['texture'].use(0)
        self._draw(p, self.velocity['write'])
        self.velocity['read'], self.velocity['write'] = self.velocity['write'], self.velocity['read']

        # advect dye
        p['program']['uVelocity'].value = 0
        p['program']['uSource'].value = 1
        p['program']['dissipation'].value = CONFIG['DENSITY_DISSIPATION']
        self.velocity['read']['texture'].use(0)
        self.dye['read']['texture'].use(1)
        self._draw(p, self.dye['write'])
        self.dye['read'], self.dye['write'] = self.dye['write'], self.dye['read']

    # ------------------------------------------------------------------
    def _blur(self, target, temp, iterations=1):
        p = self.programs['blur']
        p['program']['uTexture'].value = 0
        for _ in range(iterations):
            p['program']['texelSize'].value = (target['texel_size'][0], 0.0)
            target['texture'].use(0)
            self._draw(p, temp)
            p['program']['texelSize'].value = (0.0, target['texel_size'][1])
            temp['texture'].use(0)
            self._draw(p, target)

    def _apply_bloom(self, source, destination):
        if len(self.bloom_fbos) < 2:
            return
        self.ctx.disable(moderngl.BLEND)
        p = self.programs['bloom_pre']
        knee = CONFIG['BLOOM_THRESHOLD'] * CONFIG['BLOOM_SOFT_KNEE'] + 0.0001
        p['program']['curve'].value = (
            CONFIG['BLOOM_THRESHOLD'] - knee, knee * 2, 0.25 / knee)
        p['program']['threshold'].value = CONFIG['BLOOM_THRESHOLD']
        p['program']['uTexture'].value = 0
        source['texture'].use(0)
        last = destination
        self._draw(p, last)

        p = self.programs['bloom_blur']
        p['program']['uTexture'].value = 0
        for fb in self.bloom_fbos:
            p['program']['texelSize'].value = last['texel_size']
            last['texture'].use(0)
            self._draw(p, fb)
            last = fb

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.bloom_fbos) - 2, -1, -1):
            base = self.bloom_fbos[i]
            p['program']['texelSize'].value = last['texel_size']
            last['texture'].use(0)
            base['fbo'].use()
            self.ctx.viewport = (0, 0, base['width'], base['height'])
            p['vao'].render(moderngl.TRIANGLES, vertices=6)
            last = base

        self.ctx.disable(moderngl.BLEND)
        p = self.programs['bloom_fin']
        p['program']['texelSize'].value = last['texel_size']
        p['program']['intensity'].value = CONFIG['BLOOM_INTENSITY']
        p['program']['uTexture'].value = 0
        last['texture'].use(0)
        self._draw(p, destination)

    def _apply_sunrays(self, source, mask, destination):
        self.ctx.disable(moderngl.BLEND)
        p = self.programs['sun_mask']
        p['program']['uTexture'].value = 0
        source['texture'].use(0)
        self._draw(p, mask)

        p = self.programs['sun']
        p['program']['weight'].value = CONFIG['SUNRAYS_WEIGHT']
        p['program']['uTexture'].value = 0
        mask['texture'].use(0)
        self._draw(p, destination)

    # ------------------------------------------------------------------
    def render_to_screen(self):
        ctx = self.ctx
        if CONFIG['BLOOM']:
            self._apply_bloom(self.dye['read'], self.bloom)
        if CONFIG['SUNRAYS']:
            self._apply_sunrays(self.dye['read'], self.dye['write'], self.sunrays)
            self._blur(self.sunrays, self.sunrays_temp, 1)

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA

        # background
        p = self.programs['color']
        r, g, b = [c / 255.0 for c in CONFIG['BACK_COLOR']]
        p['program']['color'].value = (r, g, b, 1.0)
        self._blit(None)
        p['vao'].render(moderngl.TRIANGLES, vertices=6)

        # display
        p = self.programs['display']
        p['program']['texelSize'].value = (1.0 / self.width, 1.0 / self.height)
        p['program']['uTexture'].value = 0
        self.dye['read']['texture'].use(0)
        if CONFIG['BLOOM']:
            p['program']['uBloom'].value = 1
            p['program']['uDithering'].value = 2
            self.bloom['texture'].use(1)
            self.dither.use(2)
            p['program']['ditherScale'].value = (
                self.width / self.dither.width,
                self.height / self.dither.height)
        if CONFIG['SUNRAYS']:
            p['program']['uSunrays'].value = 3
            self.sunrays['texture'].use(3)
        self._blit(None)
        p['vao'].render(moderngl.TRIANGLES, vertices=6)

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

    def simulate_pointer(self, t, frame):
        """Synthetic pointer motion: a Lissajous curve to exercise the fluid."""
        x = 0.5 + 0.35 * math.sin(t * 1.3)
        y = 0.5 + 0.30 * math.cos(t * 1.7)
        # displacement drives the splat force
        dx = 800.0 * math.cos(t * 1.3)
        dy = -800.0 * math.sin(t * 1.7)
        self.splat(x, y, dx, dy, self.pointer_color)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    np.random.seed(42)
    sim = FluidSimulation(VIDEO_WIDTH, VIDEO_HEIGHT)
    sim.multiple_splats(8)

    writer = imageio.get_writer(OUTPUT_FILE, fps=FPS, codec='libx264',
                                quality=8, macro_block_size=1)

    n_frames = FPS * DURATION_SECONDS
    dt = 1.0 / FPS
    for frame in range(n_frames):
        t = frame * dt
        sim.update_colors(dt)
        sim.simulate_pointer(t, frame)
        sim.step(dt)
        sim.render_to_screen()
        img = sim.read_pixels()
        writer.append_data(img)
        if frame % 30 == 0:
            print(f"frame {frame}/{n_frames}")

    writer.close()
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == '__main__':
    main()