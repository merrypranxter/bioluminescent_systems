// FitzhughNagumo.js — FitzHugh-Nagumo Excitable Media Engine
// Four named modes: ring storm, spiral lock, pulse cascade, slow bloom
// 50 simulation steps per render frame at dt=0.01 (stiff system)
// Default init: broken ring at radius 100 to seed spiral lock
// Texture: .r=v (potential), .g=w (recovery), .b=age, .a=emission
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source: https://github.com/merrypranxter/reposcripter2

/* global THREE */

// ── Named parameter modes ─────────────────────────────────────────────────────
const FN_MODES = {
    'ring storm':    { D: 0.30, eps: 0.020, alpha: 0.10, beta: 0.5, gamma: 1.0 },
    'spiral lock':   { D: 0.50, eps: 0.010, alpha: 0.20, beta: 0.5, gamma: 1.0 },
    'pulse cascade': { D: 0.10, eps: 0.040, alpha: 0.15, beta: 0.5, gamma: 1.0 },
    'slow bloom':    { D: 0.80, eps: 0.005, alpha: 0.10, beta: 0.5, gamma: 1.0 },
};

const VERT = `#version 300 es
in vec3 position;
void main() { gl_Position = vec4(position, 1.0); }`;

// ── FitzHugh-Nagumo update fragment shader ────────────────────────────────────
const FN_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uFN;
uniform vec2  uResolution;
uniform float uD_fn;
uniform float uEps;
uniform float uAlpha;
uniform float uBeta;
uniform float uGamma;
uniform vec2  uMouse;
uniform int   uMouseDown;
out vec4 fragColor;

const float DT = 0.01;
vec2 wrap(vec2 uv) { return mod(uv, 1.0); }

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    vec4 center = texture(uFN, uv);
    float v   = center.r;
    float w   = center.g;
    float age = center.b;

    float v_l = texture(uFN, wrap(uv + vec2(-texel.x,  0.0     ))).r;
    float v_r = texture(uFN, wrap(uv + vec2( texel.x,  0.0     ))).r;
    float v_u = texture(uFN, wrap(uv + vec2( 0.0,      texel.y ))).r;
    float v_d = texture(uFN, wrap(uv + vec2( 0.0,     -texel.y ))).r;
    float lap_v = v_l + v_r + v_u + v_d - 4.0 * v;

    float I_ext = 0.0;
    if (uMouseDown == 1) {
        vec2 mouseUV = vec2(uMouse.x, uResolution.y - uMouse.y) / uResolution;
        float dist = length(uv - mouseUV);
        if (dist < 0.04) I_ext = 0.6 * (1.0 - dist / 0.04);
    }

    // Cubic nonlinearity — stiff; dt=0.01 required
    float cubic  = v * (v - uAlpha) * (1.0 - v);
    float v_next = v + DT * (uD_fn * lap_v + cubic - w + I_ext);
    float w_next = w + DT * (uEps * (uBeta * v - uGamma * w));

    float emission = max(v_next, 0.0);
    fragColor = vec4(v_next, w_next, age + DT, emission);
}`;

const COPY_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    fragColor = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0);
}`;

// ─────────────────────────────────────────────────────────────────────────────

export class FitzhughNagumo {
    /**
     * @param {THREE.WebGLRenderer} renderer
     * @param {number} size  Texture side length (default 512)
     */
    constructor(renderer, size = 512) {
        this.renderer    = renderer;
        this.size        = size;
        this.readFBO     = null;
        this.writeFBO    = null;
        this.material    = null;
        this.scene       = new THREE.Scene();
        this.camera      = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        this.uniforms    = null;
        this.currentMode = 'spiral lock';
        this.mousePos    = new THREE.Vector2(0, 0);
        this.mouseDown   = false;

        this._quad = new THREE.PlaneGeometry(2, 2);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    _createFBO() {
        return new THREE.WebGLRenderTarget(this.size, this.size, {
            type:          THREE.FloatType,
            format:        THREE.RGBAFormat,
            minFilter:     THREE.LinearFilter,
            magFilter:     THREE.LinearFilter,
            wrapS:         THREE.RepeatWrapping,
            wrapT:         THREE.RepeatWrapping,
            depthBuffer:   false,
            stencilBuffer: false,
        });
    }

    _renderToFBO(material, target) {
        const mesh  = new THREE.Mesh(this._quad, material);
        const scene = new THREE.Scene();
        scene.add(mesh);
        this.renderer.setRenderTarget(target);
        this.renderer.render(scene, this.camera);
        this.renderer.setRenderTarget(null);
        scene.remove(mesh);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Initialise with a broken ring to seed spiral lock mode.
     * Ring: centre=(256,256), radius=100, lineWidth=3px, bottom half only
     * (zeroing the top half creates a free end that curls into a spiral).
     * @param {string} mode  Named mode (default 'spiral lock')
     */
    init(mode = 'spiral lock') {
        this.readFBO  = this._createFBO();
        this.writeFBO = this._createFBO();

        const N    = this.size;
        const data = new Float32Array(N * N * 4); // initialised to 0

        // Broken ring: full ring → zero top half → free end curls into spiral
        const cx     = N / 2;
        const cy     = N / 2;
        const radius = 100;
        const width  = 3;

        for (let y = 0; y < N; y++) {
            for (let x = 0; x < N; x++) {
                const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
                // Full ring, bottom half only (y > cy means lower half in screen space)
                if (Math.abs(dist - radius) < width && y > cy) {
                    const idx = (y * N + x) * 4;
                    data[idx + 0] = 1.0; // v = excited
                }
            }
        }

        const initTex = new THREE.DataTexture(
            data, N, N, THREE.RGBAFormat, THREE.FloatType
        );
        initTex.needsUpdate = true;

        const copyMat = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: COPY_FRAG,
            uniforms:       { uTex: { value: initTex } },
        });
        this._renderToFBO(copyMat, this.readFBO);
        copyMat.dispose();
        initTex.dispose();

        // Build update material
        const params = FN_MODES[mode] || FN_MODES['spiral lock'];
        this.uniforms = {
            uFN:         { value: this.readFBO.texture },
            uResolution: { value: new THREE.Vector2(N, N) },
            uD_fn:       { value: params.D },
            uEps:        { value: params.eps },
            uAlpha:      { value: params.alpha },
            uBeta:       { value: params.beta },
            uGamma:      { value: params.gamma },
            uMouse:      { value: new THREE.Vector2(0, 0) },
            uMouseDown:  { value: 0 },
        };

        this.material = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: FN_FRAG,
            uniforms:       this.uniforms,
        });

        const mesh = new THREE.Mesh(this._quad, this.material);
        this.scene.add(mesh);
        this.currentMode = mode;
    }

    /**
     * Run 50 update passes at dt=0.01 (stiff cubic nonlinearity — never use dt > 0.02).
     */
    step() {
        this.uniforms.uMouse.value.copy(this.mousePos);
        this.uniforms.uMouseDown.value = this.mouseDown ? 1 : 0;

        for (let i = 0; i < 50; i++) {
            this.uniforms.uFN.value = this.readFBO.texture;
            this.renderer.setRenderTarget(this.writeFBO);
            this.renderer.render(this.scene, this.camera);
            const tmp     = this.readFBO;
            this.readFBO  = this.writeFBO;
            this.writeFBO = tmp;
        }
        this.renderer.setRenderTarget(null);
    }

    /** @returns {THREE.Texture} */
    getTexture() {
        return this.readFBO.texture;
    }

    /**
     * @param {string} name  One of: 'ring storm', 'spiral lock', 'pulse cascade', 'slow bloom'
     */
    setMode(name) {
        const params = FN_MODES[name];
        if (!params) return;
        this.uniforms.uD_fn.value   = params.D;
        this.uniforms.uEps.value    = params.eps;
        this.uniforms.uAlpha.value  = params.alpha;
        this.uniforms.uBeta.value   = params.beta;
        this.uniforms.uGamma.value  = params.gamma;
        this.currentMode            = name;
    }

    /**
     * Apply I_ext stimulus at pixel position (x, y) while mouse is held.
     * @param {number} x
     * @param {number} y
     */
    disturb(x, y) {
        this.mousePos.set(x, y);
        this.mouseDown = true;
    }

    /** Call when mouse button is released. */
    releaseDisturbance() {
        this.mouseDown = false;
    }
}
