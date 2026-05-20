// GrayScott.js — Gray-Scott Reaction-Diffusion Engine
// Six named parameter regimes: coral spawn, deep vein, mitosis,
// void dissolution, full spectrum, kelp forest
// 8 simulation steps per render frame, dt=1.0, clamp u,v ∈ [0,1]
// Ping-pong FBO pattern, 512×512 RGBA32F
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source: https://github.com/merrypranxter/reposcripter2

/* global THREE */

// ── Named parameter regimes ───────────────────────────────────────────────────
const GS_REGIMES = {
    'coral spawn':      { Du: 0.16, Dv: 0.08, F: 0.037, k: 0.060 },
    'deep vein':        { Du: 0.16, Dv: 0.08, F: 0.022, k: 0.051 },
    'mitosis':          { Du: 0.16, Dv: 0.08, F: 0.028, k: 0.053 },
    'void dissolution': { Du: 0.19, Dv: 0.05, F: 0.014, k: 0.054 },
    'full spectrum':    { Du: 0.16, Dv: 0.08, F: 0.050, k: 0.065 },
    'kelp forest':      { Du: 0.16, Dv: 0.08, F: 0.029, k: 0.057 },
};

// ── Shared vertex shader for all fullscreen quad passes ───────────────────────
const VERT = `#version 300 es
in vec3 position;
void main() { gl_Position = vec4(position, 1.0); }`;

// ── Gray-Scott update fragment shader (inline) ────────────────────────────────
const GS_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uGS;
uniform vec2  uResolution;
uniform float uF;
uniform float uK;
uniform float uDu;
uniform float uDv;
uniform vec2  uMouse;
uniform int   uSeed;
out vec4 fragColor;

vec2 wrap(vec2 uv) { return mod(uv, 1.0); }

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    vec4 center = texture(uGS, uv);
    float u = center.r;
    float v = center.g;

    float u_l = texture(uGS, wrap(uv + vec2(-texel.x,  0.0     ))).r;
    float u_r = texture(uGS, wrap(uv + vec2( texel.x,  0.0     ))).r;
    float u_u = texture(uGS, wrap(uv + vec2( 0.0,      texel.y ))).r;
    float u_d = texture(uGS, wrap(uv + vec2( 0.0,     -texel.y ))).r;
    float lap_u = u_l + u_r + u_u + u_d - 4.0 * u;

    float v_l = texture(uGS, wrap(uv + vec2(-texel.x,  0.0     ))).g;
    float v_r = texture(uGS, wrap(uv + vec2( texel.x,  0.0     ))).g;
    float v_u = texture(uGS, wrap(uv + vec2( 0.0,      texel.y ))).g;
    float v_d = texture(uGS, wrap(uv + vec2( 0.0,     -texel.y ))).g;
    float lap_v = v_l + v_r + v_u + v_d - 4.0 * v;

    float uvv   = u * v * v;
    float u_next = clamp(u + (uDu * lap_u - uvv + uF * (1.0 - u)), 0.0, 1.0);
    float v_next = clamp(v + (uDv * lap_v + uvv - (uF + uK) * v),  0.0, 1.0);

    if (uSeed == 1) {
        vec2 mouseUV = vec2(uMouse.x, uResolution.y - uMouse.y) / uResolution;
        if (length(uv - mouseUV) < 0.025) { u_next = 0.5; v_next = 0.25; }
    }

    fragColor = vec4(u_next, v_next, center.b, v_next);
}`;

// ── Copy / blit fragment shader (used during initialisation) ──────────────────
const COPY_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    fragColor = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0);
}`;

// ─────────────────────────────────────────────────────────────────────────────

export class GrayScott {
    /**
     * @param {THREE.WebGLRenderer} renderer
     * @param {number} size  Texture side length (default 512)
     */
    constructor(renderer, size = 512) {
        this.renderer      = renderer;
        this.size          = size;
        this.readFBO       = null;
        this.writeFBO      = null;
        this.material      = null;
        this.scene         = new THREE.Scene();
        this.camera        = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        this.uniforms      = null;
        this.currentRegime = 'coral spawn';
        this.mousePos      = new THREE.Vector2(0, 0);
        this.shouldSeed    = false;

        // Reusable PlaneGeometry for the fullscreen quad
        this._quad = new THREE.PlaneGeometry(2, 2);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    _createFBO() {
        return new THREE.WebGLRenderTarget(this.size, this.size, {
            type:        THREE.FloatType,
            format:      THREE.RGBAFormat,
            minFilter:   THREE.LinearFilter,
            magFilter:   THREE.LinearFilter,
            wrapS:       THREE.RepeatWrapping,
            wrapT:       THREE.RepeatWrapping,
            depthBuffer: false,
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
     * Initialise ping-pong FBOs and upload the seed state.
     * @param {string} regime  Named parameter regime (default 'coral spawn')
     */
    init(regime = 'coral spawn') {
        this.readFBO  = this._createFBO();
        this.writeFBO = this._createFBO();

        // Build Float32Array seed: u=1, v=0 everywhere; centre 20×20 patch u=0.5, v=0.25
        const N    = this.size;
        const data = new Float32Array(N * N * 4);
        for (let i = 0; i < N * N; i++) {
            data[i * 4 + 0] = 1.0; // u
            data[i * 4 + 1] = 0.0; // v
            data[i * 4 + 2] = 0.0;
            data[i * 4 + 3] = 0.0;
        }
        const cx = Math.floor(N / 2);
        const cy = Math.floor(N / 2);
        for (let dy = -10; dy < 10; dy++) {
            for (let dx = -10; dx < 10; dx++) {
                const x   = cx + dx;
                const y   = cy + dy;
                const idx = (y * N + x) * 4;
                data[idx + 0] = 0.5;
                data[idx + 1] = 0.25;
            }
        }

        const initTex = new THREE.DataTexture(
            data, N, N, THREE.RGBAFormat, THREE.FloatType
        );
        initTex.needsUpdate = true;

        // Blit DataTexture → readFBO via a copy shader
        const copyMat = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: COPY_FRAG,
            uniforms:       { uTex: { value: initTex } },
        });
        this._renderToFBO(copyMat, this.readFBO);
        copyMat.dispose();
        initTex.dispose();

        // Build update material
        const params = GS_REGIMES[regime] || GS_REGIMES['coral spawn'];
        this.uniforms = {
            uGS:         { value: this.readFBO.texture },
            uResolution: { value: new THREE.Vector2(N, N) },
            uF:          { value: params.F },
            uK:          { value: params.k },
            uDu:         { value: params.Du },
            uDv:         { value: params.Dv },
            uMouse:      { value: new THREE.Vector2(0, 0) },
            uSeed:       { value: 0 },
        };

        this.material = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: GS_FRAG,
            uniforms:       this.uniforms,
        });

        const mesh = new THREE.Mesh(this._quad, this.material);
        this.scene.add(mesh);
        this.currentRegime = regime;
    }

    /**
     * Run 8 simulation update passes (ping-pong).
     */
    step() {
        this.uniforms.uMouse.value.copy(this.mousePos);
        this.uniforms.uSeed.value = this.shouldSeed ? 1 : 0;

        for (let i = 0; i < 8; i++) {
            this.uniforms.uGS.value = this.readFBO.texture;
            this.renderer.setRenderTarget(this.writeFBO);
            this.renderer.render(this.scene, this.camera);
            // Swap ping-pong
            const tmp    = this.readFBO;
            this.readFBO  = this.writeFBO;
            this.writeFBO = tmp;
        }
        this.renderer.setRenderTarget(null);
        this.shouldSeed = false;
    }

    /** @returns {THREE.Texture} Current simulation state texture */
    getTexture() {
        return this.readFBO.texture;
    }

    /**
     * Switch named parameter regime.
     * @param {string} name
     */
    setRegime(name) {
        const params = GS_REGIMES[name];
        if (!params) return;
        this.uniforms.uF.value   = params.F;
        this.uniforms.uK.value   = params.k;
        this.uniforms.uDu.value  = params.Du;
        this.uniforms.uDv.value  = params.Dv;
        this.currentRegime       = name;
    }

    /**
     * Inject a Gray-Scott seed at pixel position (x, y).
     * The seed is applied on the next call to step().
     * @param {number} x  Pixel column (0 = left)
     * @param {number} y  Pixel row   (0 = top)
     */
    disturb(x, y) {
        this.mousePos.set(x, y);
        this.shouldSeed = true;
    }
}
