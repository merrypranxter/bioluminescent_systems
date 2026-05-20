// AdvectionDiffusion.js — Fluid Advection-Diffusion Light Field Engine
// ∂φ/∂t + u⃗·∇φ = D·∇²φ + S(x,t)
// Upwind differencing for advection (stability-preserving — never central diff)
// Curl-noise divergence-free velocity + explicit vortex sums
// Gray-Scott v field injected as bioluminescent source term
//
// Four visual configurations:
//   plankton drift   D=0.002  curl-noise, GS source
//   vortex garden    D=0.001  explicit vortices, point sources
//   turbulent bloom  D=0.005  high-frequency noise, dense source
//   ink in water     D=0.008  high diffusion, slow advection
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source: https://github.com/merrypranxter/reposcripter2

/* global THREE */

const ADV_CONFIGS = {
    'plankton drift':  { D: 0.002, velScale: 0.30 },
    'vortex garden':   { D: 0.001, velScale: 0.20 },
    'turbulent bloom': { D: 0.005, velScale: 0.40 },
    'ink in water':    { D: 0.008, velScale: 0.15 },
};

const VERT = `#version 300 es
in vec3 position;
void main() { gl_Position = vec4(position, 1.0); }`;

// ── Advection-Diffusion update shader ─────────────────────────────────────────
const ADV_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uAdv;
uniform sampler2D uGSSource;
uniform vec2  uResolution;
uniform float uD_adv;
uniform float uVelScale;
uniform float uTime;
uniform vec2  uMouse;
uniform vec3  uVortices[8];
uniform int   uNumVortices;
out vec4 fragColor;

const float PI = 3.14159265358979;

float hash(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i),               hash(i + vec2(1.0,0.0)), u.x),
               mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}
vec2 curlNoise(vec2 uv, float t) {
    float eps = 0.001;
    float psy  = fbm((uv + vec2(0.0,  eps)) * 3.0 + t * 0.0005);
    float psym = fbm((uv - vec2(0.0,  eps)) * 3.0 + t * 0.0005);
    float psx  = fbm((uv + vec2(eps,  0.0)) * 3.0 + t * 0.0005);
    float psxm = fbm((uv - vec2(eps,  0.0)) * 3.0 + t * 0.0005);
    return vec2((psy - psym) / (2.0 * eps), -(psx - psxm) / (2.0 * eps));
}

vec2 wrap(vec2 uv) { return mod(uv, 1.0); }

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    float phi   = texture(uAdv, uv).r;
    float phi_l = texture(uAdv, wrap(uv + vec2(-texel.x,  0.0     ))).r;
    float phi_r = texture(uAdv, wrap(uv + vec2( texel.x,  0.0     ))).r;
    float phi_u = texture(uAdv, wrap(uv + vec2( 0.0,      texel.y ))).r;
    float phi_d = texture(uAdv, wrap(uv + vec2( 0.0,     -texel.y ))).r;

    // Curl-noise divergence-free velocity
    vec2 vel = curlNoise(uv, uTime) * uVelScale;

    // Explicit vortex contributions (Biot-Savart, regularised core ε²=0.0001)
    for (int i = 0; i < 8; i++) {
        if (i >= uNumVortices) break;
        vec2  diff  = uv - uVortices[i].xy;
        float dist2 = dot(diff, diff) + 0.0001;
        float Gamma = uVortices[i].z;
        vel += (Gamma / (2.0 * PI)) * vec2(-diff.y, diff.x) / dist2;
    }

    // CFL clamp: upwind stable if |vel| ≤ 1 pixel/step
    float vmag = length(vel);
    if (vmag > 0.95) vel *= 0.95 / vmag;

    // ── Upwind advection ──────────────────────────────────────────────────────
    float dphi_dx = (vel.x >= 0.0) ? (phi - phi_l) : (phi_r - phi);
    float dphi_dy = (vel.y >= 0.0) ? (phi - phi_d) : (phi_u - phi);

    // ── Central-difference Laplacian for diffusion ────────────────────────────
    float lap = phi_l + phi_r + phi_u + phi_d - 4.0 * phi;

    // ── Source: Gray-Scott v field as bioluminescent injection ────────────────
    float S = texture(uGSSource, uv).g * 0.018;

    float phi_next = phi + (-vel.x * dphi_dx - vel.y * dphi_dy + uD_adv * lap + S);
    phi_next = clamp(phi_next, 0.0, 1.0);

    float s_acc = clamp(texture(uAdv, uv).a * 0.98 + S, 0.0, 1.0);
    fragColor = vec4(phi_next, vel.x * 0.5 + 0.5, vel.y * 0.5 + 0.5, s_acc);
}`;

const COPY_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    fragColor = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0);
}`;

// ─────────────────────────────────────────────────────────────────────────────

export class AdvectionDiffusion {
    /**
     * @param {THREE.WebGLRenderer} renderer
     * @param {number} size
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
        this.currentConfig = 'plankton drift';
        this.vortices      = [];   // [{x, y, strength}] max 8

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

    _buildVortexUniforms() {
        // Update pre-allocated array in-place (no allocations on hot path)
        if (!this._vortexArr) {
            this._vortexArr = Array.from({ length: 8 }, () => new THREE.Vector3());
        }
        for (let i = 0; i < 8; i++) {
            const v = this.vortices[i];
            if (v) {
                this._vortexArr[i].set(v.x / this.size, v.y / this.size, v.strength);
            } else {
                this._vortexArr[i].set(0, 0, 0);
            }
        }
        return this._vortexArr;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Initialise with small random emission field.
     * @param {string} config  Named configuration (default 'plankton drift')
     */
    init(config = 'plankton drift') {
        this.readFBO  = this._createFBO();
        this.writeFBO = this._createFBO();

        const N    = this.size;
        const data = new Float32Array(N * N * 4);
        for (let i = 0; i < N * N; i++) {
            data[i * 4 + 0] = Math.random() * 0.08; // sparse initial emission
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

        const params = ADV_CONFIGS[config] || ADV_CONFIGS['plankton drift'];

        this.uniforms = {
            uAdv:          { value: this.readFBO.texture },
            uGSSource:     { value: null },           // set each step() call
            uResolution:   { value: new THREE.Vector2(N, N) },
            uD_adv:        { value: params.D },
            uVelScale:     { value: params.velScale },
            uTime:         { value: 0.0 },
            uMouse:        { value: new THREE.Vector2(0, 0) },
            uVortices:     { value: this._buildVortexUniforms() },
            uNumVortices:  { value: 0 },
        };

        this.material = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: ADV_FRAG,
            uniforms:       this.uniforms,
        });

        const mesh = new THREE.Mesh(this._quad, this.material);
        this.scene.add(mesh);
        this.currentConfig = config;
    }

    /**
     * Run 1 advection-diffusion step.
     * @param {THREE.Texture} gsTexture  Gray-Scott texture (v channel = source term)
     * @param {number}        time       Elapsed time (for curl-noise animation)
     */
    step(gsTexture, time = 0) {
        this.uniforms.uAdv.value      = this.readFBO.texture;
        this.uniforms.uGSSource.value = gsTexture;
        this.uniforms.uTime.value     = time;
        this.uniforms.uVortices.value = this._buildVortexUniforms();
        this.uniforms.uNumVortices.value = Math.min(this.vortices.length, 8);

        this.renderer.setRenderTarget(this.writeFBO);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        const tmp     = this.readFBO;
        this.readFBO  = this.writeFBO;
        this.writeFBO = tmp;
    }

    /** @returns {THREE.Texture} */
    getTexture() {
        return this.readFBO.texture;
    }

    /**
     * @param {string} name  'plankton drift' | 'vortex garden' | 'turbulent bloom' | 'ink in water'
     */
    setConfig(name) {
        const params = ADV_CONFIGS[name];
        if (!params) return;
        this.uniforms.uD_adv.value    = params.D;
        this.uniforms.uVelScale.value = params.velScale;
        this.currentConfig            = name;
    }

    /**
     * Add an explicit vortex (Biot-Savart contribution to velocity field).
     * @param {number} x        Pixel column (0=left)
     * @param {number} y        Pixel row   (0=top)
     * @param {number} strength Circulation Γ (positive = counter-clockwise)
     */
    addVortex(x, y, strength) {
        if (this.vortices.length >= 8) this.vortices.shift(); // ring-buffer
        this.vortices.push({ x, y: this.size - y, strength });
    }

    /** Clear all explicit vortices. */
    clearVortices() {
        this.vortices = [];
    }
}
