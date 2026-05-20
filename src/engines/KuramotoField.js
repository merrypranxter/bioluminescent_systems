// KuramotoField.js — Kuramoto Phase-Coupled Oscillator Field
// Four named states: plankton chaos, bloom nucleation, shockwave sync, global pulse
// Phase θ packed into [0,1] in texture; unpacked to [0,2π] in shader
// Emission: I = cos²(θ/2) — peaks at θ=0, dark at θ=π
// 1 simulation step per render frame
//
// Natural frequencies drawn from Lorentzian distribution:
//   ω = ω₀ + Δ·tan(π·(U − 0.5))   where U ~ Uniform[0,1]
//   packed as omega_packed = (ω − ω₀) / (4·spread) + 0.5
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source: https://github.com/merrypranxter/reposcripter2

/* global THREE */

// ── Named states ──────────────────────────────────────────────────────────────
const KU_STATES = {
    'plankton chaos':   { K: 0.10, spread: 0.80, omega0: 1.0, D: 0.05 },
    'bloom nucleation': { K: 0.80, spread: 0.50, omega0: 1.0, D: 0.10 },
    'shockwave sync':   { K: 1.50, spread: 0.30, omega0: 1.0, D: 0.20 },
    'global pulse':     { K: 3.00, spread: 0.10, omega0: 1.0, D: 0.50 },
};

const VERT = `#version 300 es
in vec3 position;
void main() { gl_Position = vec4(position, 1.0); }`;

// ── Kuramoto update fragment shader ───────────────────────────────────────────
const KU_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uKU;
uniform vec2  uResolution;
uniform float uK_kur;
uniform float uOmega0;
uniform float uSpread;
uniform float uD_kur;
uniform float uTime;
out vec4 fragColor;

const float TWO_PI = 6.28318530717959;
const float DT     = 0.10;
vec2 wrap(vec2 uv) { return mod(uv, 1.0); }

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    vec4  center    = texture(uKU, uv);
    float th_packed = center.r;
    float om_packed = center.g;

    float theta = th_packed * TWO_PI;
    float omega = (om_packed - 0.5) * 4.0 * uSpread + uOmega0;

    // 5×5 neighbourhood order parameter: r·exp(iΨ)
    vec2  sum_sc = vec2(0.0);
    float count  = 0.0;
    for (int di = -2; di <= 2; di++) {
        for (int dj = -2; dj <= 2; dj++) {
            vec2  nuv = wrap(uv + vec2(float(di), float(dj)) * texel);
            float nth = texture(uKU, nuv).r * TWO_PI;
            sum_sc += vec2(cos(nth), sin(nth));
            count  += 1.0;
        }
    }
    vec2  order = sum_sc / count;
    float r     = length(order);
    float Psi   = atan(order.y, order.x);

    // Phase Laplacian (central differences)
    float th_l = texture(uKU, wrap(uv + vec2(-texel.x,  0.0     ))).r * TWO_PI;
    float th_r = texture(uKU, wrap(uv + vec2( texel.x,  0.0     ))).r * TWO_PI;
    float th_u = texture(uKU, wrap(uv + vec2( 0.0,      texel.y ))).r * TWO_PI;
    float th_d = texture(uKU, wrap(uv + vec2( 0.0,     -texel.y ))).r * TWO_PI;
    float lap_th = th_l + th_r + th_u + th_d - 4.0 * theta;

    float dtheta     = omega + uK_kur * r * sin(Psi - theta) + uD_kur * lap_th;
    float theta_next = mod(theta + DT * dtheta, TWO_PI);

    float c        = cos(theta_next * 0.5);
    float emission = c * c;

    fragColor = vec4(theta_next / TWO_PI, om_packed, r, emission);
}`;

const COPY_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    fragColor = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0);
}`;

// ─────────────────────────────────────────────────────────────────────────────

export class KuramotoField {
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
        this.currentState  = 'bloom nucleation';
        this._time         = 0;

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
     * Initialise with random phases and Lorentzian-distributed frequencies.
     * @param {string} state  Named state (default 'bloom nucleation')
     */
    init(state = 'bloom nucleation') {
        this.readFBO  = this._createFBO();
        this.writeFBO = this._createFBO();

        const params = KU_STATES[state] || KU_STATES['bloom nucleation'];
        const { omega0, spread } = params;
        const N    = this.size;
        const data = new Float32Array(N * N * 4);
        const TWO_PI = Math.PI * 2;

        for (let i = 0; i < N * N; i++) {
            // Random initial phase
            const theta       = Math.random() * TWO_PI;
            const thetaPacked = theta / TWO_PI;

            // Lorentzian frequency via inverse CDF:
            //   ω = ω₀ + spread·tan(π·(U − 0.5))
            const U       = Math.random();
            const rawOmega = omega0 + spread * Math.tan(Math.PI * (U - 0.5));
            // Clamp to ±5 spread widths to avoid extreme tail values
            const omegaClamped = Math.max(omega0 - 5 * spread,
                                   Math.min(omega0 + 5 * spread, rawOmega));
            // Pack: omega_packed ∈ [0,1], centre = 0.5 at ω = ω₀
            const omegaPacked = (omegaClamped - omega0) / (4.0 * spread) + 0.5;

            data[i * 4 + 0] = thetaPacked;
            data[i * 4 + 1] = Math.min(1.0, Math.max(0.0, omegaPacked));
            data[i * 4 + 2] = 0.0;
            data[i * 4 + 3] = 0.0;
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
        this.uniforms = {
            uKU:         { value: this.readFBO.texture },
            uResolution: { value: new THREE.Vector2(N, N) },
            uK_kur:      { value: params.K },
            uOmega0:     { value: params.omega0 },
            uSpread:     { value: params.spread },
            uD_kur:      { value: params.D },
            uTime:       { value: 0.0 },
        };

        this.material = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: KU_FRAG,
            uniforms:       this.uniforms,
        });

        const mesh = new THREE.Mesh(this._quad, this.material);
        this.scene.add(mesh);
        this.currentState = state;
    }

    /**
     * Run 1 simulation step per frame.
     * @param {number} time  Elapsed time in seconds (for noise animation)
     */
    step(time = 0) {
        this._time              = time;
        this.uniforms.uTime.value = time;
        this.uniforms.uKU.value   = this.readFBO.texture;

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
     * @param {string} name  One of: 'plankton chaos', 'bloom nucleation',
     *                               'shockwave sync', 'global pulse'
     */
    setState(name) {
        const params = KU_STATES[name];
        if (!params) return;
        this.uniforms.uK_kur.value  = params.K;
        this.uniforms.uOmega0.value = params.omega0;
        this.uniforms.uSpread.value = params.spread;
        this.uniforms.uD_kur.value  = params.D;
        this.currentState           = name;
    }

    /**
     * Adjust coupling strength K by delta (scroll-wheel binding).
     * K is clamped to [0.01, 6.0].
     * @param {number} delta
     */
    adjustK(delta) {
        const newK = Math.max(0.01, Math.min(6.0,
            this.uniforms.uK_kur.value + delta));
        this.uniforms.uK_kur.value = newK;
    }
}
