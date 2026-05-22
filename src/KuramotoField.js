// KuramotoField.js — Kuramoto coupled-oscillator phase synchronization engine
// Part of the bioluminescent_systems GPGPU pipeline
// THREE is a global — no imports

const KURAMOTO_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const KURAMOTO_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform float uK;
uniform float uD;
uniform vec2 uResolution;
uniform float uOrderR;
uniform float uOrderPsi;
uniform float uDt;

#define TWO_PI 6.28318530718

void main() {
    vec2 texel = 1.0 / uResolution;

    // Wrapped neighbor UVs (toroidal boundary)
    vec2 uvLeft  = vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y);
    vec2 uvRight = vec2(mod(vUv.x + texel.x,        1.0), vUv.y);
    vec2 uvUp    = vec2(vUv.x, mod(vUv.y + texel.y,        1.0));
    vec2 uvDown  = vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0));

    vec4 center = texture2D(uTexture, vUv);
    vec4 left   = texture2D(uTexture, uvLeft);
    vec4 right  = texture2D(uTexture, uvRight);
    vec4 up     = texture2D(uTexture, uvUp);
    vec4 down   = texture2D(uTexture, uvDown);

    // Unpack phase and natural frequency
    // r channel: theta packed to [0,1]
    // g channel: omega packed to [0,1] via (omega / (TWO_PI * 2.0)) + 0.5
    float theta = center.r * TWO_PI;
    float omega = (center.g - 0.5) * TWO_PI * 2.0;   // unpack → real omega

    // Neighbor phases (unpacked)
    float thetaLeft  = left.r  * TWO_PI;
    float thetaRight = right.r * TWO_PI;
    float thetaUp    = up.r    * TWO_PI;
    float thetaDown  = down.r  * TWO_PI;

    // Phase-aware spatial Laplacian via sin differences — handles wrap-around correctly
    float lapTheta = sin(thetaLeft  - theta)
                   + sin(thetaRight - theta)
                   + sin(thetaUp    - theta)
                   + sin(thetaDown  - theta);

    // Kuramoto mean-field coupling + spatial diffusion
    float dtheta = omega
                 + uK * uOrderR * sin(uOrderPsi - theta)
                 + uD * lapTheta;

    float thetaNext = mod(theta + uDt * dtheta, TWO_PI);

    // Emission: I(θ) = cos²(θ/2)
    float cosHalf = cos(thetaNext * 0.5);
    float emission = cosHalf * cosHalf;

    // r=packed thetaNext, g=preserved omega packing, b=0, a=emission
    gl_FragColor = vec4(thetaNext / TWO_PI, center.g, 0.0, emission);
}
`;

const KURAMOTO_STATES = {
    'plankton chaos':   { K: 0.10, spread: 0.80, omega0: 1.0, D: 0.05 },
    'bloom nucleation': { K: 0.80, spread: 0.50, omega0: 1.0, D: 0.10 },
    'shockwave sync':   { K: 1.50, spread: 0.30, omega0: 1.0, D: 0.20 },
    'global pulse':     { K: 3.00, spread: 0.10, omega0: 1.0, D: 0.50 },
};

class KuramotoField {
    constructor(renderer, width = 512, height = 512) {
        this.renderer = renderer;
        this.width    = width;
        this.height   = height;

        // Default state: bloom nucleation
        this.params = { ...KURAMOTO_STATES['bloom nucleation'] };

        this.fbo      = [null, null];
        this.current  = 0;
        this.scene    = null;
        this.camera   = null;
        this.mesh     = null;
        this.material = null;

        // CPU-side order parameter estimates (updated every 10 frames)
        this.orderR   = 0.0;
        this.orderPsi = 0.0;
        this._frameCount = 0;

        // Small readback FBO for order-parameter computation (32×32)
        this._readbackFBO  = null;
        this._readbackBuf  = new Float32Array(32 * 32 * 4);

        this.init();
    }

    // ─── FBO helpers ──────────────────────────────────────────────────────────

    _makeFBO(w, h) {
        w = w || this.width;
        h = h || this.height;
        return new THREE.WebGLRenderTarget(w, h, {
            format:        THREE.RGBAFormat,
            type:          THREE.FloatType,
            minFilter:     THREE.NearestFilter,
            magFilter:     THREE.NearestFilter,
            depthBuffer:   false,
            stencilBuffer: false,
        });
    }

    // ─── Initialise ───────────────────────────────────────────────────────────

    init() {
        if (this.fbo[0]) { this.fbo[0].dispose(); this.fbo[1].dispose(); }
        this.fbo[0] = this._makeFBO();
        this.fbo[1] = this._makeFBO();
        this.current = 0;

        if (!this._readbackFBO) {
            this._readbackFBO = this._makeFBO(32, 32);
        }

        this.camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
        this.scene  = new THREE.Scene();

        const geo = new THREE.PlaneGeometry(1, 1);
        this.material = new THREE.ShaderMaterial({
            vertexShader:   KURAMOTO_VERT,
            fragmentShader: KURAMOTO_FRAG,
            uniforms: {
                uTexture:   { value: null },
                uK:         { value: this.params.K      },
                uD:         { value: this.params.D      },
                uResolution:{ value: new THREE.Vector2(this.width, this.height) },
                uOrderR:    { value: 0.0 },
                uOrderPsi:  { value: 0.0 },
                uDt:        { value: 0.05 },
            },
        });
        this.mesh = new THREE.Mesh(geo, this.material);
        this.scene.add(this.mesh);

        this._uploadInitialState();

        // Reset order parameter
        this.orderR      = 0.0;
        this.orderPsi    = 0.0;
        this._frameCount = 0;
    }

    _uploadInitialState() {
        const w  = this.width;
        const h  = this.height;
        const PI = Math.PI;
        const TWO_PI = 2.0 * PI;
        const { omega0, spread } = this.params;
        const data = new Float32Array(w * h * 4);

        for (let i = 0; i < w * h; i++) {
            // theta: random in [0, TWO_PI], packed to [0,1]
            const theta = Math.random() * TWO_PI;

            // omega: Lorentzian(omega0, spread) via tan transform
            //   omega = omega0 + spread * tan(PI * (u1 - 0.5))
            // clamped to [omega0 - 3*spread, omega0 + 3*spread]
            const u1 = Math.random();
            let omega = omega0 + spread * Math.tan(PI * (u1 - 0.5));
            omega = Math.max(omega0 - 3.0 * spread, Math.min(omega0 + 3.0 * spread, omega));

            // Pack omega: (omega / (TWO_PI * 2.0)) + 0.5  →  nominally [0,1]
            const omegaPacked = (omega / (TWO_PI * 2.0)) + 0.5;

            data[i * 4 + 0] = theta / TWO_PI;        // r = packed theta
            data[i * 4 + 1] = omegaPacked;            // g = packed omega
            data[i * 4 + 2] = 0.0;
            data[i * 4 + 3] = 0.0;
        }

        const tex = new THREE.DataTexture(
            data, w, h,
            THREE.RGBAFormat, THREE.FloatType
        );
        tex.needsUpdate = true;
        tex.minFilter   = THREE.NearestFilter;
        tex.magFilter   = THREE.NearestFilter;

        const tempMat   = new THREE.MeshBasicMaterial({ map: tex });
        const tempMesh  = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), tempMat);
        const tempScene = new THREE.Scene();
        tempScene.add(tempMesh);

        this.renderer.setRenderTarget(this.fbo[0]);
        this.renderer.render(tempScene, this.camera);
        this.renderer.setRenderTarget(this.fbo[1]);
        this.renderer.render(tempScene, this.camera);
        this.renderer.setRenderTarget(null);

        tempMat.dispose();
        tempMesh.geometry.dispose();
        tex.dispose();
    }

    // ─── Order parameter readback ─────────────────────────────────────────────

    computeOrderParameter() {
        // Downsample to 32×32, read back, compute r·exp(iΨ) = mean(exp(iθ))
        const cam32 = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
        const sc32  = new THREE.Scene();
        const mat32 = new THREE.MeshBasicMaterial({ map: this.fbo[this.current].texture });
        const msh32 = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat32);
        sc32.add(msh32);

        this.renderer.setRenderTarget(this._readbackFBO);
        this.renderer.render(sc32, cam32);

        // Read 32×32 pixels
        this.renderer.readRenderTargetPixels(
            this._readbackFBO,
            0, 0, 32, 32,
            this._readbackBuf
        );
        this.renderer.setRenderTarget(null);

        mat32.dispose();
        msh32.geometry.dispose();

        // Compute order parameter from sampled theta values
        const TWO_PI = 2.0 * Math.PI;
        let sumCos = 0.0;
        let sumSin = 0.0;
        const N = 32 * 32;

        for (let i = 0; i < N; i++) {
            const thetaPacked = this._readbackBuf[i * 4 + 0];
            const theta = thetaPacked * TWO_PI;
            sumCos += Math.cos(theta);
            sumSin += Math.sin(theta);
        }

        sumCos /= N;
        sumSin /= N;

        this.orderR   = Math.sqrt(sumCos * sumCos + sumSin * sumSin);
        this.orderPsi = Math.atan2(sumSin, sumCos);
    }

    // ─── Simulation step ──────────────────────────────────────────────────────

    step(dt = 0.05) {
        this._frameCount++;

        // Update order parameter estimate every 10 frames
        if (this._frameCount % 10 === 0) {
            this.computeOrderParameter();
        }

        const front = this.current;
        const back  = 1 - this.current;

        const u = this.material.uniforms;
        u.uTexture.value  = this.fbo[front].texture;
        u.uK.value        = this.params.K;
        u.uD.value        = this.params.D;
        u.uOrderR.value   = this.orderR;
        u.uOrderPsi.value = this.orderPsi;
        u.uDt.value       = dt;

        this.renderer.setRenderTarget(this.fbo[back]);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        this.current = back;
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    setState(name) {
        const s = KURAMOTO_STATES[name];
        if (!s) {
            console.warn(`KuramotoField: unknown state "${name}"`);
            return;
        }
        this.params = { ...s };
        // Re-initialize omega distribution when state changes (spread has changed)
        this._uploadInitialState();
        this.orderR      = 0.0;
        this.orderPsi    = 0.0;
        this._frameCount = 0;
    }

    setParams(params) {
        Object.assign(this.params, params);
    }

    getTexture() {
        return this.fbo[this.current].texture;
    }

    reset() {
        this._uploadInitialState();
        this.current     = 0;
        this.orderR      = 0.0;
        this.orderPsi    = 0.0;
        this._frameCount = 0;
    }
}

// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2
