// AdvectionDiffusion.js — Curl-noise advection-diffusion emission field engine
// Part of the bioluminescent_systems GPGPU pipeline
// THREE is a global — no imports

const ADV_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ADV_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform sampler2D uGSTexture;
uniform float uD;
uniform vec2 uResolution;
uniform float uTime;
uniform vec2 uVortexPos;
uniform float uVortexActive;
uniform float uDt;

// ---- Hash-based value noise ----
float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i),                hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

// Stream function ψ(uv, t) — layered fbm for divergence-free curl-noise
float psi(vec2 uv, float t) {
    float A1 = 1.0;  float f1 = 0.003 * 512.0; float s1 = 0.0005;
    float A2 = 0.3;  float f2 = 0.008 * 512.0; float s2 = 0.0012;
    return A1 * noise(uv * f1 + t * s1) + A2 * noise(uv * f2 + t * s2);
}

void main() {
    vec2 texel = 1.0 / uResolution;
    float eps = 1.0 / 512.0;

    // ---- Curl-noise velocity from stream function: vel = (∂ψ/∂y, -∂ψ/∂x) ----
    float psi_up    = psi(vUv + vec2(0.0,  eps), uTime);
    float psi_down  = psi(vUv - vec2(0.0,  eps), uTime);
    float psi_right = psi(vUv + vec2(eps,  0.0), uTime);
    float psi_left  = psi(vUv - vec2(eps,  0.0), uTime);

    vec2 vel;
    vel.x =  (psi_up   - psi_down)  / (2.0 * eps);
    vel.y = -(psi_right - psi_left) / (2.0 * eps);

    // ---- Optional explicit vortex (mouse drag) ----
    if (uVortexActive > 0.5) {
        float Gamma = 0.5;
        vec2 d = vUv - uVortexPos;
        // Biot-Savart rotational velocity: perpendicular to d, decaying with |d|²+ε
        vec2 u_vortex = (Gamma / (2.0 * 3.14159265358979)) *
                        vec2(-d.y, d.x) / (dot(d, d) + 0.001);
        vel += u_vortex;
    }

    // ---- Current phi and its neighbors ----
    float phi       = texture2D(uTexture, vUv).r;
    float phi_left  = texture2D(uTexture, vUv - vec2(texel.x, 0.0)).r;
    float phi_right = texture2D(uTexture, vUv + vec2(texel.x, 0.0)).r;
    float phi_down  = texture2D(uTexture, vUv - vec2(0.0, texel.y)).r;
    float phi_up    = texture2D(uTexture, vUv + vec2(0.0, texel.y)).r;

    // ---- Upwind advection (first-order) — avoids unphysical oscillations ----
    float dphiDx, dphiDy;

    if (vel.x > 0.0) {
        dphiDx = (phi - phi_left) * uResolution.x;
    } else {
        dphiDx = (phi_right - phi) * uResolution.x;
    }

    if (vel.y > 0.0) {
        dphiDy = (phi - phi_down) * uResolution.y;
    } else {
        dphiDy = (phi_up - phi) * uResolution.y;
    }

    // ---- Diffusion: central Laplacian ----
    float lapPhi = (phi_left + phi_right + phi_down + phi_up) - 4.0 * phi;

    // ---- Source term: GS v-field × 0.1 ----
    float S = texture2D(uGSTexture, vUv).g * 0.1;

    // ---- Forward Euler update ----
    float phiNext = phi + uDt * (S - vel.x * dphiDx - vel.y * dphiDy + uD * lapPhi);
    phiNext = clamp(phiNext, 0.0, 1.0);

    // r=phi, g=u_vel packed [0,1], b=v_vel packed [0,1], a=emission
    gl_FragColor = vec4(phiNext, vel.x * 0.5 + 0.5, vel.y * 0.5 + 0.5, phiNext);
}
`;

const ADV_CONFIGS = {
    'plankton drift': { D: 0.002 },
    'vortex garden':  { D: 0.001 },
};

class AdvectionDiffusion {
    constructor(renderer, width = 512, height = 512) {
        this.renderer = renderer;
        this.width    = width;
        this.height   = height;

        // Default: plankton drift
        this.params = { ...ADV_CONFIGS['plankton drift'] };

        this.fbo      = [null, null];
        this.current  = 0;
        this.scene    = null;
        this.camera   = null;
        this.mesh     = null;
        this.material = null;

        this.init();
    }

    // ─── FBO helpers ──────────────────────────────────────────────────────────

    _makeFBO() {
        return new THREE.WebGLRenderTarget(this.width, this.height, {
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

        this.camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
        this.scene  = new THREE.Scene();

        const geo = new THREE.PlaneGeometry(1, 1);
        this.material = new THREE.ShaderMaterial({
            vertexShader:   ADV_VERT,
            fragmentShader: ADV_FRAG,
            uniforms: {
                uTexture:      { value: null },
                uGSTexture:    { value: null },
                uD:            { value: this.params.D },
                uResolution:   { value: new THREE.Vector2(this.width, this.height) },
                uTime:         { value: 0.0 },
                uVortexPos:    { value: new THREE.Vector2(0.5, 0.5) },
                uVortexActive: { value: 0.0 },
                uDt:           { value: 1.0 },
            },
        });
        this.mesh = new THREE.Mesh(geo, this.material);
        this.scene.add(this.mesh);

        this._clearFBOs();
    }

    _clearFBOs() {
        // phi = 0.0 everywhere
        const w    = this.width;
        const h    = this.height;
        const data = new Float32Array(w * h * 4); // all zeros

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

    // ─── Simulation step ──────────────────────────────────────────────────────

    step({ gsTexture = null, mousePos = null, mouseActive = false, mouseDragging = false, time = 0.0 } = {}) {
        const front = this.current;
        const back  = 1 - this.current;

        const u = this.material.uniforms;
        u.uTexture.value   = this.fbo[front].texture;
        u.uGSTexture.value = gsTexture;
        u.uD.value         = this.params.D;
        u.uTime.value      = time;
        u.uDt.value        = 1.0;

        // Explicit vortex when mouse is dragging
        if (mouseDragging && mousePos) {
            u.uVortexActive.value = 1.0;
            u.uVortexPos.value.set(mousePos.x, mousePos.y);
        } else {
            u.uVortexActive.value = 0.0;
        }

        this.renderer.setRenderTarget(this.fbo[back]);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        this.current = back;
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    setConfig(name) {
        const c = ADV_CONFIGS[name];
        if (!c) {
            console.warn(`AdvectionDiffusion: unknown config "${name}"`);
            return;
        }
        this.params = { ...c };
    }

    setParams(params) {
        Object.assign(this.params, params);
    }

    getTexture() {
        return this.fbo[this.current].texture;
    }

    reset() {
        this._clearFBOs();
        this.current = 0;
    }
}

// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2
