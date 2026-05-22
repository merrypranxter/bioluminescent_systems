// FitzhughNagumo.js — Fitzhugh-Nagumo excitable media engine
// Part of the bioluminescent_systems GPGPU pipeline
// THREE is a global — no imports

const FN_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FN_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform float uD;
uniform float uAlpha;
uniform float uEpsilon;
uniform float uBeta;
uniform float uGamma;
uniform float uDelta;
uniform vec2 uResolution;
uniform vec2 uMousePos;
uniform float uMouseActive;
uniform float uDt;

void main() {
  vec2 texel = 1.0 / uResolution;

  // Wrapped neighbor sampling (toroidal boundary)
  vec2 uvLeft  = vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y);
  vec2 uvRight = vec2(mod(vUv.x + texel.x,        1.0), vUv.y);
  vec2 uvUp    = vec2(vUv.x, mod(vUv.y + texel.y,        1.0));
  vec2 uvDown  = vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0));

  vec4 center = texture2D(uTexture, vUv);
  vec4 left   = texture2D(uTexture, uvLeft);
  vec4 right  = texture2D(uTexture, uvRight);
  vec4 up     = texture2D(uTexture, uvUp);
  vec4 down   = texture2D(uTexture, uvDown);

  float v = center.r;
  float w = center.g;

  // 5-point Laplacian of v (membrane potential only)
  float lapV = (left.r + right.r + up.r + down.r) - 4.0 * v;

  // Cubic nonlinearity: v*(v - alpha)*(1 - v)
  float cubic = v * (v - uAlpha) * (1.0 - v);

  // External stimulus from mouse
  float iExt = 0.0;
  if (uMouseActive > 0.5) {
    float d = distance(vUv, uMousePos);
    if (d < 0.03) {
      iExt = 0.5;
    }
  }

  // Fitzhugh-Nagumo update — dt MUST be ≤ 0.01 (stiff cubic term)
  float vNext = v + uDt * (uD * lapV + cubic - w + iExt);
  float wNext = w + uDt * (uEpsilon * (uBeta * v - uGamma * w - uDelta));

  // Age counter in blue channel — increment each step, normalize by 1000
  float age = center.b + (1.0 / 1000.0);

  // Emission: positive v only (negative = refractory darkness)
  float emission = max(vNext, 0.0);

  // r=v(potential), g=w(recovery), b=age, a=emission
  gl_FragColor = vec4(vNext, wNext, age, emission);
}
`;

const FN_MODES = {
  'ring storm':    { D: 0.30, eps: 0.020, alpha: 0.10, beta: 0.5, gamma: 1.0, delta: 0.0 },
  'spiral lock':   { D: 0.50, eps: 0.010, alpha: 0.20, beta: 0.5, gamma: 1.0, delta: 0.0 },
  'pulse cascade': { D: 0.10, eps: 0.040, alpha: 0.15, beta: 0.5, gamma: 1.0, delta: 0.0 },
  'slow bloom':    { D: 0.80, eps: 0.005, alpha: 0.10, beta: 0.5, gamma: 1.0, delta: 0.0 },
};

class FitzhughNagumo {
  constructor(renderer, width = 512, height = 512) {
    this.renderer = renderer;
    this.width    = width;
    this.height   = height;

    // Default mode: spiral lock
    this.params = { ...FN_MODES['spiral lock'] };

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
      vertexShader:   FN_VERT,
      fragmentShader: FN_FRAG,
      uniforms: {
        uTexture:    { value: null },
        uD:          { value: this.params.D     },
        uAlpha:      { value: this.params.alpha  },
        uEpsilon:    { value: this.params.eps    },
        uBeta:       { value: this.params.beta   },
        uGamma:      { value: this.params.gamma  },
        uDelta:      { value: this.params.delta  },
        uResolution: { value: new THREE.Vector2(this.width, this.height) },
        uMousePos:   { value: new THREE.Vector2(0.5, 0.5) },
        uMouseActive:{ value: 0.0 },
        uDt:         { value: 0.01 },
      },
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.scene.add(this.mesh);

    this._uploadInitialState();
  }

  _uploadInitialState() {
    const w = this.width;
    const h = this.height;
    const data = new Float32Array(w * h * 4);

    // Small random noise ±0.01 around 0 for both v and w
    for (let i = 0; i < w * h; i++) {
      data[i * 4 + 0] = (Math.random() - 0.5) * 0.02;  // v
      data[i * 4 + 1] = (Math.random() - 0.5) * 0.02;  // w
      data[i * 4 + 2] = 0.0;                             // age
      data[i * 4 + 3] = 0.0;                             // emission
    }

    // Seed two quarter-ring arcs of excited cells (v≈1.0) to guarantee spiral formation.
    // Arc 1: upper-right quadrant centred at (cx-40, cy+40)
    // Arc 2: lower-left  quadrant centred at (cx+40, cy-40)
    const cx = Math.floor(w / 2);
    const cy = Math.floor(h / 2);
    const R  = 30;  // ring radius in pixels

    const seedArc = (acx, acy, startAngle, endAngle) => {
      const steps = 300;
      for (let s = 0; s <= steps; s++) {
        const angle = startAngle + (endAngle - startAngle) * (s / steps);
        // Ring thickness: 3 pixels
        for (let r = R - 2; r <= R + 2; r++) {
          const px = Math.round(acx + r * Math.cos(angle));
          const py = Math.round(acy + r * Math.sin(angle));
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          const idx = (py * w + px) * 4;
          data[idx + 0] = 1.0;  // v = excited
          data[idx + 1] = 0.0;  // w = not recovered yet
        }
      }
    };

    const PI = Math.PI;
    // Arc 1: top-right, upper half (0 → PI)
    seedArc(cx - 40, cy + 40, 0, PI);
    // Arc 2: bottom-left, lower half (PI → 2*PI)
    seedArc(cx + 40, cy - 40, PI, 2 * PI);

    const tex = new THREE.DataTexture(
      data, w, h,
      THREE.RGBAFormat, THREE.FloatType
    );
    tex.needsUpdate  = true;
    tex.minFilter    = THREE.NearestFilter;
    tex.magFilter    = THREE.NearestFilter;

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

  step({ mousePos = null, mouseActive = false, dt = 0.01 } = {}) {
    // Clamp dt — the cubic term is stiff; larger steps diverge to NaN
    const safeDt = Math.min(dt, 0.01);

    const front = this.current;
    const back  = 1 - this.current;

    const u = this.material.uniforms;
    u.uTexture.value     = this.fbo[front].texture;
    u.uD.value           = this.params.D;
    u.uAlpha.value       = this.params.alpha;
    u.uEpsilon.value     = this.params.eps;
    u.uBeta.value        = this.params.beta;
    u.uGamma.value       = this.params.gamma;
    u.uDelta.value       = this.params.delta;
    u.uDt.value          = safeDt;
    u.uMouseActive.value = mouseActive ? 1.0 : 0.0;
    if (mousePos) {
      u.uMousePos.value.set(mousePos.x, mousePos.y);
    }

    this.renderer.setRenderTarget(this.fbo[back]);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);

    this.current = back;
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  setMode(name) {
    const m = FN_MODES[name];
    if (!m) {
      console.warn(`FitzhughNagumo: unknown mode "${name}"`);
      return;
    }
    this.params = { ...m };
  }

  setParams(params) {
    Object.assign(this.params, params);
  }

  getTexture() {
    return this.fbo[this.current].texture;
  }

  reset() {
    this._uploadInitialState();
    this.current = 0;
  }
}

// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2
