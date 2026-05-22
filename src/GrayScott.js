// GrayScott.js — Gray-Scott reaction-diffusion engine
// Part of the bioluminescent_systems GPGPU pipeline
// THREE is a global — no imports

const GS_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const GS_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform float uDu;
uniform float uDv;
uniform float uF;
uniform float uK;
uniform vec2 uResolution;
uniform vec2 uMousePos;
uniform float uMouseActive;

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

  float u = center.r;
  float v = center.g;

  // 5-point Laplacian
  float lapU = (left.r + right.r + up.r + down.r) - 4.0 * u;
  float lapV = (left.g + right.g + up.g + down.g) - 4.0 * v;

  float dt  = 1.0;
  float uvv = u * v * v;

  // Gray-Scott equations
  float uNext = u + dt * (uDu * lapU - uvv + uF * (1.0 - u));
  float vNext = v + dt * (uDv * lapV + uvv - (uF + uK) * v);

  // Clamp to valid range — critical: negative u corrupts autocatalytic term
  uNext = clamp(uNext, 0.0, 1.0);
  vNext = clamp(vNext, 0.0, 1.0);

  // Mouse injection: seed v at cursor position
  if (uMouseActive > 0.5) {
    float d = distance(vUv, uMousePos);
    if (d < 0.02) {
      vNext = min(vNext + 0.5, 1.0);
      uNext = max(uNext - 0.3, 0.0);
    }
  }

  // r=u, g=v, b=unused, a=emission (v field)
  gl_FragColor = vec4(uNext, vNext, 0.0, vNext);
}
`;

const GS_REGIMES = {
  'coral spawn':      { Du: 0.16, Dv: 0.08, F: 0.037, k: 0.060 },
  'deep vein':        { Du: 0.16, Dv: 0.08, F: 0.022, k: 0.051 },
  'mitosis':          { Du: 0.16, Dv: 0.08, F: 0.028, k: 0.053 },
  'void dissolution': { Du: 0.19, Dv: 0.05, F: 0.014, k: 0.054 },
  'full spectrum':    { Du: 0.16, Dv: 0.08, F: 0.050, k: 0.065 },
  'kelp forest':      { Du: 0.16, Dv: 0.08, F: 0.029, k: 0.057 },
};

class GrayScott {
  constructor(renderer, width = 512, height = 512) {
    this.renderer = renderer;
    this.width    = width;
    this.height   = height;

    // Default regime: coral spawn
    this.params = { ...GS_REGIMES['coral spawn'] };

    this.fbo     = [null, null];
    this.current = 0;          // fbo[current] = front (readable)
    this.scene   = null;
    this.camera  = null;
    this.mesh    = null;
    this.material = null;

    this.init();
  }

  // ─── FBO helpers ──────────────────────────────────────────────────────────

  _makeFBO() {
    return new THREE.WebGLRenderTarget(this.width, this.height, {
      format:      THREE.RGBAFormat,
      type:        THREE.FloatType,
      minFilter:   THREE.NearestFilter,
      magFilter:   THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
  }

  // ─── Initialise ───────────────────────────────────────────────────────────

  init() {
    // Ping-pong render targets
    if (this.fbo[0]) { this.fbo[0].dispose(); this.fbo[1].dispose(); }
    this.fbo[0] = this._makeFBO();
    this.fbo[1] = this._makeFBO();
    this.current = 0;

    // Orthographic camera + fullscreen quad
    this.camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
    this.scene  = new THREE.Scene();

    const geo = new THREE.PlaneGeometry(1, 1);
    this.material = new THREE.ShaderMaterial({
      vertexShader:   GS_VERT,
      fragmentShader: GS_FRAG,
      uniforms: {
        uTexture:    { value: null },
        uDu:         { value: this.params.Du },
        uDv:         { value: this.params.Dv },
        uF:          { value: this.params.F  },
        uK:          { value: this.params.k  },
        uResolution: { value: new THREE.Vector2(this.width, this.height) },
        uMousePos:   { value: new THREE.Vector2(0.5, 0.5) },
        uMouseActive:{ value: 0.0 },
      },
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.scene.add(this.mesh);

    // Upload initial state via DataTexture
    this._uploadInitialState();
  }

  _uploadInitialState() {
    const w = this.width;
    const h = this.height;
    const data = new Float32Array(w * h * 4);

    // u=1.0 everywhere, v=0.0 everywhere
    for (let i = 0; i < w * h; i++) {
      data[i * 4 + 0] = 1.0;  // u
      data[i * 4 + 1] = 0.0;  // v
      data[i * 4 + 2] = 0.0;
      data[i * 4 + 3] = 0.0;
    }

    // Seed: 20×20 center region — u=0.5, v=0.25
    const cx = Math.floor(w / 2);
    const cy = Math.floor(h / 2);
    const halfSeed = 10;
    for (let dy = -halfSeed; dy < halfSeed; dy++) {
      for (let dx = -halfSeed; dx < halfSeed; dx++) {
        const px = cx + dx;
        const py = cy + dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;
        const idx = (py * w + px) * 4;
        data[idx + 0] = 0.5;
        data[idx + 1] = 0.25;
      }
    }

    // Random scatter: ~200 small 3×3 seed spots
    const rng = () => Math.random();
    for (let s = 0; s < 200; s++) {
      const sx = Math.floor(rng() * (w - 4)) + 2;
      const sy = Math.floor(rng() * (h - 4)) + 2;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const idx = ((sy + dy) * w + (sx + dx)) * 4;
          data[idx + 0] = 0.5 + rng() * 0.1;
          data[idx + 1] = 0.25 + rng() * 0.1;
        }
      }
    }

    const tex = new THREE.DataTexture(
      data, w, h,
      THREE.RGBAFormat, THREE.FloatType
    );
    tex.needsUpdate = true;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;

    // Blit into both FBOs so whichever is back gets the correct seed
    const tempMat = new THREE.MeshBasicMaterial({ map: tex });
    const tempMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), tempMat);
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

  step({ mousePos = null, mouseActive = false } = {}) {
    const front = this.current;
    const back  = 1 - this.current;

    const u = this.material.uniforms;
    u.uTexture.value     = this.fbo[front].texture;
    u.uDu.value          = this.params.Du;
    u.uDv.value          = this.params.Dv;
    u.uF.value           = this.params.F;
    u.uK.value           = this.params.k;
    u.uMouseActive.value = mouseActive ? 1.0 : 0.0;
    if (mousePos) {
      u.uMousePos.value.set(mousePos.x, mousePos.y);
    }

    this.renderer.setRenderTarget(this.fbo[back]);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);

    // Swap front ↔ back
    this.current = back;
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  setRegime(name) {
    const r = GS_REGIMES[name];
    if (!r) {
      console.warn(`GrayScott: unknown regime "${name}"`);
      return;
    }
    this.params = { ...r };
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
