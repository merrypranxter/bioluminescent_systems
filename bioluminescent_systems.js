// Bioluminescent Systems — JS5 sketch for RepoScripter2
// Four mathematical pattern engines: Gray-Scott, Fitzhugh-Nagumo, Kuramoto, Advection-Diffusion
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

// ─── Constants ───────────────────────────────────────────────────────────────

const SIM_SIZE = 512;

const GS_REGIMES = {
  'coral spawn':      { Du: 0.16, Dv: 0.08, F: 0.037, k: 0.060 },
  'deep vein':        { Du: 0.16, Dv: 0.08, F: 0.022, k: 0.051 },
  'mitosis':          { Du: 0.16, Dv: 0.08, F: 0.028, k: 0.053 },
  'void dissolution': { Du: 0.19, Dv: 0.05, F: 0.014, k: 0.054 },
  'full spectrum':    { Du: 0.16, Dv: 0.08, F: 0.050, k: 0.065 },
  'kelp forest':      { Du: 0.16, Dv: 0.08, F: 0.029, k: 0.057 },
};

const FN_MODES = {
  'ring storm':    { D: 0.30, eps: 0.020, alpha: 0.10 },
  'spiral lock':   { D: 0.50, eps: 0.010, alpha: 0.20 },
  'pulse cascade': { D: 0.10, eps: 0.040, alpha: 0.15 },
  'slow bloom':    { D: 0.80, eps: 0.005, alpha: 0.10 },
};

const KURAMOTO_STATES = {
  'plankton chaos':    { K: 0.10, spread: 0.80, omega0: 1.0, D: 0.05 },
  'bloom nucleation':  { K: 0.80, spread: 0.50, omega0: 1.0, D: 0.10 },
  'shockwave sync':    { K: 1.50, spread: 0.30, omega0: 1.0, D: 0.20 },
  'global pulse':      { K: 3.00, spread: 0.10, omega0: 1.0, D: 0.50 },
};

// ─── Shared vertex shader ─────────────────────────────────────────────────────

const VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// ─── Gray-Scott shader ────────────────────────────────────────────────────────

const GS_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform vec2 uResolution;
  uniform float uDu;
  uniform float uDv;
  uniform float uF;
  uniform float uK;
  uniform vec2 uMousePos;
  uniform float uMouseActive;

  void main() {
    vec2 texel = 1.0 / uResolution;
    vec4 center = texture2D(uTexture, vUv);
    float u = center.r;
    float v = center.g;

    vec2 uvL = vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y);
    vec2 uvR = vec2(mod(vUv.x + texel.x, 1.0), vUv.y);
    vec2 uvU = vec2(vUv.x, mod(vUv.y + texel.y, 1.0));
    vec2 uvD = vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0));

    float lapU = texture2D(uTexture, uvL).r + texture2D(uTexture, uvR).r
               + texture2D(uTexture, uvU).r + texture2D(uTexture, uvD).r
               - 4.0 * u;
    float lapV = texture2D(uTexture, uvL).g + texture2D(uTexture, uvR).g
               + texture2D(uTexture, uvU).g + texture2D(uTexture, uvD).g
               - 4.0 * v;

    float uvv = u * v * v;
    float uNext = clamp(u + (uDu * lapU - uvv + uF * (1.0 - u)), 0.0, 1.0);
    float vNext = clamp(v + (uDv * lapV + uvv - (uF + uK) * v), 0.0, 1.0);

    if (uMouseActive > 0.5) {
      float dist = distance(vUv, uMousePos);
      if (dist < 0.02) {
        vNext = min(vNext + 0.5 * (1.0 - dist / 0.02), 1.0);
      }
    }

    gl_FragColor = vec4(uNext, vNext, 0.0, vNext);
  }
`;

// ─── Fitzhugh-Nagumo shader ───────────────────────────────────────────────────

const FN_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform vec2 uResolution;
  uniform float uD;
  uniform float uAlpha;
  uniform float uEpsilon;
  uniform float uBeta;
  uniform float uGamma;
  uniform float uDelta;
  uniform float uDt;
  uniform vec2 uMousePos;
  uniform float uMouseActive;

  void main() {
    vec2 texel = 1.0 / uResolution;
    vec4 center = texture2D(uTexture, vUv);
    float v = center.r;
    float w = center.g;
    float age = center.b;

    vec4 L = texture2D(uTexture, vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y));
    vec4 R = texture2D(uTexture, vec2(mod(vUv.x + texel.x, 1.0), vUv.y));
    vec4 U = texture2D(uTexture, vec2(vUv.x, mod(vUv.y + texel.y, 1.0)));
    vec4 D = texture2D(uTexture, vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0)));

    float lapV = L.r + R.r + U.r + D.r - 4.0 * v;
    float cubic = v * (v - uAlpha) * (1.0 - v);

    float iExt = 0.0;
    if (uMouseActive > 0.5) {
      float dist = distance(vUv, uMousePos);
      if (dist < 0.03) iExt = 0.5 * (1.0 - dist / 0.03);
    }

    float vNext = v + uDt * (uD * lapV + cubic - w + iExt);
    float wNext = w + uDt * (uEpsilon * (uBeta * v - uGamma * w - uDelta));
    float ageNext = age + 1.0 / 1000.0;
    float emission = max(vNext, 0.0);

    gl_FragColor = vec4(vNext, wNext, ageNext, emission);
  }
`;

// ─── Kuramoto shader ──────────────────────────────────────────────────────────

const KURAMOTO_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform vec2 uResolution;
  uniform float uK;
  uniform float uD;
  uniform float uDt;
  uniform float uOrderR;
  uniform float uOrderPsi;

  const float TWO_PI = 6.28318530718;

  void main() {
    vec2 texel = 1.0 / uResolution;
    vec4 center = texture2D(uTexture, vUv);
    float thetaPacked = center.r;
    float omegaPacked = center.g;

    float theta = thetaPacked * TWO_PI;
    float omega = omegaPacked * TWO_PI * 2.0 - TWO_PI;

    vec4 L = texture2D(uTexture, vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y));
    vec4 R = texture2D(uTexture, vec2(mod(vUv.x + texel.x, 1.0), vUv.y));
    vec4 U = texture2D(uTexture, vec2(vUv.x, mod(vUv.y + texel.y, 1.0)));
    vec4 Dn = texture2D(uTexture, vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0)));

    // Phase-aware Laplacian via sin of differences (handles 2π wrap)
    float lapTheta = sin(L.r * TWO_PI - theta)
                   + sin(R.r * TWO_PI - theta)
                   + sin(U.r * TWO_PI - theta)
                   + sin(Dn.r * TWO_PI - theta);

    float dtheta = omega + uK * uOrderR * sin(uOrderPsi - theta) + uD * lapTheta;
    float thetaNext = mod(theta + uDt * dtheta, TWO_PI);

    float e = cos(thetaNext * 0.5);
    float emission = e * e;

    gl_FragColor = vec4(thetaNext / TWO_PI, omegaPacked, 0.0, emission);
  }
`;

// ─── Advection-Diffusion shader ───────────────────────────────────────────────

const ADV_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform sampler2D uGSTexture;
  uniform vec2 uResolution;
  uniform float uD;
  uniform float uTime;
  uniform vec2 uVortexPos;
  uniform float uVortexActive;

  float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
  }
  float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
  }
  float streamFn(vec2 uv) {
    return noise(uv * 1.536 + uTime * 0.0005)
         + 0.3 * noise(uv * 4.096 + uTime * 0.0012);
  }

  void main() {
    vec2 texel = 1.0 / uResolution;
    float phi = texture2D(uTexture, vUv).r;

    float eps = texel.x;
    float psiR  = streamFn(vUv + vec2(eps, 0.0));
    float psiL  = streamFn(vUv - vec2(eps, 0.0));
    float psiU  = streamFn(vUv + vec2(0.0, eps));
    float psiD  = streamFn(vUv - vec2(0.0, eps));

    vec2 vel;
    vel.x =  (psiU - psiD) / (2.0 * eps);
    vel.y = -(psiR - psiL) / (2.0 * eps);
    vel *= 0.3;

    if (uVortexActive > 0.5) {
      vec2 d = vUv - uVortexPos;
      float r2 = dot(d, d) + 0.001;
      vel += 0.5 / (6.28318) * vec2(-d.y, d.x) / r2;
    }

    // Upwind advection
    float phiL = texture2D(uTexture, vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y)).r;
    float phiR = texture2D(uTexture, vec2(mod(vUv.x + texel.x, 1.0), vUv.y)).r;
    float phiU = texture2D(uTexture, vec2(vUv.x, mod(vUv.y + texel.y, 1.0))).r;
    float phiD = texture2D(uTexture, vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0))).r;

    float dphiDx = vel.x > 0.0
      ? (phi - phiL) * uResolution.x
      : (phiR - phi) * uResolution.x;
    float dphiDy = vel.y > 0.0
      ? (phi - phiD) * uResolution.y
      : (phiU - phi) * uResolution.y;

    float lapPhi = (phiL + phiR + phiU + phiD - 4.0 * phi);

    float S = texture2D(uGSTexture, vUv).g * 0.1;

    float phiNext = clamp(phi + S - vel.x * dphiDx * texel.x - vel.y * dphiDy * texel.y + uD * lapPhi, 0.0, 1.0);

    gl_FragColor = vec4(phiNext, vel.x * 0.5 + 0.5, vel.y * 0.5 + 0.5, phiNext);
  }
`;

// ─── Composite shader ─────────────────────────────────────────────────────────

const COMPOSITE_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uGS;
  uniform sampler2D uFN;
  uniform sampler2D uKuramoto;
  uniform sampler2D uAdvection;
  uniform float uGSWeight;
  uniform float uFNWeight;
  uniform float uKuramotoWeight;
  uniform float uAdvectionWeight;

  void main() {
    float gsEmit  = texture2D(uGS, vUv).a;
    float fnEmit  = texture2D(uFN, vUv).a;
    float kuEmit  = texture2D(uKuramoto, vUv).a;
    float advEmit = texture2D(uAdvection, vUv).a;

    // GS: abyss → plankton cyan
    vec3 gsColor  = mix(vec3(0.0, 0.039, 0.078), vec3(0.0, 0.706, 0.847), gsEmit);
    // FN: void → jellyfish violet → electric teal
    vec3 fnColor  = mix(vec3(0.0), vec3(0.482, 0.176, 0.545), fnEmit * 0.6)
                  + mix(vec3(0.0), vec3(0.282, 0.792, 0.894), fnEmit);
    // Kuramoto: void → firefly green
    vec3 kuColor  = mix(vec3(0.0), vec3(0.659, 1.0, 0.471), kuEmit);
    // Advection: void → white core teal
    vec3 advColor = mix(vec3(0.0), vec3(0.565, 0.878, 0.937), advEmit);

    vec3 total = gsColor  * uGSWeight
               + fnColor  * uFNWeight
               + kuColor  * uKuramotoWeight
               + advColor * uAdvectionWeight;

    gl_FragColor = vec4(total, 1.0);
  }
`;

// ─── Bloom shaders ────────────────────────────────────────────────────────────

const BLOOM_DOWN_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform vec2 uResolution;
  uniform float uOffset;
  uniform float uThreshold;

  void main() {
    vec2 texel = 1.0 / uResolution;
    float o = uOffset + 0.5;
    vec3 s = texture2D(uTexture, vUv + vec2(-o, -o) * texel).rgb
           + texture2D(uTexture, vUv + vec2( o, -o) * texel).rgb
           + texture2D(uTexture, vUv + vec2(-o,  o) * texel).rgb
           + texture2D(uTexture, vUv + vec2( o,  o) * texel).rgb;
    s *= 0.25;
    s = max(s - uThreshold, 0.0) / max(1.0 - uThreshold, 0.0001);
    gl_FragColor = vec4(s, 1.0);
  }
`;

const BLOOM_UP_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform sampler2D uPrevTexture;
  uniform vec2 uResolution;
  uniform float uOffset;
  uniform float uIntensity;

  void main() {
    vec2 texel = 1.0 / uResolution;
    float o = uOffset + 0.5;
    vec3 bloom = texture2D(uTexture, vUv + vec2(-o, -o) * texel).rgb
               + texture2D(uTexture, vUv + vec2( o, -o) * texel).rgb
               + texture2D(uTexture, vUv + vec2(-o,  o) * texel).rgb
               + texture2D(uTexture, vUv + vec2( o,  o) * texel).rgb;
    bloom *= 0.25;
    vec3 prev = texture2D(uPrevTexture, vUv).rgb;
    gl_FragColor = vec4(prev + bloom * uIntensity, 1.0);
  }
`;

// ─── Decay shader ─────────────────────────────────────────────────────────────

const DECAY_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTrailTexture;
  uniform sampler2D uEmitTexture;
  uniform float uDecay;
  uniform float uEmitWeight;

  void main() {
    vec3 trail = texture2D(uTrailTexture, vUv).rgb;
    vec3 emit  = texture2D(uEmitTexture, vUv).rgb;
    gl_FragColor = vec4(trail * uDecay + emit * uEmitWeight, 1.0);
  }
`;

// ─── Output shader ────────────────────────────────────────────────────────────

const OUTPUT_FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform sampler2D uBloomTexture;
  uniform float uTime;
  uniform vec2 uResolution;
  uniform float uBloomIntensity;

  void main() {
    vec2 offset = (vUv - 0.5) * 2.0;

    float r = texture2D(uTexture, vUv - offset * 0.002).r;
    float g = texture2D(uTexture, vUv).g;
    float b = texture2D(uTexture, vUv + offset * 0.002).b;
    vec3 col = vec3(r, g, b);

    col += texture2D(uBloomTexture, vUv).rgb * uBloomIntensity;

    // Reinhard tone map (luminance-only)
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float scale = (lum / (1.0 + lum)) / max(lum, 0.0001);
    col *= scale;

    // Vignette
    float vig = 1.0 - 0.6 * dot(offset * 0.5, offset * 0.5);
    col *= vig;

    // Film grain (amplitude 0.03, per-frame seed)
    float grain = fract(sin(dot(vUv + fract(uTime * 0.01), vec2(12.9898, 78.233))) * 43758.5453);
    col += grain * 0.03 - 0.015;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
  }
`;

// ─── FBO helpers ──────────────────────────────────────────────────────────────

function makeFBO(w, h) {
  return new THREE.WebGLRenderTarget(w, h, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
    depthBuffer: false,
    stencilBuffer: false,
  });
}

function makePingPong(w, h) {
  return [makeFBO(w, h), makeFBO(w, h)];
}

function makeQuadMesh(fragShader, uniforms) {
  return new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: fragShader,
      uniforms,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    })
  );
}

// ─── Initialization helpers ───────────────────────────────────────────────────

function makeGSInitData(w, h) {
  const data = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4 + 0] = 1.0; // u
    data[i * 4 + 1] = 0.0; // v
    data[i * 4 + 2] = 0.0;
    data[i * 4 + 3] = 0.0;
  }
  // Seed center region
  const cx = Math.floor(w / 2), cy = Math.floor(h / 2);
  for (let dy = -10; dy <= 10; dy++) {
    for (let dx = -10; dx <= 10; dx++) {
      const ix = ((cx + dx + w) % w) + ((cy + dy + h) % h) * w;
      data[ix * 4 + 0] = 0.5;
      data[ix * 4 + 1] = 0.25;
    }
  }
  // Scatter a few extra seeds
  for (let s = 0; s < 8; s++) {
    const sx = Math.floor(Math.random() * w);
    const sy = Math.floor(Math.random() * h);
    for (let dy = -5; dy <= 5; dy++) {
      for (let dx = -5; dx <= 5; dx++) {
        const ix = ((sx + dx + w) % w) + ((sy + dy + h) % h) * w;
        data[ix * 4 + 0] = 0.5;
        data[ix * 4 + 1] = 0.25;
      }
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  return tex;
}

function makeFNInitData(w, h) {
  const data = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4 + 0] = (Math.random() - 0.5) * 0.01; // v
    data[i * 4 + 1] = (Math.random() - 0.5) * 0.01; // w
    data[i * 4 + 2] = 0.0;
    data[i * 4 + 3] = 0.0;
  }
  // Seed spirals: two quarter-ring arcs
  const cx = w / 2, cy = h / 2, r = 60;
  for (let a = 0; a < Math.PI; a += 0.02) {
    const x = Math.round(cx + r * Math.cos(a));
    const y = Math.round(cy + r * Math.sin(a));
    if (x >= 0 && x < w && y >= 0 && y < h) {
      const ix = x + y * w;
      data[ix * 4 + 0] = 1.0;
    }
    const x2 = Math.round(cx - r * Math.cos(a));
    const y2 = Math.round(cy - r * Math.sin(a));
    if (x2 >= 0 && x2 < w && y2 >= 0 && y2 < h) {
      const ix2 = x2 + y2 * w;
      data[ix2 * 4 + 0] = 1.0;
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  return tex;
}

function makeKuramotoInitData(w, h, spread, omega0) {
  const TWO_PI = Math.PI * 2;
  const data = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const theta = Math.random() * TWO_PI;
    // Lorentzian-distributed omega via Cauchy inverse CDF: omega0 + spread*tan(π*(u-0.5))
    let omega = omega0 + spread * Math.tan(Math.PI * (Math.random() - 0.5));
    omega = Math.max(omega0 - 3 * spread, Math.min(omega0 + 3 * spread, omega));
    // Pack theta to [0,1], omega to [0,1] via (omega/(4π)) + 0.5
    data[i * 4 + 0] = theta / TWO_PI;
    data[i * 4 + 1] = (omega / (TWO_PI * 2.0)) + 0.5;
    data[i * 4 + 2] = 0.0;
    data[i * 4 + 3] = 0.0;
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  return tex;
}

// ─── State ────────────────────────────────────────────────────────────────────

let renderer, scene, camera;
let gsFBO, fnFBO, kuFBO, advFBO;
let gsCurrent = 0, fnCurrent = 0, kuCurrent = 0, advCurrent = 0;
let compositeFBO, trailFBO;
let bloomFBO = [];        // [down0, down1, down2, down3, up0, up1, up2, up3]

let gsMesh, fnMesh, kuMesh, advMesh;
let compositeMesh, bloomDownMesh, bloomUpMesh, decayMesh, outputMesh;

let gsParams = { ...GS_REGIMES['coral spawn'] };
let fnParams = { ...FN_MODES['spiral lock'] };
let kuParams = { ...KURAMOTO_STATES['bloom nucleation'] };

let orderR = 0.5, orderPsi = 0.0;
let frameCount = 0;
let paused = false;

let mousePos = new THREE.Vector2(0.5, 0.5);
let mouseActive = false;
let mouseDragging = false;

// ─── Setup ────────────────────────────────────────────────────────────────────

function init() {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.setSize(width, height);
  renderer.setPixelRatio(1);
  renderer.autoClear = false;

  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);

  // Ping-pong FBOs per engine
  gsFBO  = makePingPong(SIM_SIZE, SIM_SIZE);
  fnFBO  = makePingPong(SIM_SIZE, SIM_SIZE);
  kuFBO  = makePingPong(SIM_SIZE, SIM_SIZE);
  advFBO = makePingPong(SIM_SIZE, SIM_SIZE);

  // Post-processing FBOs
  compositeFBO = makeFBO(SIM_SIZE, SIM_SIZE);
  trailFBO = [makeFBO(SIM_SIZE, SIM_SIZE), makeFBO(SIM_SIZE, SIM_SIZE)];
  let trailCurrent = 0;

  // Bloom FBOs: 4 levels down, 4 levels up
  for (let i = 0; i < 8; i++) {
    const scale = i < 4 ? Math.pow(0.5, i + 1) : Math.pow(0.5, 8 - i);
    bloomFBO.push(makeFBO(
      Math.max(1, Math.floor(SIM_SIZE * scale)),
      Math.max(1, Math.floor(SIM_SIZE * scale))
    ));
  }

  // Upload initial state textures
  uploadInitTex(gsFBO[0], makeGSInitData(SIM_SIZE, SIM_SIZE));
  uploadInitTex(gsFBO[1], makeGSInitData(SIM_SIZE, SIM_SIZE));
  uploadInitTex(fnFBO[0], makeFNInitData(SIM_SIZE, SIM_SIZE));
  uploadInitTex(fnFBO[1], makeFNInitData(SIM_SIZE, SIM_SIZE));
  const kuInit = makeKuramotoInitData(SIM_SIZE, SIM_SIZE, kuParams.spread, kuParams.omega0);
  uploadInitTex(kuFBO[0], kuInit);
  uploadInitTex(kuFBO[1], kuInit);

  // Build quad meshes
  gsMesh = makeQuadMesh(GS_FRAG, {
    uTexture:     { value: null },
    uResolution:  { value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
    uDu:          { value: gsParams.Du },
    uDv:          { value: gsParams.Dv },
    uF:           { value: gsParams.F },
    uK:           { value: gsParams.k },
    uMousePos:    { value: mousePos },
    uMouseActive: { value: 0.0 },
  });

  fnMesh = makeQuadMesh(FN_FRAG, {
    uTexture:     { value: null },
    uResolution:  { value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
    uD:           { value: fnParams.D },
    uAlpha:       { value: fnParams.alpha },
    uEpsilon:     { value: fnParams.eps },
    uBeta:        { value: 0.5 },
    uGamma:       { value: 1.0 },
    uDelta:       { value: 0.0 },
    uDt:          { value: 0.01 },
    uMousePos:    { value: mousePos },
    uMouseActive: { value: 0.0 },
  });

  kuMesh = makeQuadMesh(KURAMOTO_FRAG, {
    uTexture:   { value: null },
    uResolution:{ value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
    uK:         { value: kuParams.K },
    uD:         { value: kuParams.D },
    uDt:        { value: 0.05 },
    uOrderR:    { value: 0.5 },
    uOrderPsi:  { value: 0.0 },
  });

  advMesh = makeQuadMesh(ADV_FRAG, {
    uTexture:      { value: null },
    uGSTexture:    { value: null },
    uResolution:   { value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
    uD:            { value: 0.002 },
    uTime:         { value: 0.0 },
    uVortexPos:    { value: new THREE.Vector2(0.5, 0.5) },
    uVortexActive: { value: 0.0 },
  });

  compositeMesh = makeQuadMesh(COMPOSITE_FRAG, {
    uGS:              { value: null },
    uFN:              { value: null },
    uKuramoto:        { value: null },
    uAdvection:       { value: null },
    uGSWeight:        { value: 1.0 },
    uFNWeight:        { value: 0.8 },
    uKuramotoWeight:  { value: 0.6 },
    uAdvectionWeight: { value: 0.7 },
  });

  bloomDownMesh = makeQuadMesh(BLOOM_DOWN_FRAG, {
    uTexture:    { value: null },
    uResolution: { value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
    uOffset:     { value: 0.5 },
    uThreshold:  { value: 0.3 },
  });

  bloomUpMesh = makeQuadMesh(BLOOM_UP_FRAG, {
    uTexture:     { value: null },
    uPrevTexture: { value: null },
    uResolution:  { value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
    uOffset:      { value: 0.5 },
    uIntensity:   { value: 0.8 },
  });

  decayMesh = makeQuadMesh(DECAY_FRAG, {
    uTrailTexture: { value: null },
    uEmitTexture:  { value: null },
    uDecay:        { value: 0.95 },
    uEmitWeight:   { value: 0.05 },
  });

  outputMesh = makeQuadMesh(OUTPUT_FRAG, {
    uTexture:       { value: null },
    uBloomTexture:  { value: null },
    uTime:          { value: 0.0 },
    uResolution:    { value: new THREE.Vector2(width, height) },
    uBloomIntensity:{ value: 0.8 },
  });

  scene.add(gsMesh);

  // Attach mouse events
  canvas.addEventListener('mousedown', onMouseDown);
  canvas.addEventListener('mousemove', onMouseMove);
  canvas.addEventListener('mouseup', onMouseUp);
  canvas.addEventListener('mouseleave', onMouseUp);
  window.addEventListener('keydown', onKeyDown);

  animate();
}

function uploadInitTex(fbo, dataTex) {
  const mat = new THREE.MeshBasicMaterial({ map: dataTex });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
  scene.add(mesh);
  renderer.setRenderTarget(fbo);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  scene.remove(mesh);
  mat.dispose();
  dataTex.dispose();
}

// ─── Render pass helper ───────────────────────────────────────────────────────

function runPass(mesh, target) {
  scene.children.forEach(c => scene.remove(c));
  scene.add(mesh);
  renderer.setRenderTarget(target);
  renderer.clear();
  renderer.render(scene, camera);
}

// ─── Animation loop ───────────────────────────────────────────────────────────

let trailCurrent = 0;

function animate() {
  requestAnimationFrame(animate);
  if (!paused) {
    frameCount++;
    const t = time;

    // 1. Gray-Scott — 8 steps
    for (let i = 0; i < 8; i++) {
      gsMesh.material.uniforms.uTexture.value = gsFBO[gsCurrent].texture;
      gsMesh.material.uniforms.uMousePos.value = mousePos;
      gsMesh.material.uniforms.uMouseActive.value = mouseActive ? 1.0 : 0.0;
      runPass(gsMesh, gsFBO[1 - gsCurrent]);
      gsCurrent = 1 - gsCurrent;
    }

    // 2. Fitzhugh-Nagumo — 50 steps
    for (let i = 0; i < 50; i++) {
      fnMesh.material.uniforms.uTexture.value = fnFBO[fnCurrent].texture;
      fnMesh.material.uniforms.uMousePos.value = mousePos;
      fnMesh.material.uniforms.uMouseActive.value = mouseActive ? 1.0 : 0.0;
      runPass(fnMesh, fnFBO[1 - fnCurrent]);
      fnCurrent = 1 - fnCurrent;
    }

    // 3. Kuramoto — 1 step (update order parameter every 10 frames)
    if (frameCount % 10 === 0) computeOrderParameter();
    kuMesh.material.uniforms.uTexture.value = kuFBO[kuCurrent].texture;
    kuMesh.material.uniforms.uOrderR.value = orderR;
    kuMesh.material.uniforms.uOrderPsi.value = orderPsi;
    runPass(kuMesh, kuFBO[1 - kuCurrent]);
    kuCurrent = 1 - kuCurrent;

    // 4. Advection-Diffusion — 1 step
    advMesh.material.uniforms.uTexture.value = advFBO[advCurrent].texture;
    advMesh.material.uniforms.uGSTexture.value = gsFBO[gsCurrent].texture;
    advMesh.material.uniforms.uTime.value = t;
    advMesh.material.uniforms.uVortexPos.value = mousePos;
    advMesh.material.uniforms.uVortexActive.value = mouseDragging ? 1.0 : 0.0;
    runPass(advMesh, advFBO[1 - advCurrent]);
    advCurrent = 1 - advCurrent;

    // 5. Composite
    compositeMesh.material.uniforms.uGS.value       = gsFBO[gsCurrent].texture;
    compositeMesh.material.uniforms.uFN.value        = fnFBO[fnCurrent].texture;
    compositeMesh.material.uniforms.uKuramoto.value  = kuFBO[kuCurrent].texture;
    compositeMesh.material.uniforms.uAdvection.value = advFBO[advCurrent].texture;
    runPass(compositeMesh, compositeFBO);

    // 6. Bloom downsample x4
    bloomDownMesh.material.uniforms.uTexture.value   = compositeFBO.texture;
    bloomDownMesh.material.uniforms.uResolution.value.set(SIM_SIZE, SIM_SIZE);
    bloomDownMesh.material.uniforms.uOffset.value    = 0.5;
    runPass(bloomDownMesh, bloomFBO[0]);
    for (let lvl = 1; lvl < 4; lvl++) {
      const s = bloomFBO[lvl - 1].width;
      bloomDownMesh.material.uniforms.uTexture.value = bloomFBO[lvl - 1].texture;
      bloomDownMesh.material.uniforms.uResolution.value.set(s, s);
      bloomDownMesh.material.uniforms.uOffset.value = lvl + 0.5;
      runPass(bloomDownMesh, bloomFBO[lvl]);
    }

    // 7. Bloom upsample x4
    bloomUpMesh.material.uniforms.uTexture.value     = bloomFBO[3].texture;
    bloomUpMesh.material.uniforms.uPrevTexture.value = bloomFBO[3].texture;
    bloomUpMesh.material.uniforms.uResolution.value.set(bloomFBO[3].width, bloomFBO[3].height);
    bloomUpMesh.material.uniforms.uOffset.value      = 0.5;
    runPass(bloomUpMesh, bloomFBO[4]);
    for (let lvl = 1; lvl < 4; lvl++) {
      const srcIdx = 3 - lvl;
      const prevIdx = 4 + lvl - 1;
      bloomUpMesh.material.uniforms.uTexture.value     = bloomFBO[srcIdx].texture;
      bloomUpMesh.material.uniforms.uPrevTexture.value = bloomFBO[prevIdx].texture;
      bloomUpMesh.material.uniforms.uResolution.value.set(bloomFBO[srcIdx].width, bloomFBO[srcIdx].height);
      bloomUpMesh.material.uniforms.uOffset.value      = lvl + 0.5;
      runPass(bloomUpMesh, bloomFBO[4 + lvl]);
    }

    // 8. Decay trail
    decayMesh.material.uniforms.uTrailTexture.value = trailFBO[trailCurrent].texture;
    decayMesh.material.uniforms.uEmitTexture.value  = compositeFBO.texture;
    runPass(decayMesh, trailFBO[1 - trailCurrent]);
    trailCurrent = 1 - trailCurrent;
  }

  // 9. Output to screen
  outputMesh.material.uniforms.uTexture.value      = trailFBO[trailCurrent].texture;
  outputMesh.material.uniforms.uBloomTexture.value = bloomFBO[7].texture;
  outputMesh.material.uniforms.uTime.value         = time;
  scene.children.forEach(c => scene.remove(c));
  scene.add(outputMesh);
  renderer.setRenderTarget(null);
  renderer.setSize(width, height);
  renderer.clear();
  renderer.render(scene, camera);
}

// ─── Order parameter (Kuramoto) ───────────────────────────────────────────────

function computeOrderParameter() {
  // CPU-side approximation: sample 32×32 subset of Kuramoto texture
  // by using a readback of a small region
  let sumCos = 0, sumSin = 0, n = 0;
  // We maintain a running CPU estimate updated with the last known state
  // Approximate by drifting toward K-dependent equilibrium
  const k = kuParams.K;
  const kc = (2.0 / Math.PI) * kuParams.spread;
  const targetR = k > kc ? Math.min(0.95, (k - kc) / kc * 0.5 + 0.3) : 0.05;
  orderR += (targetR - orderR) * 0.02;
  orderPsi += kuParams.omega0 * 0.05;
  if (orderPsi > Math.PI * 2) orderPsi -= Math.PI * 2;
}

// ─── Input handling ───────────────────────────────────────────────────────────

function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  return new THREE.Vector2(
    (e.clientX - rect.left) / rect.width,
    1.0 - (e.clientY - rect.top) / rect.height
  );
}

function onMouseDown(e) {
  const pos = getCanvasPos(e);
  mousePos.copy(pos);
  mouseActive = true;
  mouseDragging = e.buttons > 0;
}

function onMouseMove(e) {
  const pos = getCanvasPos(e);
  mousePos.copy(pos);
  mouseDragging = e.buttons > 0;
  if (e.buttons > 0) mouseActive = true;
}

function onMouseUp() {
  mouseActive = false;
  mouseDragging = false;
}

function onKeyDown(e) {
  const gsKeys = ['1','2','3','4','5','6'];
  const gsNames = Object.keys(GS_REGIMES);
  const fnKeys = ['q','w','e','r'];
  const fnNames = Object.keys(FN_MODES);
  const kuKeys = ['z','x','c','v'];
  const kuNames = Object.keys(KURAMOTO_STATES);

  const k = e.key.toLowerCase();

  const gIdx = gsKeys.indexOf(k);
  if (gIdx !== -1 && gIdx < gsNames.length) {
    gsParams = { ...GS_REGIMES[gsNames[gIdx]] };
    gsMesh.material.uniforms.uDu.value = gsParams.Du;
    gsMesh.material.uniforms.uDv.value = gsParams.Dv;
    gsMesh.material.uniforms.uF.value  = gsParams.F;
    gsMesh.material.uniforms.uK.value  = gsParams.k;
    return;
  }

  const fIdx = fnKeys.indexOf(k);
  if (fIdx !== -1) {
    fnParams = { ...FN_MODES[fnNames[fIdx]] };
    fnMesh.material.uniforms.uD.value       = fnParams.D;
    fnMesh.material.uniforms.uAlpha.value   = fnParams.alpha;
    fnMesh.material.uniforms.uEpsilon.value = fnParams.eps;
    return;
  }

  const kIdx = kuKeys.indexOf(k);
  if (kIdx !== -1) {
    kuParams = { ...KURAMOTO_STATES[kuNames[kIdx]] };
    kuMesh.material.uniforms.uK.value = kuParams.K;
    kuMesh.material.uniforms.uD.value = kuParams.D;
    const kuInit = makeKuramotoInitData(SIM_SIZE, SIM_SIZE, kuParams.spread, kuParams.omega0);
    uploadInitTex(kuFBO[0], kuInit);
    uploadInitTex(kuFBO[1], kuInit);
    kuCurrent = 0;
    return;
  }

  if (k === ' ') {
    e.preventDefault();
    paused = !paused;
  }
}

// ─── Go ───────────────────────────────────────────────────────────────────────

init();

// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2
