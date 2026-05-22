// BioluminescentCompositor.js — Master compositor: owns all 4 engines + post-processing
// Part of the bioluminescent_systems GPGPU pipeline
// THREE is a global — no imports
// Requires: GrayScott, FitzhughNagumo, KuramotoField, AdvectionDiffusion (also globals)

// ─── Shared vertex shader (all post-processing passes) ────────────────────────

const COMP_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ─── Composite pass: blend all four emission layers ───────────────────────────

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
    float gsEmit  = texture2D(uGS,       vUv).a;
    float fnEmit  = texture2D(uFN,       vUv).a;
    float kuEmit  = texture2D(uKuramoto, vUv).a;
    float advEmit = texture2D(uAdvection,vUv).a;

    // GS: deep navy → plankton cyan
    vec3 gsBase  = vec3(0.0,   0.039, 0.078);
    vec3 gsPeak  = vec3(0.0,   0.706, 0.847);
    vec3 gsColor = mix(gsBase, gsPeak, gsEmit);

    // FN: black → jellyfish violet + black → electric teal
    vec3 fnViolet = mix(vec3(0.0), vec3(0.482, 0.176, 0.545), fnEmit * 0.5);
    vec3 fnTeal   = mix(vec3(0.0), vec3(0.282, 0.792, 0.894), fnEmit);
    vec3 fnColor  = fnViolet + fnTeal;

    // Kuramoto: black → firefly green
    vec3 kuColor  = mix(vec3(0.0), vec3(0.659, 1.0, 0.471), kuEmit);

    // Advection: black → white-core teal
    vec3 advColor = mix(vec3(0.0), vec3(0.565, 0.878, 0.937), advEmit);

    vec3 totalColor = gsColor  * uGSWeight
                    + fnColor  * uFNWeight
                    + kuColor  * uKuramotoWeight
                    + advColor * uAdvectionWeight;

    gl_FragColor = vec4(totalColor, 1.0);
}
`;

// ─── Bloom downsample pass (dual Kawase) ──────────────────────────────────────

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

    vec3 s0 = texture2D(uTexture, vUv + vec2(-o, -o) * texel).rgb;
    vec3 s1 = texture2D(uTexture, vUv + vec2( o, -o) * texel).rgb;
    vec3 s2 = texture2D(uTexture, vUv + vec2(-o,  o) * texel).rgb;
    vec3 s3 = texture2D(uTexture, vUv + vec2( o,  o) * texel).rgb;

    vec3 result = (s0 + s1 + s2 + s3) * 0.25;

    // Luminance threshold — soft knee at uThreshold
    result = max(result - uThreshold, 0.0) / max(1.0 - uThreshold, 0.0001);

    gl_FragColor = vec4(result, 1.0);
}
`;

// ─── Bloom upsample pass (dual Kawase) ────────────────────────────────────────

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

    vec3 s0 = texture2D(uTexture, vUv + vec2(-o, -o) * texel).rgb;
    vec3 s1 = texture2D(uTexture, vUv + vec2( o, -o) * texel).rgb;
    vec3 s2 = texture2D(uTexture, vUv + vec2(-o,  o) * texel).rgb;
    vec3 s3 = texture2D(uTexture, vUv + vec2( o,  o) * texel).rgb;

    vec3 bloom = (s0 + s1 + s2 + s3) * 0.25;
    vec3 prev  = texture2D(uPrevTexture, vUv).rgb;

    gl_FragColor = vec4(bloom + prev * uIntensity, 1.0);
}
`;

// ─── Temporal decay pass ──────────────────────────────────────────────────────

const DECAY_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uTrailTexture;
uniform sampler2D uEmitTexture;
uniform float uDecay;
uniform float uEmitWeight;

void main() {
    vec3 trail = texture2D(uTrailTexture, vUv).rgb;
    vec3 emit  = texture2D(uEmitTexture,  vUv).rgb;

    vec3 result = trail * uDecay + emit * uEmitWeight;

    gl_FragColor = vec4(result, 1.0);
}
`;

// ─── Output pass: tone map + bloom add + grain + vignette + chrom. ab. ────────

const OUTPUT_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform sampler2D uBloomTexture;
uniform float uTime;
uniform vec2 uResolution;
uniform float uBloomIntensity;

void main() {
    // 1. Chromatic aberration — offset grows linearly from center to edge
    vec2 offset = (vUv - 0.5) * 2.0 / uResolution * 2.0;

    float r = texture2D(uTexture, vUv - offset * 0.002).r;
    float g = texture2D(uTexture, vUv               ).g;
    float b = texture2D(uTexture, vUv + offset * 0.002).b;

    vec3 col = vec3(r, g, b);

    // 2. Add bloom
    col += texture2D(uBloomTexture, vUv).rgb * uBloomIntensity;

    // 3. Reinhard tone mapping (luminance-preserving, hue intact)
    float lum   = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float scale = (lum / (1.0 + lum)) / max(lum, 0.0001);
    col *= scale;

    // 4. Vignette
    float vig = 1.0 - 0.6 * dot(vUv - 0.5, vUv - 0.5) * 4.0;
    col *= max(vig, 0.0);

    // 5. Film grain (mean-zero, amplitude 0.03)
    float grain = fract(
        sin(dot(vUv + fract(uTime), vec2(12.9898, 78.233))) * 43758.5
    ) * 0.03 - 0.015;
    col += grain;

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

// ─── BioluminescentCompositor ─────────────────────────────────────────────────

class BioluminescentCompositor {
    constructor(renderer, width = 512, height = 512) {
        this.renderer = renderer;
        this.width    = width;
        this.height   = height;

        // Simulation engines
        this.gs        = null;
        this.fn        = null;
        this.kuramoto  = null;
        this.advection = null;

        // Post-processing FBOs
        this.compositeFBO = null;  // one full-res composite FBO
        this.bloomFBO     = [];    // [4 down + 4 up], each half-res of previous
        this.trailFBO     = [null, null]; // ping-pong for decay pass
        this._trailCurrent = 0;

        // Post-processing shader materials
        this._compositeMat  = null;
        this._bloomDownMats = [];  // 4 materials, one per downsample level
        this._bloomUpMats   = [];  // 4 materials, one per upsample level
        this._decayMat      = null;
        this._outputMat     = null;

        // Shared scene/camera/quad for post passes
        this._postScene  = null;
        this._postCamera = null;
        this._postMesh   = null;

        this.init();
    }

    // ─── FBO helpers ──────────────────────────────────────────────────────────

    _makeFBO(w, h) {
        return new THREE.WebGLRenderTarget(w, h, {
            format:        THREE.RGBAFormat,
            type:          THREE.FloatType,
            minFilter:     THREE.LinearFilter,
            magFilter:     THREE.LinearFilter,
            depthBuffer:   false,
            stencilBuffer: false,
        });
    }

    _makeShaderMat(fragSrc, uniforms) {
        return new THREE.ShaderMaterial({
            vertexShader:   COMP_VERT,
            fragmentShader: fragSrc,
            uniforms:       uniforms,
        });
    }

    // ─── Initialise ───────────────────────────────────────────────────────────

    init() {
        const W = this.width;
        const H = this.height;

        // ── Simulation engines ──────────────────────────────────────────────
        this.gs        = new GrayScott(this.renderer, W, H);
        this.fn        = new FitzhughNagumo(this.renderer, W, H);
        this.kuramoto  = new KuramotoField(this.renderer, W, H);
        this.advection = new AdvectionDiffusion(this.renderer, W, H);

        // ── Shared post-processing quad ─────────────────────────────────────
        this._postCamera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
        this._postScene  = new THREE.Scene();
        // mesh material will be swapped per pass via mesh.material assignment
        this._postMesh   = new THREE.Mesh(
            new THREE.PlaneGeometry(1, 1),
            new THREE.MeshBasicMaterial()   // placeholder, replaced per pass
        );
        this._postScene.add(this._postMesh);

        // ── Composite FBO ───────────────────────────────────────────────────
        this.compositeFBO = this._makeFBO(W, H);

        // ── Bloom FBOs: 4 down levels + 4 up levels ─────────────────────────
        // Down: level 0 = W/2, level 1 = W/4, level 2 = W/8, level 3 = W/16
        // Up:   mirrors the down chain back to W/2
        this.bloomFBO = [];
        for (let i = 0; i < 4; i++) {
            const s = Math.pow(2, i + 1);
            this.bloomFBO.push(this._makeFBO(Math.max(1, Math.floor(W / s)),
                                             Math.max(1, Math.floor(H / s))));
        }
        for (let i = 0; i < 4; i++) {
            const s = Math.pow(2, 4 - i);
            this.bloomFBO.push(this._makeFBO(Math.max(1, Math.floor(W / s)),
                                             Math.max(1, Math.floor(H / s))));
        }

        // ── Trail (decay) ping-pong FBOs ────────────────────────────────────
        this.trailFBO[0]  = this._makeFBO(W, H);
        this.trailFBO[1]  = this._makeFBO(W, H);
        this._trailCurrent = 0;

        // ── Composite material ───────────────────────────────────────────────
        this._compositeMat = this._makeShaderMat(COMPOSITE_FRAG, {
            uGS:              { value: null },
            uFN:              { value: null },
            uKuramoto:        { value: null },
            uAdvection:       { value: null },
            uGSWeight:        { value: 1.0 },
            uFNWeight:        { value: 1.0 },
            uKuramotoWeight:  { value: 1.0 },
            uAdvectionWeight: { value: 1.0 },
        });

        // ── Bloom down materials (4 levels) ──────────────────────────────────
        this._bloomDownMats = [];
        for (let i = 0; i < 4; i++) {
            const w = Math.max(1, Math.floor(W / Math.pow(2, i + 1)));
            const h = Math.max(1, Math.floor(H / Math.pow(2, i + 1)));
            this._bloomDownMats.push(this._makeShaderMat(BLOOM_DOWN_FRAG, {
                uTexture:    { value: null },
                uResolution: { value: new THREE.Vector2(w, h) },
                uOffset:     { value: i },
                uThreshold:  { value: 0.3 },
            }));
        }

        // ── Bloom up materials (4 levels) ────────────────────────────────────
        this._bloomUpMats = [];
        for (let i = 0; i < 4; i++) {
            const s = Math.pow(2, 4 - i);
            const w = Math.max(1, Math.floor(W / s));
            const h = Math.max(1, Math.floor(H / s));
            this._bloomUpMats.push(this._makeShaderMat(BLOOM_UP_FRAG, {
                uTexture:     { value: null },
                uPrevTexture: { value: null },
                uResolution:  { value: new THREE.Vector2(w, h) },
                uOffset:      { value: 3 - i },
                uIntensity:   { value: 0.8 },
            }));
        }

        // ── Decay material ───────────────────────────────────────────────────
        this._decayMat = this._makeShaderMat(DECAY_FRAG, {
            uTrailTexture: { value: null },
            uEmitTexture:  { value: null },
            uDecay:        { value: 0.95 },
            uEmitWeight:   { value: 0.05 },
        });

        // ── Output material ──────────────────────────────────────────────────
        this._outputMat = this._makeShaderMat(OUTPUT_FRAG, {
            uTexture:       { value: null },
            uBloomTexture:  { value: null },
            uTime:          { value: 0.0 },
            uResolution:    { value: new THREE.Vector2(W, H) },
            uBloomIntensity:{ value: 0.8 },
        });
    }

    // ─── Internal render-pass helper ──────────────────────────────────────────

    _renderPass(material, targetFBO) {
        this._postMesh.material = material;
        if (targetFBO) {
            this.renderer.setRenderTarget(targetFBO);
        } else {
            this.renderer.setRenderTarget(null);
        }
        this.renderer.render(this._postScene, this._postCamera);
    }

    // ─── Update (called every frame) ──────────────────────────────────────────

    update({ time = 0.0, mouse = null, mouseActive = false, mouseDragging = false, paused = false } = {}) {
        const mousePos = mouse || { x: 0.5, y: 0.5 };

        // ── 1. Simulation steps ─────────────────────────────────────────────
        if (!paused) {
            // Gray-Scott: 8 substeps per frame
            for (let i = 0; i < 8; i++) {
                this.gs.step({ mousePos, mouseActive });
            }

            // Fitzhugh-Nagumo: 50 substeps per frame
            for (let i = 0; i < 50; i++) {
                this.fn.step({ mousePos, mouseActive, dt: 0.01 });
            }

            // Kuramoto: 1 step per frame
            this.kuramoto.step(0.05);

            // Advection-Diffusion: 1 step per frame, sourced from GS
            this.advection.step({
                gsTexture:    this.gs.getTexture(),
                mousePos:     mousePos,
                mouseActive:  mouseActive,
                mouseDragging:mouseDragging,
                time:         time,
            });
        }

        // ── 2. Composite pass: blend all 4 emission layers → compositeFBO ──
        const cm = this._compositeMat.uniforms;
        cm.uGS.value        = this.gs.getTexture();
        cm.uFN.value        = this.fn.getTexture();
        cm.uKuramoto.value  = this.kuramoto.getTexture();
        cm.uAdvection.value = this.advection.getTexture();
        this._renderPass(this._compositeMat, this.compositeFBO);

        // ── 3. Bloom downsample ×4 ─────────────────────────────────────────
        // Level 0 input: composite; levels 1-3 chain from previous down level
        for (let i = 0; i < 4; i++) {
            const dm = this._bloomDownMats[i].uniforms;
            dm.uTexture.value = (i === 0)
                ? this.compositeFBO.texture
                : this.bloomFBO[i - 1].texture;
            this._renderPass(this._bloomDownMats[i], this.bloomFBO[i]);
        }

        // ── 4. Bloom upsample ×4 ──────────────────────────────────────────
        // Level 0 (up) starts from the deepest down level (bloomFBO[3])
        // each up level blends with the corresponding down level above it
        for (let i = 0; i < 4; i++) {
            const um = this._bloomUpMats[i].uniforms;
            // Source texture: deepest level for first up pass, else previous up FBO
            um.uTexture.value = (i === 0)
                ? this.bloomFBO[3].texture
                : this.bloomFBO[4 + i - 1].texture;
            // Previous (higher-res) texture to add to
            um.uPrevTexture.value = (i === 0)
                ? this.bloomFBO[2].texture
                : this.bloomFBO[3 - i].texture;
            this._renderPass(this._bloomUpMats[i], this.bloomFBO[4 + i]);
        }

        // Final bloom result is bloomFBO[7] (the most upsampled level)
        const bloomResult = this.bloomFBO[7].texture;

        // ── 5. Decay pass: trail × 0.95 + emit × 0.05 → trailFBO (swap) ──
        const trailFront = this._trailCurrent;
        const trailBack  = 1 - this._trailCurrent;

        const dc = this._decayMat.uniforms;
        dc.uTrailTexture.value = this.trailFBO[trailFront].texture;
        dc.uEmitTexture.value  = this.compositeFBO.texture;
        this._renderPass(this._decayMat, this.trailFBO[trailBack]);
        this._trailCurrent = trailBack;

        // ── 6. Output pass → screen ─────────────────────────────────────────
        const oc = this._outputMat.uniforms;
        oc.uTexture.value      = this.trailFBO[this._trailCurrent].texture;
        oc.uBloomTexture.value = bloomResult;
        oc.uTime.value         = time;
        this._renderPass(this._outputMat, null);  // null = render to screen
    }

    // ─── Delegation API ───────────────────────────────────────────────────────

    setGSRegime(name) {
        this.gs.setRegime(name);
    }

    setFNMode(name) {
        this.fn.setMode(name);
    }

    setKuramotoState(name) {
        this.kuramoto.setState(name);
    }

    // ─── Mouse injection ──────────────────────────────────────────────────────

    injectAtMouse(pos) {
        // Trigger a single excited step in both GS and FN at the given position,
        // forcing mouseActive=true regardless of the normal event state so the
        // next substep writes emission at the cursor.  We call one extra substep
        // of each engine with the mouse active so the injection registers even if
        // the caller doesn't set mouseActive in the main update call.
        if (!pos) return;
        this.gs.step({ mousePos: pos, mouseActive: true });
        this.fn.step({ mousePos: pos, mouseActive: true, dt: 0.01 });
    }

    // ─── Reset all engines ────────────────────────────────────────────────────

    resetAll() {
        this.gs.reset();
        this.fn.reset();
        this.kuramoto.reset();
        this.advection.reset();
        this._trailCurrent = 0;
    }
}

// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2
