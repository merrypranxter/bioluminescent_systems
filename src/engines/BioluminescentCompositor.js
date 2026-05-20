// BioluminescentCompositor.js
// Manages all four simulation engines, post-processing pipeline, and render loop.
//
// Pipeline order each frame:
//   AdvectionDiffusion.step(gsTexture)
//   GrayScott.step()
//   FitzhughNagumo.step()
//   KuramotoField.step(time)
//   → composite pass  → emitFBO
//   → bloom chain     → bloomFinal (dual Kawase, 4 levels)
//   → decay pass      → trailFBO
//   → output pass     → screen
//
// All simulation textures: 512×512 RGBA32F
// Additive blending: SRC_ALPHA, ONE
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source: https://github.com/merrypranxter/reposcripter2

/* global THREE */

import { GrayScott }         from './GrayScott.js';
import { FitzhughNagumo }    from './FitzhughNagumo.js';
import { KuramotoField }     from './KuramotoField.js';
import { AdvectionDiffusion } from './AdvectionDiffusion.js';

// ── Shared vertex shader ──────────────────────────────────────────────────────
const VERT = `#version 300 es
in vec3 position;
void main() { gl_Position = vec4(position, 1.0); }`;

// ── Composite fragment shader ─────────────────────────────────────────────────
const COMPOSITE_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uGS;
uniform sampler2D uFN;
uniform sampler2D uKU;
uniform sampler2D uAdv;
uniform float uGSWeight;
uniform float uFNWeight;
uniform float uKUWeight;
uniform float uAdvWeight;
out vec4 fragColor;

void main() {
    vec2 uv     = gl_FragCoord.xy / vec2(textureSize(uGS, 0));
    float gsV   = texture(uGS,  uv).g;
    float fnV   = clamp(texture(uFN, uv).r, 0.0, 1.0);
    float kuE   = texture(uKU,  uv).a;
    float advPhi= texture(uAdv, uv).r;

    // Advection: abyss #000a14 → plankton_cyan #00b4d8
    vec3 advColor = mix(vec3(0.0, 0.039, 0.078), vec3(0.0, 0.706, 0.847), advPhi) * uAdvWeight;
    // Gray-Scott: void → electric_teal #48cae4
    vec3 gsColor  = mix(vec3(0.0), vec3(0.282, 0.792, 0.969), gsV) * uGSWeight;
    // FitzHugh-Nagumo: jellyfish_violet #7b2d8b → firefly_green #a8ff78
    vec3 fnBase   = mix(vec3(0.482, 0.173, 0.549), vec3(0.565, 1.0, 0.471), fnV);
    vec3 fnColor  = mix(vec3(0.0), fnBase, fnV) * uFNWeight;
    // Kuramoto: spiral_purple #a855f7
    vec3 kuColor  = mix(vec3(0.0), vec3(0.659, 0.333, 0.969), kuE) * uKUWeight;

    vec3  color = advColor + gsColor + fnColor + kuColor;
    float maxEm = max(max(advPhi * uAdvWeight, gsV * uGSWeight),
                      max(fnV   * uFNWeight,   kuE * uKUWeight));
    fragColor = vec4(color, maxEm);
}`;

// ── Bloom downsample ──────────────────────────────────────────────────────────
const BLOOM_DOWN_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uInput;
uniform vec2  uResolution;
uniform float uOffset;
uniform float uThreshold;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    vec2 t  = 1.0 / uResolution;
    float o = uOffset + 0.5;
    vec4 s  = texture(uInput, uv) * 4.0;
    s += texture(uInput, uv + vec2( o,  o) * t);
    s += texture(uInput, uv + vec2(-o,  o) * t);
    s += texture(uInput, uv + vec2( o, -o) * t);
    s += texture(uInput, uv + vec2(-o, -o) * t);
    vec4 color = s / 8.0;
    if (uThreshold > 0.0) {
        float lum  = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
        float knee = max(lum - uThreshold, 0.0);
        color.rgb *= knee / max(lum, 0.0001);
    }
    fragColor = color;
}`;

// ── Bloom upsample ────────────────────────────────────────────────────────────
const BLOOM_UP_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uInput;
uniform vec2  uResolution;
uniform float uOffset;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    vec2 t  = 1.0 / uResolution;
    float o = uOffset;
    vec4 s  = vec4(0.0);
    s += texture(uInput, uv + vec2(-o,  0.0) * t);
    s += texture(uInput, uv + vec2(-o,   o ) * t) * 2.0;
    s += texture(uInput, uv + vec2( 0.0,  o) * t);
    s += texture(uInput, uv + vec2(  o,   o) * t) * 2.0;
    s += texture(uInput, uv + vec2(  o,  0.0) * t);
    s += texture(uInput, uv + vec2(  o,  -o) * t) * 2.0;
    s += texture(uInput, uv + vec2( 0.0, -o) * t);
    s += texture(uInput, uv + vec2(-o,   -o) * t) * 2.0;
    fragColor = s / 12.0;
}`;

// ── Decay (trail accumulation) ────────────────────────────────────────────────
const DECAY_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTrail;
uniform sampler2D uCurrent;
uniform float uDecay;
out vec4 fragColor;
void main() {
    vec2 uv  = gl_FragCoord.xy / vec2(textureSize(uTrail, 0));
    fragColor = texture(uTrail, uv) * uDecay + texture(uCurrent, uv) * (1.0 - uDecay);
}`;

// ── Output (tone-map, grain, vignette, chromatic aberration) ──────────────────
const OUTPUT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uEmit;
uniform sampler2D uBloom;
uniform sampler2D uTrail;
uniform float uTime;
uniform vec2  uResolution;
uniform float uBloomIntensity;
out vec4 fragColor;

float hash21(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 17.5);
    return fract(p.x * p.y);
}

void main() {
    vec2  uv     = gl_FragCoord.xy / uResolution;
    vec2  center = vec2(0.5);
    vec2  dir    = uv - center;
    float r      = length(dir);

    // Chromatic aberration: ±2px at edges (quadratic radial falloff)
    vec2  aberr = normalize(dir + vec2(0.0001)) * r * 0.004;
    float red   = texture(uEmit, uv + aberr).r;
    float green = texture(uEmit, uv        ).g;
    float blue  = texture(uEmit, uv - aberr).b;
    vec3  emit  = vec3(red, green, blue);

    vec3 bloom  = texture(uBloom, uv).rgb * uBloomIntensity;
    vec3 trail  = texture(uTrail, uv).rgb * 0.35;
    vec3 color  = emit + bloom + trail;

    // Luminance-preserving Reinhard tone mapping
    float lum    = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float lumOut = lum / (1.0 + lum);
    color        = color * (lumOut / max(lum, 0.0001));

    // Vignette: 1 − 0.6·r² (r measured from 0.5 centre, normalised so edge=1)
    float vignette = clamp(1.0 - 0.6 * dot(dir * 2.0, dir * 2.0), 0.0, 1.0);
    color *= vignette;

    // Film grain amplitude=0.03, per-frame seed
    color += (hash21(uv + fract(uTime * 0.07)) * 2.0 - 1.0) * 0.03;

    // Hard black threshold — void must be truly black
    if (dot(color, vec3(1.0)) / 3.0 < 0.02) color = vec3(0.0);

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}`;

// ─────────────────────────────────────────────────────────────────────────────

export class BioluminescentCompositor {
    /** @param {HTMLCanvasElement} canvas */
    constructor(canvas) {
        this.canvas  = canvas;
        this.renderer = null;

        // Engines
        this.grayScott    = null;
        this.fitzhughNagumo = null;
        this.kuramotoField  = null;
        this.advection      = null;

        // Post-processing FBOs
        this.emitFBO   = null;   // composite output
        this.trailFBO  = null;   // temporal accumulation
        this.bloomDown = [];     // sizes: 256, 128, 64, 32
        this.bloomUp   = [];     // sizes: 64, 128, 256
        this.bloomFinalFBO = null; // 512×512 final bloom result

        // Pass materials
        this.compositeMat  = null;
        this.bloomDownMats = [];
        this.bloomUpMats   = [];
        this.bloomFinalMat = null;
        this.decayMat      = null;
        this.outputMat     = null;

        // Shared fullscreen quad + orthographic camera
        this._quad   = new THREE.PlaneGeometry(2, 2);
        this._camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        this.simSize    = 512;
        this.paused     = false;
        this._frameTime = 0;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    _createFBO(w, h) {
        return new THREE.WebGLRenderTarget(w, h, {
            type:          THREE.FloatType,
            format:        THREE.RGBAFormat,
            minFilter:     THREE.LinearFilter,
            magFilter:     THREE.LinearFilter,
            wrapS:         THREE.ClampToEdgeWrapping,
            wrapT:         THREE.ClampToEdgeWrapping,
            depthBuffer:   false,
            stencilBuffer: false,
        });
    }

    _makeScene(material) {
        const mesh  = new THREE.Mesh(this._quad, material);
        const scene = new THREE.Scene();
        scene.add(mesh);
        return scene;
    }

    _renderPass(material, target, w, h) {
        const scene = this._makeScene(material);
        this.renderer.setRenderTarget(target);
        if (target !== null) {
            this.renderer.setViewport(0, 0, w, h);
        } else {
            this.renderer.setViewport(
                0, 0, this.canvas.width, this.canvas.height
            );
        }
        this.renderer.render(scene, this._camera);
        if (target !== null) this.renderer.setRenderTarget(null);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** Create renderer, all engines, all FBOs, all post-processing materials. */
    init() {
        // ── WebGL2 renderer ──────────────────────────────────────────────────
        this.renderer = new THREE.WebGLRenderer({
            canvas:    this.canvas,
            antialias: false,
            alpha:     false,
            powerPreference: 'high-performance',
        });
        this.renderer.setPixelRatio(1);
        this.renderer.setSize(this.canvas.width, this.canvas.height, false);
        this.renderer.autoClear = false;

        // Additive blending for all emission layers
        this.renderer.setOpaqueSort(null);

        const S = this.simSize;

        // ── Simulation engines ────────────────────────────────────────────────
        this.grayScott      = new GrayScott(this.renderer, S);
        this.fitzhughNagumo = new FitzhughNagumo(this.renderer, S);
        this.kuramotoField  = new KuramotoField(this.renderer, S);
        this.advection      = new AdvectionDiffusion(this.renderer, S);

        this.grayScott.init('coral spawn');
        this.fitzhughNagumo.init('spiral lock');
        this.kuramotoField.init('bloom nucleation');
        this.advection.init('plankton drift');

        // ── Post-processing FBOs ──────────────────────────────────────────────
        this.emitFBO       = this._createFBO(S, S);
        this.trailFBO      = this._createFBO(S, S);
        this._trailSwap    = this._createFBO(S, S);  // ping-pong trail pair
        this.bloomFinalFBO = this._createFBO(S, S);

        // Bloom downsample chain: 256, 128, 64, 32
        const downSizes = [256, 128, 64, 32];
        downSizes.forEach(sz => this.bloomDown.push(this._createFBO(sz, sz)));

        // Bloom upsample chain: 64, 128, 256
        const upSizes = [64, 128, 256];
        upSizes.forEach(sz => this.bloomUp.push(this._createFBO(sz, sz)));

        // ── Composite material ────────────────────────────────────────────────
        this.compositeUniforms = {
            uGS:       { value: this.grayScott.getTexture() },
            uFN:       { value: this.fitzhughNagumo.getTexture() },
            uKU:       { value: this.kuramotoField.getTexture() },
            uAdv:      { value: this.advection.getTexture() },
            uGSWeight: { value: 1.0 },
            uFNWeight: { value: 1.0 },
            uKUWeight: { value: 1.0 },
            uAdvWeight:{ value: 0.8 },
        };
        this.compositeMat = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: COMPOSITE_FRAG,
            uniforms:       this.compositeUniforms,
        });

        // ── Bloom downsample materials ────────────────────────────────────────
        // Pass 0 uses threshold=0.3; subsequent passes use threshold=0
        const bloomDownSrc = [this.emitFBO, ...this.bloomDown.slice(0, 3)];
        this.bloomDownMats = bloomDownSrc.map((src, i) => {
            const u = {
                uInput:     { value: src.texture },
                uResolution:{ value: new THREE.Vector2(downSizes[i], downSizes[i]) },
                uOffset:    { value: 1.0 },
                uThreshold: { value: i === 0 ? 0.3 : 0.0 },
            };
            return new THREE.RawShaderMaterial({
                vertexShader:   VERT,
                fragmentShader: BLOOM_DOWN_FRAG,
                uniforms:       u,
            });
        });

        // ── Bloom upsample materials ──────────────────────────────────────────
        // upsample sources: bloomDown[3] → bloomUp[0], bloomUp[0] → bloomUp[1], ...
        const bloomUpSrc  = [this.bloomDown[3], ...this.bloomUp.slice(0, 2)];
        this.bloomUpMats  = bloomUpSrc.map((src, i) => {
            const u = {
                uInput:     { value: src.texture },
                uResolution:{ value: new THREE.Vector2(upSizes[i], upSizes[i]) },
                uOffset:    { value: 1.0 },
            };
            return new THREE.RawShaderMaterial({
                vertexShader:   VERT,
                fragmentShader: BLOOM_UP_FRAG,
                uniforms:       u,
            });
        });

        // Final upsample: bloomUp[2] → bloomFinalFBO (512×512)
        this.bloomFinalUniforms = {
            uInput:     { value: this.bloomUp[2].texture },
            uResolution:{ value: new THREE.Vector2(S, S) },
            uOffset:    { value: 1.0 },
        };
        this.bloomFinalMat = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: BLOOM_UP_FRAG,
            uniforms:       this.bloomFinalUniforms,
        });

        // ── Decay material ────────────────────────────────────────────────────
        this.decayUniforms = {
            uTrail:   { value: this.trailFBO.texture },
            uCurrent: { value: this.emitFBO.texture },
            uDecay:   { value: 0.95 },
        };
        this.decayMat = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: DECAY_FRAG,
            uniforms:       this.decayUniforms,
        });

        // ── Output material ───────────────────────────────────────────────────
        this.outputUniforms = {
            uEmit:           { value: this.emitFBO.texture },
            uBloom:          { value: this.bloomFinalFBO.texture },
            uTrail:          { value: this.trailFBO.texture },
            uTime:           { value: 0.0 },
            uResolution:     { value: new THREE.Vector2(
                this.canvas.width, this.canvas.height) },
            uBloomIntensity: { value: 0.8 },
        };
        this.outputMat = new THREE.RawShaderMaterial({
            vertexShader:   VERT,
            fragmentShader: OUTPUT_FRAG,
            uniforms:       this.outputUniforms,
        });
    }

    /**
     * Step all engines + run all post-processing passes (except final output).
     * @param {number} time   Elapsed seconds
     * @param {{x:number, y:number}} mouse  Cursor position in canvas pixels
     */
    update(time, mouse) {
        if (this.paused) return;
        this._frameTime = time;

        const S = this.simSize;

        // ── Simulation steps ──────────────────────────────────────────────────
        this.advection.step(this.grayScott.getTexture(), time);
        this.grayScott.step();
        this.fitzhughNagumo.step();
        this.kuramotoField.step(time);

        // ── Update composite uniforms ─────────────────────────────────────────
        this.compositeUniforms.uGS.value  = this.grayScott.getTexture();
        this.compositeUniforms.uFN.value  = this.fitzhughNagumo.getTexture();
        this.compositeUniforms.uKU.value  = this.kuramotoField.getTexture();
        this.compositeUniforms.uAdv.value = this.advection.getTexture();

        // ── Composite pass → emitFBO ──────────────────────────────────────────
        this._renderPass(this.compositeMat, this.emitFBO, S, S);

        // ── Bloom downsample chain ────────────────────────────────────────────
        const downSizes = [256, 128, 64, 32];
        this.bloomDownMats.forEach((mat, i) => {
            const src = i === 0 ? this.emitFBO : this.bloomDown[i - 1];
            mat.uniforms.uInput.value = src.texture;
            this._renderPass(mat, this.bloomDown[i],
                downSizes[i], downSizes[i]);
        });

        // ── Bloom upsample chain ──────────────────────────────────────────────
        const upSizes = [64, 128, 256];
        this.bloomUpMats.forEach((mat, i) => {
            const src = i === 0 ? this.bloomDown[3] : this.bloomUp[i - 1];
            mat.uniforms.uInput.value = src.texture;
            this._renderPass(mat, this.bloomUp[i], upSizes[i], upSizes[i]);
        });

        // Final bloom upsample → bloomFinalFBO (512×512)
        this.bloomFinalUniforms.uInput.value = this.bloomUp[2].texture;
        this._renderPass(this.bloomFinalMat, this.bloomFinalFBO, S, S);

        // ── Decay pass → _trailSwap (ping-pong to avoid read/write aliasing) ────
        // Read from trailFBO, write to _trailSwap, then swap
        this.decayUniforms.uTrail.value   = this.trailFBO.texture;
        this.decayUniforms.uCurrent.value = this.emitFBO.texture;
        this._renderPass(this.decayMat, this._trailSwap, S, S);

        // Swap ping-pong trail pair
        const tmp       = this.trailFBO;
        this.trailFBO   = this._trailSwap;
        this._trailSwap = tmp;

        // Update output uniforms
        this.outputUniforms.uEmit.value  = this.emitFBO.texture;
        this.outputUniforms.uBloom.value = this.bloomFinalFBO.texture;
        this.outputUniforms.uTrail.value = this.trailFBO.texture;
        this.outputUniforms.uTime.value  = time;
    }

    /** Run the final output pass to the screen canvas. */
    render() {
        this.outputUniforms.uResolution.value.set(
            this.canvas.width, this.canvas.height
        );
        this._renderPass(this.outputMat, null,
            this.canvas.width, this.canvas.height);
    }

    /**
     * Toggle which engine is displayed at full weight, dim others.
     * @param {string} name  'gray-scott' | 'fitzhugh-nagumo' | 'kuramoto' | 'advection' | 'all'
     */
    setActiveEngine(name) {
        const cu = this.compositeUniforms;
        // Reset all to 0.15 (dim but not invisible)
        cu.uGSWeight.value  = 0.15;
        cu.uFNWeight.value  = 0.15;
        cu.uKUWeight.value  = 0.15;
        cu.uAdvWeight.value = 0.15;
        switch (name) {
            case 'gray-scott':     cu.uGSWeight.value  = 1.0; break;
            case 'fitzhugh-nagumo':cu.uFNWeight.value  = 1.0; break;
            case 'kuramoto':       cu.uKUWeight.value  = 1.0; break;
            case 'advection':      cu.uAdvWeight.value = 1.0; break;
            default: // 'all'
                cu.uGSWeight.value  = 1.0;
                cu.uFNWeight.value  = 1.0;
                cu.uKUWeight.value  = 1.0;
                cu.uAdvWeight.value = 0.8;
        }
    }

    /**
     * Sample the local order parameter r from the Kuramoto texture centre.
     * Reads a single pixel from the GPU — use sparingly.
     * @returns {number} r ∈ [0,1]
     */
    getOrderParameter() {
        const buf = new Float32Array(4);
        try {
            this.renderer.readRenderTargetPixels(
                this.kuramotoField.readFBO,
                Math.floor(this.simSize / 2),
                Math.floor(this.simSize / 2),
                1, 1, buf
            );
        } catch (_) {
            return 0;
        }
        return buf[2]; // .b = r_local
    }

    /** Handle canvas resize. */
    resize(w, h) {
        this.renderer.setSize(w, h, false);
        this.outputUniforms.uResolution.value.set(w, h);
    }

    // ── Internal trail init ───────────────────────────────────────────────────
    _initTrailSwap() {
        // _trailSwap is now created inline in init(); this is a no-op kept for API compat.
    }
}

// No monkey-patch required — _trailSwap is allocated inside init() above.
