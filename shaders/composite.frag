#version 300 es
// Composite Shader — blends all four emission layers with distinct palette colors
// Layer order: advection → gray-scott → fitzhugh-nagumo → kuramoto
// Additive blending produces hot white cores at dense emission zones
//
// Color palette (exact hex):
//   abyss          #000a14  →  plankton_cyan #00b4d8  (advection)
//   void #000000   →  electric_teal #48cae4            (gray-scott)
//   jellyfish_violet #7b2d8b → firefly_green #a8ff78   (fitzhugh-nagumo)
//   spiral_purple  #a855f7                              (kuramoto)
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uGS;
uniform sampler2D uFN;
uniform sampler2D uKU;
uniform sampler2D uAdv;
uniform float     uGSWeight;
uniform float     uFNWeight;
uniform float     uKUWeight;
uniform float     uAdvWeight;

out vec4 fragColor;

void main() {
    // Compute UV from texel coordinates (all textures same resolution)
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uGS, 0));

    // ── Sample each simulation texture ──────────────────────────────────────
    vec4 gsData  = texture(uGS,  uv);
    vec4 fnData  = texture(uFN,  uv);
    vec4 kuData  = texture(uKU,  uv);
    vec4 advData = texture(uAdv, uv);

    // Channel extraction (per spec):
    //   GS  .g = v (inhibitor = bioluminescent emission)
    //   FN  .r = v (membrane potential), clamped to [0,1]
    //   KU  .a = emission (cos²(θ/2))
    //   Adv .r = φ (emission field)
    float gsV    = gsData.g;
    float fnV    = clamp(fnData.r, 0.0, 1.0);
    float kuE    = kuData.a;
    float advPhi = advData.r;

    // ── Color mapping ────────────────────────────────────────────────────────

    // Advection: abyss (#000a14) → plankton_cyan (#00b4d8)
    vec3 advColor = mix(vec3(0.0, 0.039, 0.078),
                        vec3(0.0, 0.706, 0.847),
                        advPhi) * uAdvWeight;

    // Gray-Scott: void (#000000) → electric_teal (#48cae4 ≈ 0.282,0.792,0.969)
    vec3 gsColor  = mix(vec3(0.0),
                        vec3(0.282, 0.792, 0.969),
                        gsV) * uGSWeight;

    // FitzHugh-Nagumo: jellyfish_violet (#7b2d8b) → firefly_green (#a8ff78)
    // Low v → violet, high v → acid green; zero emission → black
    vec3 fnBase   = mix(vec3(0.482, 0.173, 0.549), vec3(0.565, 1.0, 0.471), fnV);
    vec3 fnColor  = mix(vec3(0.0), fnBase, fnV) * uFNWeight;

    // Kuramoto: spiral_purple (#a855f7)
    vec3 kuColor  = mix(vec3(0.0),
                        vec3(0.659, 0.333, 0.969),
                        kuE) * uKUWeight;

    // Additive blend: all layers accumulate light
    vec3  color  = advColor + gsColor + fnColor + kuColor;
    float maxEm  = max(max(advPhi * uAdvWeight, gsV * uGSWeight),
                       max(fnV   * uFNWeight,   kuE * uKUWeight));

    fragColor = vec4(color, maxEm);
}
