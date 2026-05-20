#version 300 es
// Dual Kawase Bloom — Upsample Pass
// 8-tap tent filter with weighted corners
// Used for 4-level bloom chain: 32→64→128→256→512
//
// Reference: Masaki Kawase, "Frame Buffer Postprocessing Effects in DOUBLE-S.T.E.A.L"
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uInput;
uniform vec2      uResolution;
uniform float     uOffset;

out vec4 fragColor;

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    float o = uOffset;

    // Dual Kawase upsample — 8-tap tent filter
    // Cardinal neighbours ×1, diagonal neighbours ×2 → sum / 12
    vec4 sum = vec4(0.0);
    sum += texture(uInput, uv + vec2(-o,  0.0) * texel);
    sum += texture(uInput, uv + vec2(-o,   o ) * texel) * 2.0;
    sum += texture(uInput, uv + vec2( 0.0,  o) * texel);
    sum += texture(uInput, uv + vec2(  o,   o) * texel) * 2.0;
    sum += texture(uInput, uv + vec2(  o,  0.0) * texel);
    sum += texture(uInput, uv + vec2(  o,  -o) * texel) * 2.0;
    sum += texture(uInput, uv + vec2( 0.0, -o) * texel);
    sum += texture(uInput, uv + vec2(-o,   -o) * texel) * 2.0;

    fragColor = sum / 12.0;
}
