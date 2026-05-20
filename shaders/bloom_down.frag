#version 300 es
// Dual Kawase Bloom — Downsample Pass
// Samples at ±(offset + 0.5) texel corners, weighted average
// When uThreshold > 0: extract bright regions before downsampling
// Used for 4-level bloom chain: 512→256→128→64→32
//
// Reference: Masaki Kawase, "Frame Buffer Postprocessing Effects in DOUBLE-S.T.E.A.L"
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uInput;
uniform vec2      uResolution;
uniform float     uOffset;
uniform float     uThreshold;

out vec4 fragColor;

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    // Dual Kawase downsample kernel:
    // centre × 4 + 4 corner samples = weighted 5-tap
    float o = uOffset + 0.5;

    vec4 sum = texture(uInput, uv) * 4.0;
    sum += texture(uInput, uv + vec2( o,  o) * texel);
    sum += texture(uInput, uv + vec2(-o,  o) * texel);
    sum += texture(uInput, uv + vec2( o, -o) * texel);
    sum += texture(uInput, uv + vec2(-o, -o) * texel);
    vec4 color = sum / 8.0;

    // Luminance threshold — extract only bright regions (first pass only when threshold > 0)
    if (uThreshold > 0.0) {
        float lum = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
        float knee = max(lum - uThreshold, 0.0);
        color.rgb *= knee / max(lum, 0.0001);
    }

    fragColor = color;
}
