#version 300 es
// Output / Tone-Mapping Shader
// Composites emit + bloom + trail → HDR Reinhard tone map
// + chromatic aberration ±2px at edges
// + vignette 1.0 − 0.6·r²
// + film grain amplitude=0.03
// + hard black threshold below 0.02 emission
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uEmit;
uniform sampler2D uBloom;
uniform sampler2D uTrail;
uniform float     uTime;
uniform vec2      uResolution;
uniform float     uBloomIntensity;   // 0.8

out vec4 fragColor;

// Fast hash for film grain
float hash21(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 17.5);
    return fract(p.x * p.y);
}

void main() {
    vec2 uv     = gl_FragCoord.xy / uResolution;
    vec2 center = vec2(0.5);
    vec2 dir    = uv - center;
    float r     = length(dir);

    // ── Chromatic aberration ±2px at edges ──────────────────────────────────
    // Aberration scales quadratically with distance from centre
    float aberrStrength = 0.004;          // ~2px at half-diagonal
    vec2  aberr = normalize(dir + vec2(0.0001)) * r * aberrStrength;

    float red   = texture(uEmit, uv + aberr).r;
    float green = texture(uEmit, uv        ).g;
    float blue  = texture(uEmit, uv - aberr).b;
    vec3  emit  = vec3(red, green, blue);

    // Bloom and trail contributions
    vec3 bloom  = texture(uBloom, uv).rgb * uBloomIntensity;
    vec3 trail  = texture(uTrail, uv).rgb * 0.35;

    vec3 color  = emit + bloom + trail;

    // ── Luminance-preserving Reinhard tone mapping ───────────────────────────
    float lum    = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float lumOut = lum / (1.0 + lum);
    color        = color * (lumOut / max(lum, 0.0001));

    // ── Vignette: 1 − 0.6·r² ────────────────────────────────────────────────
    float vignette = 1.0 - 0.6 * dot(dir, dir) * 4.0; // 4× so r=0.5 edge=0.4
    vignette = clamp(vignette, 0.0, 1.0);
    color *= vignette;

    // ── Film grain, amplitude 0.03, per-frame random seed ───────────────────
    float grain = hash21(uv + fract(uTime * 0.07)) * 2.0 - 1.0;
    color += grain * 0.03;

    // ── Hard black threshold — void background must be truly black ───────────
    float totalEmit = dot(color, vec3(1.0)) / 3.0;
    if (totalEmit < 0.02) {
        color = vec3(0.0);
    }

    color = clamp(color, 0.0, 1.0);
    fragColor = vec4(color, 1.0);
}
