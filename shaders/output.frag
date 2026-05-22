precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform sampler2D uBloomTexture;
uniform float uTime;
uniform vec2 uResolution;
uniform float uBloomIntensity;

void main() {
    // ---- 1. Chromatic aberration ----
    // offset grows linearly from center to edge (2px at corners)
    vec2 offset = (vUv - 0.5) * 2.0 / uResolution * 2.0;

    float r = texture2D(uTexture, vUv - offset * 0.002).r;
    float g = texture2D(uTexture, vUv               ).g;
    float b = texture2D(uTexture, vUv + offset * 0.002).b;

    vec3 col = vec3(r, g, b);

    // ---- 2. Add bloom ----
    col += texture2D(uBloomTexture, vUv).rgb * uBloomIntensity;

    // ---- 3. Reinhard tone mapping (luminance-preserving) ----
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float scale = (lum / (1.0 + lum)) / max(lum, 0.0001);
    col *= scale;

    // ---- 4. Vignette ----
    float vig = 1.0 - 0.6 * dot(vUv - 0.5, vUv - 0.5) * 4.0;
    col *= vig;

    // ---- 5. Film grain ----
    float grain = fract(
        sin(dot(vUv + fract(uTime), vec2(12.9898, 78.233))) * 43758.5
    ) * 0.03;
    col += grain - 0.015;  // centered: mean zero

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
