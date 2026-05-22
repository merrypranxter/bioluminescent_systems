precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float uOffset;
uniform float uThreshold;

void main() {
    vec2 texel = 1.0 / uResolution;
    float o = uOffset + 0.5;

    // Four diagonal bilinear taps (dual Kawase downsample)
    vec3 s0 = texture2D(uTexture, vUv + vec2(-o, -o) * texel).rgb;
    vec3 s1 = texture2D(uTexture, vUv + vec2( o, -o) * texel).rgb;
    vec3 s2 = texture2D(uTexture, vUv + vec2(-o,  o) * texel).rgb;
    vec3 s3 = texture2D(uTexture, vUv + vec2( o,  o) * texel).rgb;

    vec3 result = (s0 + s1 + s2 + s3) * 0.25;

    // Luminance threshold — soft knee
    result = max(result - uThreshold, 0.0) / (1.0 - uThreshold);

    gl_FragColor = vec4(result, 1.0);
}
