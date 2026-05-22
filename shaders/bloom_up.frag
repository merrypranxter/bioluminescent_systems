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

    // Four diagonal bilinear taps (dual Kawase upsample)
    vec3 s0 = texture2D(uTexture, vUv + vec2(-o, -o) * texel).rgb;
    vec3 s1 = texture2D(uTexture, vUv + vec2( o, -o) * texel).rgb;
    vec3 s2 = texture2D(uTexture, vUv + vec2(-o,  o) * texel).rgb;
    vec3 s3 = texture2D(uTexture, vUv + vec2( o,  o) * texel).rgb;

    vec3 bloom = (s0 + s1 + s2 + s3) * 0.25;

    vec3 prev = texture2D(uPrevTexture, vUv).rgb;

    vec3 combined = bloom + prev * uIntensity;

    gl_FragColor = vec4(combined, 1.0);
}
