precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform float uDu;
uniform float uDv;
uniform float uF;
uniform float uK;
uniform vec2 uResolution;
uniform vec2 uMousePos;
uniform float uMouseActive;

void main() {
    vec2 texel = 1.0 / uResolution;

    // Wrapped neighbor sampling
    vec2 uvLeft  = vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y);
    vec2 uvRight = vec2(mod(vUv.x + texel.x,        1.0), vUv.y);
    vec2 uvUp    = vec2(vUv.x, mod(vUv.y + texel.y,        1.0));
    vec2 uvDown  = vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0));

    vec4 center = texture2D(uTexture, vUv);
    vec4 left   = texture2D(uTexture, uvLeft);
    vec4 right  = texture2D(uTexture, uvRight);
    vec4 up     = texture2D(uTexture, uvUp);
    vec4 down   = texture2D(uTexture, uvDown);

    float u = center.r;
    float v = center.g;

    // 5-point Laplacian
    float lapU = (left.r + right.r + up.r + down.r) - 4.0 * u;
    float lapV = (left.g + right.g + up.g + down.g) - 4.0 * v;

    float dt = 1.0;
    float uvv = u * v * v;

    float uNext = u + dt * (uDu * lapU - uvv + uF * (1.0 - u));
    float vNext = v + dt * (uDv * lapV + uvv - (uF + uK) * v);

    uNext = clamp(uNext, 0.0, 1.0);
    vNext = clamp(vNext, 0.0, 1.0);

    // Mouse interaction: seed v at cursor
    if (uMouseActive > 0.5) {
        float d = distance(vUv, uMousePos);
        if (d < 0.02) {
            vNext = min(vNext + 0.5, 1.0);
        }
    }

    gl_FragColor = vec4(uNext, vNext, 0.0, vNext);
}
