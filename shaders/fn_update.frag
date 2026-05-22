precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform float uD;
uniform float uAlpha;
uniform float uEpsilon;
uniform float uBeta;
uniform float uGamma;
uniform float uDelta;
uniform vec2 uResolution;
uniform vec2 uMousePos;
uniform float uMouseActive;
uniform float uDt;

void main() {
    vec2 texel = 1.0 / uResolution;

    vec2 uvLeft  = vec2(mod(vUv.x - texel.x + 1.0, 1.0), vUv.y);
    vec2 uvRight = vec2(mod(vUv.x + texel.x,        1.0), vUv.y);
    vec2 uvUp    = vec2(vUv.x, mod(vUv.y + texel.y,        1.0));
    vec2 uvDown  = vec2(vUv.x, mod(vUv.y - texel.y + 1.0, 1.0));

    vec4 center = texture2D(uTexture, vUv);
    vec4 left   = texture2D(uTexture, uvLeft);
    vec4 right  = texture2D(uTexture, uvRight);
    vec4 up     = texture2D(uTexture, uvUp);
    vec4 down   = texture2D(uTexture, uvDown);

    float v = center.r;
    float w = center.g;

    // Laplacian of v only
    float lapV = (left.r + right.r + up.r + down.r) - 4.0 * v;

    // Cubic nonlinearity: v*(v - alpha)*(1 - v)
    float cubic = v * (v - uAlpha) * (1.0 - v);

    // External current from mouse
    float iExt = 0.0;
    if (uMouseActive > 0.5) {
        float d = distance(vUv, uMousePos);
        if (d < 0.03) {
            iExt = 0.5;
        }
    }

    float vNext = v + uDt * (uD * lapV + cubic - w + iExt);
    float wNext = w + uDt * (uEpsilon * (uBeta * v - uGamma * w - uDelta));

    // Increment age counter (stored in blue channel, normalized by /1000)
    float age = center.b + 1.0;

    float emission = max(vNext, 0.0);

    gl_FragColor = vec4(vNext, wNext, age / 1000.0, emission);
}
