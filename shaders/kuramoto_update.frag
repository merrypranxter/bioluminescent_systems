precision highp float;
varying vec2 vUv;

uniform sampler2D uTexture;
uniform float uK;
uniform float uD;
uniform vec2 uResolution;
uniform float uOrderR;
uniform float uOrderPsi;
uniform float uDt;

#define TWO_PI 6.28318530718

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

    // Unpack phase and natural frequency
    // r: theta packed to [0,1]; g: omega packed via (omega/(TWO_PI*2)) + 0.5
    float theta = center.r * TWO_PI;
    float omega = (center.g - 0.5) * TWO_PI * 2.0;

    // Spatial Laplacian of theta using sin differences to handle wrap-around
    float thetaLeft  = left.r  * TWO_PI;
    float thetaRight = right.r * TWO_PI;
    float thetaUp    = up.r    * TWO_PI;
    float thetaDown  = down.r  * TWO_PI;

    // Phase-aware finite differences via sin(neighbor - center)
    float lapTheta = sin(thetaLeft  - theta)
                   + sin(thetaRight - theta)
                   + sin(thetaUp    - theta)
                   + sin(thetaDown  - theta);

    // Kuramoto mean-field coupling + spatial diffusion
    float dtheta = omega
                 + uK  * uOrderR * sin(uOrderPsi - theta)
                 + uD  * lapTheta;

    float thetaNext = mod(theta + uDt * dtheta, TWO_PI);

    // Emission: cos²(θ/2)
    float cosHalf = cos(thetaNext * 0.5);
    float emission = cosHalf * cosHalf;

    // Pack back to [0,1]; preserve omega channel
    gl_FragColor = vec4(thetaNext / TWO_PI, center.g, 0.0, emission);
}
