#version 300 es
// FitzHugh-Nagumo Excitable Media Update Shader
// ∂v/∂t = D·∇²v + v(v−α)(1−v) − w + I_ext
// ∂w/∂t = ε(β·v − γ·w)
// dt = 0.01 (stiff system — larger dt causes blow-up)
// Texture: .r=v (potential), .g=w (recovery), .b=age, .a=emission
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uFN;
uniform vec2      uResolution;
uniform float     uD_fn;
uniform float     uEps;
uniform float     uAlpha;
uniform float     uBeta;
uniform float     uGamma;
uniform vec2      uMouse;
uniform int       uMouseDown;

out vec4 fragColor;

const float DT = 0.01;

vec2 wrap(vec2 uv) {
    return mod(uv, 1.0);
}

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    vec4 center = texture(uFN, uv);
    float v   = center.r;
    float w   = center.g;
    float age = center.b;

    // 5-point Laplacian — v channel
    float v_l = texture(uFN, wrap(uv + vec2(-texel.x,  0.0     ))).r;
    float v_r = texture(uFN, wrap(uv + vec2( texel.x,  0.0     ))).r;
    float v_u = texture(uFN, wrap(uv + vec2( 0.0,      texel.y ))).r;
    float v_d = texture(uFN, wrap(uv + vec2( 0.0,     -texel.y ))).r;
    float lap_v = v_l + v_r + v_u + v_d - 4.0 * v;

    // External stimulus from mouse drag
    float I_ext = 0.0;
    if (uMouseDown == 1) {
        vec2 mouseUV = vec2(uMouse.x, uResolution.y - uMouse.y) / uResolution;
        float dist = length(uv - mouseUV);
        if (dist < 0.04) {
            // Smooth falloff
            I_ext = 0.6 * (1.0 - dist / 0.04);
        }
    }

    // FitzHugh-Nagumo equations with dt = 0.01
    // Cubic nonlinearity: v(v-α)(1-v) — stiff, requires small dt
    float cubic = v * (v - uAlpha) * (1.0 - v);
    float v_next = v + DT * (uD_fn * lap_v + cubic - w + I_ext);
    float w_next = w + DT * (uEps * (uBeta * v - uGamma * w));

    // Age accumulates (tracks excitation history)
    float age_next = age + DT;

    // Emission: luminance from positive v (refractory w is dark)
    float emission = max(v_next, 0.0);

    fragColor = vec4(v_next, w_next, age_next, emission);
}
