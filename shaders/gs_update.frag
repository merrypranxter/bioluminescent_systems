#version 300 es
// Gray-Scott Reaction-Diffusion Update Shader
// ∂u/∂t = Du·∇²u − uv² + F(1−u)
// ∂v/∂t = Dv·∇²v + uv² − (F+k)v
// 5-point Laplacian stencil, Euler explicit, dt=1.0, clamp [0,1]
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uGS;
uniform vec2      uResolution;
uniform float     uF;
uniform float     uK;
uniform float     uDu;
uniform float     uDv;
uniform vec2      uMouse;
uniform int       uSeed;

out vec4 fragColor;

vec2 wrap(vec2 uv) {
    return mod(uv, 1.0);
}

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    vec4 center = texture(uGS, uv);
    float u = center.r;
    float v = center.g;

    // 5-point Laplacian — u channel
    float u_l = texture(uGS, wrap(uv + vec2(-texel.x,  0.0     ))).r;
    float u_r = texture(uGS, wrap(uv + vec2( texel.x,  0.0     ))).r;
    float u_u = texture(uGS, wrap(uv + vec2( 0.0,      texel.y ))).r;
    float u_d = texture(uGS, wrap(uv + vec2( 0.0,     -texel.y ))).r;
    float lap_u = u_l + u_r + u_u + u_d - 4.0 * u;

    // 5-point Laplacian — v channel
    float v_l = texture(uGS, wrap(uv + vec2(-texel.x,  0.0     ))).g;
    float v_r = texture(uGS, wrap(uv + vec2( texel.x,  0.0     ))).g;
    float v_u = texture(uGS, wrap(uv + vec2( 0.0,      texel.y ))).g;
    float v_d = texture(uGS, wrap(uv + vec2( 0.0,     -texel.y ))).g;
    float lap_v = v_l + v_r + v_u + v_d - 4.0 * v;

    // Autocatalytic reaction term
    float uvv = u * v * v;

    // Euler step, dt = 1.0
    float u_next = u + (uDu * lap_u - uvv + uF * (1.0 - u));
    float v_next = v + (uDv * lap_v + uvv - (uF + uK) * v);

    // Clamp to [0,1] — essential for autocatalytic stability
    u_next = clamp(u_next, 0.0, 1.0);
    v_next = clamp(v_next, 0.0, 1.0);

    // Mouse seed injection (uSeed=1 means active)
    if (uSeed == 1) {
        // Flip y: browser y=0 is top, texture y=0 is bottom
        vec2 mouseUV = vec2(uMouse.x, uResolution.y - uMouse.y) / uResolution;
        float dist = length(uv - mouseUV);
        if (dist < 0.025) {
            u_next = 0.5;
            v_next = 0.25;
        }
    }

    float emission = v_next;
    fragColor = vec4(u_next, v_next, center.b, emission);
}
