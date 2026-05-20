#version 300 es
// Advection-Diffusion Shader
// ∂φ/∂t + u⃗·∇φ = D·∇²φ + S(x,t)
// Upwind differencing for advection (stability-preserving)
// Curl-noise divergence-free velocity field
// Gray-Scott v field as bioluminescent source term
// Texture: .r=φ, .g=u_vel, .b=v_vel, .a=S_accumulated
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uAdv;
uniform sampler2D uGSSource;
uniform vec2      uResolution;
uniform float     uD_adv;
uniform float     uTime;
uniform vec2      uMouse;
uniform float     uVelScale;
uniform vec3      uVortices[8];
uniform int       uNumVortices;

out vec4 fragColor;

const float PI = 3.14159265358979;

// ─── FBM + Curl-Noise helpers ─────────────────────────────────────────────────
float hash(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i),               hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

vec2 curlNoise(vec2 uv, float t) {
    float eps = 0.001;
    float psi_y_plus  = fbm((uv + vec2(0.0,  eps)) * 3.0 + t * 0.0005);
    float psi_y_minus = fbm((uv - vec2(0.0,  eps)) * 3.0 + t * 0.0005);
    float psi_x_plus  = fbm((uv + vec2(eps,  0.0)) * 3.0 + t * 0.0005);
    float psi_x_minus = fbm((uv - vec2(eps,  0.0)) * 3.0 + t * 0.0005);
    float dpsi_dy = (psi_y_plus  - psi_y_minus) / (2.0 * eps);
    float dpsi_dx = (psi_x_plus  - psi_x_minus) / (2.0 * eps);
    return vec2(dpsi_dy, -dpsi_dx);
}
// ─────────────────────────────────────────────────────────────────────────────

vec2 wrap(vec2 uv) {
    return mod(uv, 1.0);
}

void main() {
    vec2 uv    = gl_FragCoord.xy / uResolution;
    vec2 texel = 1.0 / uResolution;

    float phi   = texture(uAdv, uv).r;
    float phi_l = texture(uAdv, wrap(uv + vec2(-texel.x,  0.0     ))).r;
    float phi_r = texture(uAdv, wrap(uv + vec2( texel.x,  0.0     ))).r;
    float phi_u = texture(uAdv, wrap(uv + vec2( 0.0,      texel.y ))).r;
    float phi_d = texture(uAdv, wrap(uv + vec2( 0.0,     -texel.y ))).r;

    // Curl-noise velocity — divergence-free, scale to ~0–1 pixels/step
    vec2 vel = curlNoise(uv, uTime) * uVelScale;

    // Explicit vortex contributions (Biot-Savart, regularised core ε²=0.0001)
    for (int i = 0; i < 8; i++) {
        if (i >= uNumVortices) break;
        vec2  diff   = uv - uVortices[i].xy;
        float dist2  = dot(diff, diff) + 0.0001;          // regularised ε²
        float Gamma  = uVortices[i].z;
        vec2  perp   = vec2(-diff.y, diff.x);
        vel += (Gamma / (2.0 * PI)) * perp / dist2;
    }

    // Clamp velocity for CFL stability (|vel| ≤ 1 pixel/step for upwind)
    float velMag = length(vel);
    if (velMag > 0.95) vel = vel * (0.95 / velMag);

    // ── Upwind differencing for advection ──────────────────────────────────
    // x-direction: backward if vel.x > 0 (information comes from left)
    float dphi_dx = (vel.x >= 0.0) ? (phi - phi_l) : (phi_r - phi);
    // y-direction
    float dphi_dy = (vel.y >= 0.0) ? (phi - phi_d) : (phi_u - phi);

    // ── Central-difference Laplacian for diffusion ─────────────────────────
    float lap_phi = phi_l + phi_r + phi_u + phi_d - 4.0 * phi;

    // ── Source term: Gray-Scott v field (bioluminescent injection) ──────────
    float S = texture(uGSSource, uv).g * 0.018;

    // Advection-diffusion step, dt = 1.0 (pixel grid)
    float phi_next = phi + (-vel.x * dphi_dx - vel.y * dphi_dy
                            + uD_adv * lap_phi
                            + S);
    phi_next = clamp(phi_next, 0.0, 1.0);

    // Store phi, computed velocity, and accumulated source
    float s_acc = clamp(texture(uAdv, uv).a * 0.98 + S, 0.0, 1.0);
    fragColor = vec4(phi_next, vel.x * 0.5 + 0.5, vel.y * 0.5 + 0.5, s_acc);
}
