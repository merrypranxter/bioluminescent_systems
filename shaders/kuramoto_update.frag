#version 300 es
// Kuramoto Phase Field Update Shader
// ∂θ/∂t = ω(x) + K·r·sin(Ψ − θ) + D·∇²θ
// Phase θ packed into [0,1] in texture, unpacked to [0,2π] in shader
// θ kept in [0,2π) via mod() every step
// Emission: I = cos²(θ/2) — bright at θ=0, dark at θ=π
// Texture: .r=θ_packed, .g=ω_packed, .b=r_local, .a=emission
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uKU;
uniform vec2      uResolution;
uniform float     uK_kur;
uniform float     uOmega0;
uniform float     uSpread;
uniform float     uD_kur;
uniform float     uTime;

out vec4 fragColor;

const float PI     = 3.14159265358979;
const float TWO_PI = 6.28318530717959;
const float DT_KUR = 0.10;

// ─── FBM + Curl-Noise helpers (included per spec) ────────────────────────────
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

    vec4 center     = texture(uKU, uv);
    float th_packed = center.r;
    float om_packed = center.g;

    // Unpack: θ ∈ [0, 2π), ω ∈ [ω₀ ± 2·spread]
    float theta = th_packed * TWO_PI;
    float omega = (om_packed - 0.5) * 4.0 * uSpread + uOmega0;

    // 5×5 neighbourhood order parameter: r·exp(iΨ) = mean(exp(iθ))
    vec2 sum_sc = vec2(0.0);
    float count = 0.0;
    for (int di = -2; di <= 2; di++) {
        for (int dj = -2; dj <= 2; dj++) {
            vec2 nuv = wrap(uv + vec2(float(di), float(dj)) * texel);
            float nth = texture(uKU, nuv).r * TWO_PI;
            sum_sc += vec2(cos(nth), sin(nth));
            count  += 1.0;
        }
    }
    vec2  order = sum_sc / count;
    float r     = length(order);
    float Psi   = atan(order.y, order.x);

    // Phase diffusion Laplacian (central differences)
    float th_l = texture(uKU, wrap(uv + vec2(-texel.x,  0.0     ))).r * TWO_PI;
    float th_r = texture(uKU, wrap(uv + vec2( texel.x,  0.0     ))).r * TWO_PI;
    float th_u = texture(uKU, wrap(uv + vec2( 0.0,      texel.y ))).r * TWO_PI;
    float th_d = texture(uKU, wrap(uv + vec2( 0.0,     -texel.y ))).r * TWO_PI;
    float lap_th = th_l + th_r + th_u + th_d - 4.0 * theta;

    // Kuramoto mean-field update
    float dtheta = omega + uK_kur * r * sin(Psi - theta) + uD_kur * lap_th;
    float theta_next = mod(theta + DT_KUR * dtheta, TWO_PI);

    // Emission: I = cos²(θ/2) — peaks at θ=0, zero at θ=π
    float c = cos(theta_next * 0.5);
    float emission = c * c;

    fragColor = vec4(theta_next / TWO_PI, om_packed, r, emission);
}
