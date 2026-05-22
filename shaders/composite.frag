precision highp float;
varying vec2 vUv;

uniform sampler2D uGS;
uniform sampler2D uFN;
uniform sampler2D uKuramoto;
uniform sampler2D uAdvection;
uniform float uGSWeight;
uniform float uFNWeight;
uniform float uKuramotoWeight;
uniform float uAdvectionWeight;

void main() {
    float gsEmit   = texture2D(uGS,       vUv).a;
    float fnEmit   = texture2D(uFN,       vUv).a;
    float kuEmit   = texture2D(uKuramoto, vUv).a;
    float advEmit  = texture2D(uAdvection,vUv).a;

    // GS: deep navy → cyan
    vec3 gsBase  = vec3(0.0,   0.039, 0.078);
    vec3 gsPeak  = vec3(0.0,   0.706, 0.847);
    vec3 gsColor = mix(gsBase, gsPeak, gsEmit);

    // FN: black → violet layer + black → teal layer
    vec3 fnViolet = mix(vec3(0.0), vec3(0.482, 0.176, 0.545), fnEmit * 0.5);
    vec3 fnTeal   = mix(vec3(0.0), vec3(0.282, 0.792, 0.894), fnEmit);
    vec3 fnColor  = fnViolet + fnTeal;

    // Kuramoto: black → firefly green
    vec3 kuColor = mix(vec3(0.0), vec3(0.659, 1.0, 0.471), kuEmit);

    // Advection: black → white-core teal
    vec3 advColor = mix(vec3(0.0), vec3(0.565, 0.878, 0.937), advEmit);

    vec3 totalColor = gsColor  * uGSWeight
                    + fnColor  * uFNWeight
                    + kuColor  * uKuramotoWeight
                    + advColor * uAdvectionWeight;

    gl_FragColor = vec4(totalColor, 1.0);
}
