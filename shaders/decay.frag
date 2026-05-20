#version 300 es
// Temporal Decay / Trail Accumulation
// new_trail = trail × uDecay + current × (1 − uDecay)
// At uDecay=0.95: strong persistent trails (system history visible)
// At uDecay=0.80: fast fade (only current state visible)
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

precision highp float;

uniform sampler2D uTrail;
uniform sampler2D uCurrent;
uniform float     uDecay;

out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uTrail, 0));

    vec4 trail   = texture(uTrail,   uv);
    vec4 current = texture(uCurrent, uv);

    // Temporal accumulation: preserves motion history
    fragColor = trail * uDecay + current * (1.0 - uDecay);
}
