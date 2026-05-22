precision highp float;
varying vec2 vUv;

uniform sampler2D uTrailTexture;
uniform sampler2D uEmitTexture;
uniform float uDecay;
uniform float uEmitWeight;

void main() {
    vec3 trail = texture2D(uTrailTexture, vUv).rgb;
    vec3 emit  = texture2D(uEmitTexture,  vUv).rgb;

    vec3 result = trail * uDecay + emit * uEmitWeight;

    gl_FragColor = vec4(result, 1.0);
}
