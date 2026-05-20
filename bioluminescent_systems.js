// Bioluminescent Systems — JS5 sketch for RepoScripter2
// Four mathematical pattern engines: Gray-Scott, FitzHugh-Nagumo,
// Kuramoto Phase Field, Fluid Advection-Diffusion
// Renders the visual language of deep-ocean bioluminescence:
// glowing coral lattices, traveling pulse rings, synchronized flash storms,
// light diffusing through turbulent water.
//
// Controls:
//   1–6         Gray-Scott regime (coral spawn → kelp forest)
//   Q W E R     FitzHugh-Nagumo mode (ring storm / spiral lock / pulse cascade / slow bloom)
//   A S D F     Kuramoto state (plankton chaos / bloom nucleation / shockwave sync / global pulse)
//   Space       Cycle active engine solo display
//   Shift       Pause / resume simulation
//   Scroll      Adjust Kuramoto coupling K
//   Mouse move  Disturb all active layers at cursor
//   Mouse down  Seed Gray-Scott + excite FitzHugh-Nagumo at cursor
//
// Part of the merrypranxter/ShaderForge ecosystem.
// Context source for: https://github.com/merrypranxter/reposcripter2

/* global THREE, canvas, width, height */

import { BioluminescentCompositor } from './src/engines/BioluminescentCompositor.js';

// ── Canvas setup ──────────────────────────────────────────────────────────────
// Create a dedicated WebGL2 canvas on top of the existing JS5 2D canvas
const glCanvas = document.createElement('canvas');
glCanvas.width  = window.innerWidth;
glCanvas.height = window.innerHeight;
Object.assign(glCanvas.style, {
    position:  'fixed',
    top:       '0',
    left:      '0',
    width:     '100vw',
    height:    '100vh',
    display:   'block',
    zIndex:    '1',
});
Object.assign(document.body.style, {
    margin:     '0',
    padding:    '0',
    overflow:   'hidden',
    background: '#000000',
});
document.body.appendChild(glCanvas);

// ── HUD overlay ───────────────────────────────────────────────────────────────
const hud = document.createElement('div');
Object.assign(hud.style, {
    position:      'fixed',
    top:           '12px',
    left:          '14px',
    color:         '#00b4d8',
    font:          '13px/1.7 "Courier New", monospace',
    pointerEvents: 'none',
    zIndex:        '10',
    textShadow:    '0 0 10px #00b4d8, 0 0 3px #48cae4',
    userSelect:    'none',
});
document.body.appendChild(hud);

// ── Compositor ────────────────────────────────────────────────────────────────
const compositor = new BioluminescentCompositor(glCanvas);
compositor.init();

// ── Simulation state ──────────────────────────────────────────────────────────
const GS_REGIMES = [
    'coral spawn', 'deep vein', 'mitosis',
    'void dissolution', 'full spectrum', 'kelp forest',
];
const FN_MODES   = ['ring storm', 'spiral lock', 'pulse cascade', 'slow bloom'];
const KU_STATES  = ['plankton chaos', 'bloom nucleation', 'shockwave sync', 'global pulse'];
const ENGINES    = ['all', 'gray-scott', 'fitzhugh-nagumo', 'kuramoto', 'advection'];

let currentGSIdx     = 0;   // coral spawn
let currentFNIdx     = 1;   // spiral lock (default)
let currentKUIdx     = 1;   // bloom nucleation (default)
let currentEngineIdx = 0;   // all
let paused           = false;
let isMouseDown      = false;

const mouse = { x: 0, y: 0 };

// ── Mouse handlers ────────────────────────────────────────────────────────────
glCanvas.addEventListener('mousemove', (e) => {
    const rect = glCanvas.getBoundingClientRect();
    // Convert to simulation pixel coords (512×512 space)
    const sx = (e.clientX - rect.left) / rect.width  * compositor.simSize;
    const sy = (e.clientY - rect.top)  / rect.height * compositor.simSize;
    mouse.x = sx;
    mouse.y = sy;

    // Disturb all active layers
    compositor.grayScott.disturb(sx, sy);
    compositor.fitzhughNagumo.disturb(sx, sy);
    compositor.advection.uniforms.uMouse.value.set(sx, sy);
});

glCanvas.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    const rect  = glCanvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) / rect.width  * compositor.simSize;
    const sy = (e.clientY - rect.top)  / rect.height * compositor.simSize;

    // Click → seed Gray-Scott
    compositor.grayScott.disturb(sx, sy);

    // Hold → excite FitzHugh-Nagumo
    compositor.fitzhughNagumo.disturb(sx, sy);
    compositor.fitzhughNagumo.mouseDown = true;

    // Add vortex to advection at click location
    const vortexStrength = (Math.random() > 0.5 ? 1 : -1) * 0.0003;
    compositor.advection.addVortex(sx, sy, vortexStrength);
});

glCanvas.addEventListener('mouseup', () => {
    isMouseDown = false;
    compositor.fitzhughNagumo.releaseDisturbance();
});

glCanvas.addEventListener('mouseleave', () => {
    isMouseDown = false;
    compositor.fitzhughNagumo.releaseDisturbance();
});

// Scroll: adjust Kuramoto coupling K
glCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    compositor.kuramotoField.adjustK(e.deltaY * 0.001);
    updateHUD();
}, { passive: false });

// ── Keyboard handlers ─────────────────────────────────────────────────────────
window.addEventListener('keydown', (e) => {
    const key = e.key.toUpperCase();

    // 1–6: Gray-Scott regimes
    if (key >= '1' && key <= '6') {
        currentGSIdx = parseInt(key) - 1;
        compositor.grayScott.setRegime(GS_REGIMES[currentGSIdx]);
        updateHUD();
        return;
    }

    switch (key) {
        // FitzHugh-Nagumo modes
        case 'Q': currentFNIdx = 0; compositor.fitzhughNagumo.setMode(FN_MODES[0]); break;
        case 'W': currentFNIdx = 1; compositor.fitzhughNagumo.setMode(FN_MODES[1]); break;
        case 'E': currentFNIdx = 2; compositor.fitzhughNagumo.setMode(FN_MODES[2]); break;
        case 'R': currentFNIdx = 3; compositor.fitzhughNagumo.setMode(FN_MODES[3]); break;

        // Kuramoto states
        case 'A': currentKUIdx = 0; compositor.kuramotoField.setState(KU_STATES[0]); break;
        case 'S': currentKUIdx = 1; compositor.kuramotoField.setState(KU_STATES[1]); break;
        case 'D': currentKUIdx = 2; compositor.kuramotoField.setState(KU_STATES[2]); break;
        case 'F': currentKUIdx = 3; compositor.kuramotoField.setState(KU_STATES[3]); break;

        // Space: cycle active engine solo display
        case ' ':
            e.preventDefault();
            currentEngineIdx = (currentEngineIdx + 1) % ENGINES.length;
            compositor.setActiveEngine(ENGINES[currentEngineIdx]);
            break;

        // Shift: pause / resume
        case 'SHIFT':
            paused = !paused;
            compositor.paused = paused;
            break;

        // C: clear vortices
        case 'C':
            compositor.advection.clearVortices();
            break;

        default: return;
    }
    updateHUD();
});

// ── HUD update ────────────────────────────────────────────────────────────────
function updateHUD() {
    const K   = compositor.kuramotoField.uniforms
        ? compositor.kuramotoField.uniforms.uK_kur.value.toFixed(2)
        : '—';
    const r   = compositor.getOrderParameter().toFixed(2);
    const eng = ENGINES[currentEngineIdx];

    hud.innerHTML = [
        `<b style="color:#90e0ef">BIOLUMINESCENT SYSTEMS</b>`,
        ``,
        `GS  [1–6] ${GS_REGIMES[currentGSIdx]}`,
        `FN  [Q–R] ${FN_MODES[currentFNIdx]}`,
        `KU  [A–F] ${KU_STATES[currentKUIdx]}`,
        ``,
        `K = ${K}  r = ${r}`,
        `engine: ${eng}`,
        paused ? `<span style="color:#a855f7">⏸ paused</span>` : '',
        ``,
        `[scroll] K  [space] cycle  [shift] pause`,
        `[click]  seed GS  [hold]  excite FN`,
    ].join('<br>');
}

// ── Resize handler ────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    glCanvas.width  = w;
    glCanvas.height = h;
    compositor.resize(w, h);
});

// ── Animation loop ────────────────────────────────────────────────────────────
let startTime = performance.now();

function animate() {
    requestAnimationFrame(animate);
    const elapsed = (performance.now() - startTime) * 0.001; // seconds

    compositor.update(elapsed, mouse);
    compositor.render();

    // Update HUD order parameter once per ~30 frames
    if (Math.round(elapsed * 30) % 30 === 0) {
        updateHUD();
    }
}

// ── Initialise and start ──────────────────────────────────────────────────────
updateHUD();
animate();
