# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A **context source repo** for [RepoScripter2](https://github.com/merrypranxter/reposcripter2) — an AI shader generator. It provides mathematical specifications, parameter catalogs, and working implementations of four bioluminescent pattern engines. Code generated here runs in the **JS5 RepoScripter2 sandbox**, where `canvas`, `ctx`, `width`, `height`, `mouse`, `time`, and `THREE` are globals (no imports needed for Three.js).

There is no build system, test runner, or dev server. Correctness is validated by visual output in the sandbox.

---

## Intended File Structure

```
bioluminescent_systems/
├── bioluminescent_systems.js   ← main JS5 sketch entry point
├── shaders/
│   ├── gs_update.frag
│   ├── fn_update.frag
│   ├── kuramoto_update.frag
│   ├── advection.frag
│   ├── composite.frag
│   ├── bloom_down.frag
│   ├── bloom_up.frag
│   ├── decay.frag
│   └── output.frag
└── src/
    ├── GrayScott.js
    ├── FitzhughNagumo.js
    ├── KuramotoField.js
    ├── AdvectionDiffusion.js
    └── BioluminescentCompositor.js
```

---

## GPU Pipeline Architecture

All simulation state lives in **512×512 RGBA32F ping-pong FBO pairs** (one pair per engine). Each frame:

1. Gray-Scott update ×8 steps
2. Fitzhugh-Nagumo update ×50 steps
3. Kuramoto update ×1 step (recompute `r` and `Ψ` each step)
4. Advection-Diffusion update ×1 step (source term `S` = GS v-field)
5. Composite — additive blend all four emission layers → `EMIT_FBO`
6. Bloom downsample ×4 levels (dual Kawase)
7. Bloom upsample ×4 levels (dual Kawase)
8. Decay pass: `trail × 0.95 + emit × 0.05`
9. Output: tone map + grain + vignette + chromatic aberration → screen

Blending is **always** `gl.blendFunc(SRC_ALPHA, ONE)` (additive). No exceptions.

**Texture channel layout:**

| Texture | .r | .g | .b | .a |
|---|---|---|---|---|
| Gray-Scott | u (activator) | v (inhibitor/emission) | — | emission |
| Fitzhugh-Nagumo | v (potential) | w (recovery) | age | emission |
| Kuramoto | θ packed [0,1] | ω packed | r_local | emission I(θ) |
| Advection | φ (emission density) | u_vel | v_vel | source S |

---

## Simulation Equations and Parameters

### Gray-Scott
```
∂u/∂t = Du·∇²u − u·v² + F(1−u)
∂v/∂t = Dv·∇²v + u·v² − (F+k)v
Δt = 1.0,  clamp u,v to [0,1] after every step
∇² = 5-point stencil: neighbors sum − 4·center
Du/Dv ratio MUST be ≥ 2 or Turing instability won't form
```

Named regimes (use these exact values):
```
coral spawn:      Du=0.16, Dv=0.08, F=0.037, k=0.060  ← default
deep vein:        Du=0.16, Dv=0.08, F=0.022, k=0.051
mitosis:          Du=0.16, Dv=0.08, F=0.028, k=0.053
void dissolution: Du=0.19, Dv=0.05, F=0.014, k=0.054
full spectrum:    Du=0.16, Dv=0.08, F=0.050, k=0.065
kelp forest:      Du=0.16, Dv=0.08, F=0.029, k=0.057
```

### Fitzhugh-Nagumo
```
∂v/∂t = D·∇²v + v(v−α)(1−v) − w + I_ext
∂w/∂t = ε(β·v − γ·w − δ)
Δt = 0.01 MAX — the cubic term is stiff; larger timestep blows up
Standard: β=0.5, γ=1.0, δ=0.0
```

Named modes:
```
ring storm:    D=0.30, ε=0.020, α=0.10
spiral lock:   D=0.50, ε=0.010, α=0.20  ← default
pulse cascade: D=0.10, ε=0.040, α=0.15
slow bloom:    D=0.80, ε=0.005, α=0.10
```

### Kuramoto
```
∂θ/∂t = ω(x) + K·r·sin(Ψ−θ) + D·∇²θ
θ must be kept in [0, 2π) via mod every step — phase wrapping errors corrupt patterns
Emission: I = cos²(θ/2)
Order parameter: r·exp(iΨ) = mean of exp(iθ) over grid
Critical coupling: Kc = 2/π·Δ
```

Named states:
```
plankton chaos:    K=0.10, Δ=0.80, ω₀=1.0, D=0.05
bloom nucleation:  K=0.80, Δ=0.50, ω₀=1.0, D=0.10  ← default
shockwave sync:    K=1.50, Δ=0.30, ω₀=1.0, D=0.20
global pulse:      K=3.00, Δ=0.10, ω₀=1.0, D=0.50
```

### Advection-Diffusion
```
∂φ/∂t + u⃗·∇φ = D·∇²φ + S(x,t)
Use UPWIND differencing for the advection term — central differencing causes oscillations
Curl-noise velocity: u⃗ = ∇⊥ψ, ψ = layered Perlin fbm (divergence-free)
Source term S = GS v-field (default) or mouse injection or point bursts
```

---

## Post-Processing Stack

```
Bloom:     dual Kawase (not Gaussian), threshold=0.3, 4 levels, intensity=0.8
Decay:     trail_buffer × 0.95 + new_emission × 0.05
Tone map:  Reinhard (luminance-only, preserves hue)
Grain:     amplitude=0.03, per-frame random seed
Vignette:  1.0 − 0.6 × radial²
Chrom. ab: ±2px RGB channel offset at screen edges
```

---

## Color Palette (exact hex values — do not substitute)

```
abyss background: #000a14
plankton cyan:    #00b4d8  ← primary emission
electric teal:    #48cae4
white core:       #90e0ef
pure photon:      #ffffff  (bloom overflow only)
jellyfish violet: #7b2d8b  (FN refractory overlay)
spiral purple:    #a855f7  (FN spiral cores)
firefly green:    #a8ff78  (Kuramoto sync peak)
```

---

## Keyboard Controls (implement in main sketch)

```
1–6    Switch GS regime (coral spawn / deep vein / mitosis / void dissolution / full spectrum / kelp forest)
Q–T    Switch FN mode (ring storm / spiral lock / pulse cascade / slow bloom)
Z–.    Switch Kuramoto state (plankton chaos / bloom nucleation / shockwave sync / global pulse)
Space  Pause simulation
```

Mouse: click injects emission + triggers FN excitation at cursor. Drag creates a temporary vortex in the advection layer.

---

## Critical Constraints — Never Violate

- **GS clamping**: u and v must be clamped to `[0,1]` after every simulation step. Negative values corrupt the autocatalytic `uv²` term.
- **FN timestep**: Δt ≤ 0.01. The cubic `v(v−α)(1−v)` is stiff — larger steps diverge to NaN.
- **Kuramoto phase wrap**: `mod(theta, TWO_PI)` every step. Un-wrapped phase causes incorrect `sin(Ψ−θ)` coupling.
- **Advection differencing**: upwind scheme only for the `u⃗·∇φ` term. Central differences cause unphysical oscillations.
- **Turing condition**: Du/Dv ≥ 2 always. Equal diffusion rates kill all pattern formation.
- **Blend mode**: additive `(SRC_ALPHA, ONE)` throughout. Any other mode breaks the emission accumulation model.
- **No pseudocode shaders**: every GLSL file must compile and run. No `// TODO` bodies.

---

## Ecosystem

Part of the `merrypranxter/ShaderForge` ecosystem.
Context source for: https://github.com/merrypranxter/reposcripter2
