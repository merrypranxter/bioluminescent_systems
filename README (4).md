# 🌊 Bioluminescent Systems

> *"The ocean remembers every disturbance as light."*

Four mathematical engines for generating bioluminescent pattern art — reaction-diffusion Turing structures, fluid light diffusion, excitable pulse waves, and phase-coupled synchronization. Not photorealistic sea life. The underlying mathematics that produces the same visual language: glowing coral lattices, traveling wave pulses, ink-spreading light in turbulent water, synchronized flash storms. Designed as a context source for AI-generated generative art.

[→ Use with RepoScripter2](https://github.com/merrypranxter/reposcripter2)

---

## The Bioluminescent Engines

| Engine | System | Visual Character |
|--------|--------|-----------------|
| **Gray-Scott** | Reaction-diffusion (Turing) | Coral polyp lattices, mitosis spots, labyrinthine stripes, branching dendrites — the fundamental pattern factory of bioluminescent marine organisms |
| **Fitzhugh-Nagumo** | Excitable media PDE | Traveling pulse rings, spiral waves, refractory dark zones — dinoflagellate bloom wavefronts, jellyfish bell pulse propagation |
| **Advection-Diffusion** | Light field in moving fluid | Glowing ink in turbulent water, vortex-stretched luminance filaments, billowing emission clouds dragged by curl-noise velocity fields |
| **Kuramoto Field** | Phase-coupled spatial oscillators | Synchronized flash storms — incoherent noise → coherent global pulse → shockwave synchronization fronts sweeping across the field |

---

## The Math

### Gray-Scott Reaction-Diffusion

Two chemical species — activator `u` and inhibitor `v` — diffusing and reacting on a 2D grid.

```
∂u/∂t = Dᵤ∇²u − uv² + F(1 − u)
∂v/∂t = Dᵥ∇²v + uv² − (F + k)v
```

```
u    = activator concentration ("food chemical"), range [0,1]
v    = inhibitor concentration ("bioluminescent pigment"), range [0,1]
Dᵤ   = activator diffusion rate (0.16–0.20) — always >> Dᵥ
Dᵥ   = inhibitor diffusion rate (0.05–0.08)
F    = feed rate — replenishment of u from boundary reservoir (0.010–0.082)
k    = kill rate — removal of v (0.040–0.073)
∇²   = discrete Laplacian (5-point stencil on GPU texture)
uv²  = autocatalytic reaction — v grows by consuming u, accelerating itself
```

**The key insight**: Dᵤ >> Dᵥ creates Turing instability — the activator diffuses fast enough to create long-range inhibition, locking patterns into stable spatial structures. Different `(F, k)` parameter pairs produce qualitatively different regimes: spots, stripes, labyrinthine mazes, moving solitons, or dead flat fields. The inhibitor field `v` maps to emission luminance.

**Named parameter regimes:**
```
"coral spawn"      F=0.037, k=0.060  →  dense hexagonal spot lattice
"deep vein"        F=0.022, k=0.051  →  branching labyrinthine stripes
"mitosis"          F=0.028, k=0.053  →  self-replicating spot colonies
"void dissolution" F=0.014, k=0.054  →  sparse drifting worms
"full spectrum"    F=0.050, k=0.065  →  chaotic coral-maze transition zone
```

---

### Fitzhugh-Nagumo Excitable Media

A reduced model of neural excitation applied to spatial tissue — every grid cell is an excitable "cell" that fires, propagates, and recovers.

```
∂v/∂t = D∇²v + v(v − α)(1 − v) − w + I_ext
∂w/∂t = ε(βv − γw − δ)
```

```
v      = membrane potential / local luminance (−0.5 to 1.5)
w      = recovery variable (refractory dark period)
D      = spatial diffusion coefficient (0.1–1.0) — controls pulse width
α      = threshold for excitation (0.10–0.30)
ε      = timescale ratio — how fast recovery lags excitation (0.005–0.05)
β      = recovery sensitivity (0.5)
γ      = recovery damping (1.0)
δ      = resting potential offset (0.0)
I_ext  = external stimulus field (point sources trigger pulse rings)
```

**What this produces**: Single excitation events propagate outward as ring waves. Colliding rings annihilate (no superposition). With spiral initial conditions, you get permanent rotating spirals — the exact pattern seen in dinoflagellate bloom wavefronts and jellyfish nervous tissue. The refractory period `w` controls the dark band behind each pulse.

**Named modes:**
```
"ring storm"       D=0.3, ε=0.02, α=0.1  →  many expanding rings, clean annihilation
"spiral lock"      D=0.5, ε=0.01, α=0.2  →  2–4 permanent counter-rotating spirals
"pulse cascade"    D=0.1, ε=0.04, α=0.15 →  rapid dense wavefronts, moiré-like interference
"slow bloom"       D=0.8, ε=0.005, α=0.1 →  wide slow pulses, long dark recovery zones
```

---

### Fluid Advection-Diffusion (Light in Moving Water)

The bioluminescent emission field `φ` is advected by a 2D velocity field and simultaneously diffuses.

```
∂φ/∂t + u⃗·∇φ = D∇²φ + S(x⃗, t)
```

```
φ(x⃗,t)  = emission intensity field (luminance), range [0,1]
u⃗        = 2D velocity field (see below)
D        = molecular diffusion of light emission (0.001–0.01)
S(x⃗,t)  = bioluminescent source term — driven by Gray-Scott v field or external input
∇φ       = spatial gradient (upwind differencing for stability)
∇²φ      = diffusion (central differencing)
```

**Velocity field construction** — two options:

Option A: Curl-noise (divergence-free by construction)
```
u⃗ = ∇⊥ψ = (∂ψ/∂y, −∂ψ/∂x)
ψ(x,t) = Σₙ Aₙ · noise(x·fₙ + t·sₙ)   ← layered octave Perlin noise
```

Option B: Explicit vortex sums
```
u⃗(x⃗) = Σᵢ (Γᵢ / 2π) · (x⃗ − x⃗ᵢ)⊥ / |x⃗ − x⃗ᵢ|²
Γᵢ     = circulation strength of vortex i (signed)
x⃗ᵢ    = vortex center position
```

**Visual behavior**: High-emission zones get stretched into filaments by shear flow. Low-D keeps filament structure sharp (glowing thread-like plankton trails). High-D blurs into volumetric clouds. Vortex-dominated fields spiral emission into glowing catherine wheels.

---

### Kuramoto Phase Field (Synchronized Flash Storms)

Every spatial point is an oscillator with its own natural frequency. Neighboring oscillators couple and pull toward synchrony.

```
∂θ/∂t = ω(x⃗) + K·r·sin(Ψ − θ) + D∇²θ
```

```
θ(x⃗,t)  = local oscillator phase, range [0, 2π)
ω(x⃗)    = natural frequency field — drawn from Lorentzian distribution
             ω ~ L(ω₀, Δ) → mean ω₀, spread Δ (wider = harder to sync)
K        = global mean-field coupling strength
r·e^iΨ  = order parameter: r = (1/N)·|Σ e^iθ| ∈ [0,1]
             r ≈ 0  → incoherent phase noise
             r ≈ 1  → global synchrony (every point flashes together)
D        = spatial diffusion of phase (smooths synchronization fronts)
```

**Emission function:**
```
I(x⃗,t) = cos²(θ(x⃗,t) / 2)    ← bright peak when θ = 0, dark at θ = π
```

**Phase transition**: At K = Kc = 2/π·Δ (critical coupling), the system jumps from incoherent noise to global synchrony. Just below Kc: local sync islands nucleate and compete. Just above Kc: synchronization fronts sweep across the field as traveling waves. The transition zone is visually catastrophic — the most interesting place to render.

**Named states:**
```
"plankton chaos"   K=0.1, Δ=0.8   →  incoherent — each cell twinkles independently
"bloom nucleation" K=0.8, Δ=0.5   →  sync islands form and drift
"shockwave sync"   K=1.5, Δ=0.3   →  synchronization front sweeps field in one pass
"global pulse"     K=3.0, Δ=0.1   →  entire field flashes together periodically
```

---

## GPU Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BIOLUMINESCENT SYSTEMS                │
│                   GPU PIPELINE (Three.js WebGL2)        │
└─────────────────────────────────────────────────────────┘

Simulation Layer (ping-pong FBOs — 512×512 RGBA32F each)
─────────────────────────────────────────────────────────

  GS_FBO_A ──→ [ Gray-Scott GLSL shader ] ──→ GS_FBO_B
  FN_FBO_A ──→ [ Fitzhugh-Nagumo shader ] ──→ FN_FBO_B
  KU_FBO_A ──→ [ Kuramoto phase shader  ] ──→ KU_FBO_B
  
  GS_FBO_B ──→ [ Advection-Diffusion   ] ──→ LIGHT_FBO
  (v channel feeds S source term)
  
  FBOs swap each frame. Each texture encodes:
    .r = u  (activator / potential / phase)
    .g = v  (inhibitor / recovery / frequency)
    .b = reserved / age / local order parameter
    .a = emission intensity (pre-composited)

Compositing Layer
─────────────────────────────────────────────────────────

  LIGHT_FBO ──┐
  GS_FBO_B ───┤
  FN_FBO_B ───┼──→ [ Mix/Blend shader ] ──→ EMIT_FBO
  KU_FBO_B ───┘
  (weighted sum — tunable per-engine opacity)

Post-Processing Layer
─────────────────────────────────────────────────────────

  EMIT_FBO ──→ [ Bloom (dual kawase blur) ] ──→ BLOOM_FBO
  EMIT_FBO ──→ [ Decay pass × 0.95       ] ──→ TRAIL_FBO
  BLOOM_FBO ──┐
  TRAIL_FBO ──┼──→ [ Tone-map + output ] ──→ screen
  EMIT_FBO ───┘

  gl.blendFunc(SRC_ALPHA, ONE) — additive blending
  Light accumulates at dense emission zones → blown-out hot cores
```

---

## Rendering Techniques

**Dual Kawase Bloom**
Downsample emission buffer through 4 mip levels, upsample back with bilinear filtering. Two passes per level — no Gaussian kernel required. Controls: `threshold` (what luminance triggers bloom), `radius` (spread), `intensity` (additive weight). At high intensity: pinpoint emissions become volumetric glowing orbs.

**Temporal Decay Trails**
Each frame: `trail_buffer × decay_factor + new_emission × emission_weight`. At `decay = 0.97`: faint ghost trails accumulate behind moving patterns, visualizing system history. At `decay = 0.80`: rapid fade, only current state visible. Trails are essential for Kuramoto — they reveal the arc of synchronization fronts after they pass.

**Additive Blending**
`gl.blendFunc(SRC_ALPHA, ONE)` — emission from every system layer accumulates. Overlapping active zones from multiple engines create super-bright intersection points. Sparse emission is faint teal; dense overlap burns to pure white. This is how real bioluminescent concentrations look in wave foam.

**Emission Color Mapping**
```
low emission   →  #001a33  (deep ocean black-blue)
dim pulse      →  #003d66  (abyssal navy)
medium glow    →  #00b4d8  (bioluminescent cyan)
peak intensity →  #90e0ef  (near-white electric teal)
bloom overflow →  #ffffff  (pure white at >1.0 HDR)
jellyfish pulse→  #7b2d8b  (violet — FitzHugh-Nagumo spirals)
firefly phase  →  #a8ff78  (acid green — Kuramoto sync moments)
```

**Curl-Noise Velocity Visualization** (optional overlay)
Render velocity field as semi-transparent directional glyphs (short line segments, opacity = magnitude). Shows where emission filaments are being stretched. Useful for debugging and beautiful on its own.

---

## Aesthetic Post-Processing

- **HDR tone mapping** — Reinhard or ACES filmic curve: preserves dark ocean background while clipping overdriven bloom to clean white
- **Chromatic aberration** (±2–4px RGB channel offset): simulates light scattering through water column — makes emission feel physically volumetric
- **Vignette** (radial darkening, 0.6 falloff): forces eye to center, amplifies sense of deep-sea isolation
- **Film grain** (0.03 amplitude, per-frame random seed): breaks up smooth GPU gradients — looks like biological noise, like real bioluminescence footage
- **Emission threshold mask**: hard cutoff below 0.02 emission → black, not dark grey. Keeps the background truly void.

---

## Files

| File | Contents |
|------|---------|
| `README.md` | This document |
| `repo_seed.txt` | Full mathematical context — all equations, parameter catalogs, aesthetic specs, AI generation prompts |
| `context.manifest.json` | Structured metadata for RepoScripter2 ingestion |
| `.github/agents/bioluminescence-expert.md` | Copilot custom agent — mathematical expert in all four systems, builds repo implementation |

---

## Used By

This repo is a context source for [RepoScripter2](https://github.com/merrypranxter/reposcripter2) — select it as input and generate new bioluminescent-pattern art with AI.

Also part of [ShaderForge](https://github.com/merrypranxter/shaderforge3) ecosystem.

---

*four mathematical systems. one ocean. every photon earned.*
