## Project Context

**RESUM_FLEX** is a modular refactor of the original RESuM framework (located at `/home/yuema137/resum`, reference paper: [openreview lqTILjL6lP](https://openreview.net/pdf?id=lqTILjL6lP)). RESuM solves the **Rare Event Design (RED)** problem in physics simulations (e.g. NLDBD experiments) where the design metric `y = m/N` is a discrete count with high variance.

The pipeline has three stages:
1. **CNP** denoises discrete counts by reconstructing the underlying Bernoulli parameter `p` (continuous score `β ∈ [0,1]`).
2. **MFGP** (Multi-Fidelity GP, via Emukit + GPy) fuses low-fidelity CNP scores with high-fidelity raw counts to produce a posterior over `y`.
3. **Active Learning** (Integrated Variance Reduction) selects the next design `θ_next`.

### Two kinds of inputs (core architectural distinction)
- `θ` — **design parameters**: global, static per simulation run.
- `φ` — **event parameters**: local stochasticity, unique per event.
- `X` — binary event labels (Bernoulli draws).

Three input modalities must be supported: `FULL (θ,φ,X)`, `EVENT_ONLY (φ,X)`, `DESIGN_ONLY (θ,X)`. Missing components are handled via a **Universal Encoder with learnable null embeddings** (`theta_null`, `phi_null`) — not by branching code paths.

### Module layout & contracts
| File | Role | Key I/O |
|---|---|---|
| `core/data_engine.py` | Standardize raw sim files → `StandardBatch` | `theta:[B,Dθ]?`, `phi:[B,N,Dφ]?`, `labels:[B,N,1]`, `masks` |
| `core/networks.py` | Dual-Latent Encoder w/ null handling | → `z_θ:[B,Z]`, `z_φ:[B,N,Z]` |
| `core/surrogate_cnp.py` | Event-level CNP, reconstructs `p` | → `BetaScore:[B,N,1]` |
| `core/surrogate_mfgp.py` | Design-level MFGP (Emukit/GPy) | → `(μ_θ, σ²_θ)` |
| `core/optimizer.py` | IVR active learning loop | → `NextTheta` |
| `schemas/data_models.py` | pydantic schemas: `DesignPoint`, `EventBatch`, `StandardBatch`, `ModelPrediction` | — |

### Hard decoupling rules
- `core/networks.py` (PyTorch) MUST NOT import `core/surrogate_mfgp.py` (GPy/Emukit). The two stacks stay separate.
- The encoder must be swappable (MLP ↔ Transformer) without changes to `data_engine`.
- All hyperparameters (layer sizes, kernel types, mixup α, etc.) live in `config.yaml`, loaded via pydantic.
- `StandardBatch.theta` and `StandardBatch.phi` are `Optional`; presence is communicated via boolean masks.

### Status
First task is to define `schemas/data_models.py`. No code exists yet beyond this CLAUDE.md.

## Validation Matrix (Combinatorial Dimension Test)

The refactor must pass an 8-scenario test grid that combines input mode × θ-dim × φ-dim. Every module's I/O contract is verified against this grid; visual outputs are part of acceptance, not optional.

| ID | Mode | dim(θ) | dim(φ) | Visualization required |
|---|---|---|---|---|
| S1 | FULL | 1 | 1 | 2D heatmap (x=θ, y=φ) |
| S2 | FULL | 2 | 1 | 3D slices or multi-subplot |
| S3 | FULL | 1 | 2 | 3D slices or multi-subplot |
| S4 | FULL | 2 | 2 | dim-reduced projection plots |
| S5 | EVENT_ONLY | — | 1 | 1D curve |
| S6 | EVENT_ONLY | — | 2 | 2D heatmap |
| S7 | DESIGN_ONLY | 1 | — | 1D curve |
| S8 | DESIGN_ONLY | 2 | — | 2D heatmap |

### Plot dispatch rule
- `dim == 1` → line plot with mean ± variance band.
- `dim == 2` → contour plot or heatmap.
- `dim ≥ 3` → projection / sliced subplots.

This rule lives in a single utility (`viz/dispatch.py` or similar) — modules call it; they do not branch on dim themselves.

### Acceptance criterion (per scenario)
`MAE(predicted_β, ground_truth_p) < threshold` on held-out points from the pseudo-data generator. Threshold lives in `config.yaml` per scenario (looser tolerance for higher-dim cases is expected).

## Phased Implementation Plan

Phases run in order. Each phase has a hard acceptance gate before the next begins.

### Phase 0 — Schemas (`schemas/data_models.py`)
- Define `DesignPoint`, `EventBatch`, `StandardBatch`, `ModelPrediction` as pydantic models.
- `StandardBatch.theta` and `StandardBatch.phi` are `Optional`; presence carried by mask flags.
- Tensor shape validators allow arbitrary `dim(θ)` and `dim(φ)`.
- **Gate:** instantiate empty/null/full `StandardBatch` for all 8 scenarios without errors.

### Phase 1 — Pseudo-Data Generator (`data/pseudo_generator.py`)
- `PseudoDataGenerator.generate(mode, dim_theta, dim_phi, n_trials, n_events) → StandardBatch`.
- Internally defines a known ground-truth `t(θ,φ)` (e.g. a smooth function with a localized peak); samples `X ~ Bernoulli(t(θ,φ))`.
- Returns both the `StandardBatch` and the analytical `t(θ,φ)` ground truth (for later validation).
- **Gate:** for all 8 scenarios, plot ground-truth `p` using the dim-dispatch rule; verify `X` is a Bernoulli noisy realization of `p`.

### Phase 2 — Universal Encoder (`core/networks.py`)
- Implements learnable `theta_null` and `phi_null` parameters.
- Forward pass returns `z_θ:[B,Z]` and `z_φ:[B,N,Z]` regardless of which inputs were `None`.
- **Gate (Dimension Test):** print output shapes for S1–S8; all `[B,Z]` / `[B,N,Z]` aligned. **Gate (Null Test):** `theta=None` and `phi=None` paths execute without error and produce non-NaN tensors.

### Phase 3 — CNP (`core/surrogate_cnp.py`)
- Train CNP on pseudo-data; recover continuous `β ≈ p`.
- **Gate (1D):** for 1D θ or φ, regression curve must pass through the dense center of the binary `X` cloud.
- **Gate (2D):** for 2D inputs, predicted `β` heatmap "peaks" must align with ground-truth `p` heatmap peaks.
- **Quantitative gate:** MAE(β, p) below per-scenario threshold from config.

### Phase 4 — MFGP (`core/surrogate_mfgp.py`)
- Co-kriging across `y_CNP^LF`, `y_CNP^HF`, `y_Raw^HF`, parameterized over arbitrary `dim(θ)`.
- **Gate (1D θ):** plot posterior mean with shaded confidence band.
- **Gate (2D θ):** plot 3D response surface.
- **Gate (coverage):** on held-out HF samples, ±1σ/±2σ/±3σ coverage approaches 68/95/99.7%.

### Phase 5 — IVR Optimizer (`core/optimizer.py`)
Final phase, only after MFGP gate passes. Out of scope until Phase 4 is green.

## Math & Concepts (load-bearing for implementation)

These are the formulas that pin down loss functions, output shapes, and validation criteria. Anchors against drift.

### RED problem setup
- Per event `i` in trial `k`: `X_ki ∈ {0,1}` is a Bernoulli draw with `p = t(θ_k, φ_ki)`.
- Trial-level count: `m = Σ_i X_i`. Raw design metric: `y_Raw = m/N`.
- True objective (what we actually want to minimize): the **marginalized triggering rate**
  `t̄(θ) = ∫ t(θ, φ) g(φ) dφ`, where `g(φ)` is the event-parameter distribution.
- Optimization target: `θ* = argmin_{θ ∈ Θ} t̄(θ)`.
- Rare-event regime: `m ≪ N`, so `m ~ Poisson(N·t̄(θ))`. At small `N`, `y` lives on a discrete grid `{0/N, 1/N, …}` with high variance — this is *why* a surrogate is needed.

### CNP (event-level)
- Replaces binary `X_ki` with a continuous score `β_ki ≈ t(θ_k, φ_ki) ∈ [0,1]`.
- Bayesian view: CNP is a VAE-like estimator of the latent function `t(θ,φ)`. The "decoder" is the *predefined* Bernoulli; the "encoder" `q_NN` is what we train.
  `q_NN(t(θ,φ)) = N(μ_NN(θ,φ;w), σ²_NN(θ,φ;w))` conditioned on `{X_ki, φ_ki, θ_k}`.
- **Training loss** = negative log-likelihood of observed binary data under the Bernoulli model marginalized over `q_NN` (Eq. 11 in paper):
  `L = Π_k Π_i ∫ Bernoulli(X_ki | p = t(θ_k,φ_ki)) · q_NN(t) dt`
  Practically: sample `β` from `q_NN`, compute Bernoulli NLL on `X`. **Do NOT train CNP to predict `X` directly via BCE — train it to estimate `p` via the marginalized likelihood.**
- Architecture: `encoder (MLP) → mean-aggregator → decoder`. Context point = concat(θ, φ_i, X_i). β output bounded to `[0,1]` (sigmoid or equivalent).
- `β_ki` is **fidelity-invariant** by construction — it depends only on `(θ, φ)`, so the same CNP applies to LF and HF events.

### Aggregated CNP score (trial-level)
Per simulation trial:
`y_CNP(θ_k) = (1/N) Σ_{i=1..N} β_ki`

### MFGP (trial-level, three fidelities)
Three input metrics into the GP, all keyed by `θ`:
1. `y_CNP^LF` — averaged β over LF events (cheap, broad coverage).
2. `y_CNP^HF` — averaged β over HF events.
3. `y_Raw^HF` — `m/N` from HF (the ultimate target).

**Co-kriging recursion** (Kennedy–O'Hagan):
`f_H(θ) = ρ · f_L(θ) + δ(θ)`, with `δ ~ GP` and scalar `ρ`. Joint covariance:
`Σ = [[K_LL,        ρ·K_LL          ],
      [ρ·K_LL,      ρ²·K_LL + K_δ   ]]`
Recurses for >2 levels: `f_{H_i}(θ) = ρ_i · f_{L_i}(θ) + δ_i(θ)`.

GP posterior at `θ*`:
`μ* = K(θ*, Θ) K(Θ,Θ)⁻¹ y`
`σ²* = K(θ*, θ*) − K(θ*, Θ) K(Θ,Θ)⁻¹ K(Θ, θ*)`

### Active Learning — Integrated Variance Reduction (IVR)
Pick next design point by minimizing expected total posterior variance:
`I(θ_new) = ∫_Θ σ²(θ | θ_new) dθ`,  `θ_new = argmin_θ I(θ)`.

Tractable approximation (RBF kernel, MC sample of `θ_i`):
`I(θ) ≈ (1/N) Σ_i k²(θ_i, θ) / σ²(θ)`.
Constraints on Θ are enforced via penalty terms that drive the acquisition to zero in infeasible regions.

### Class imbalance & mixup
- Signal:background ratio in raw events is ~`1 : 5·10⁴`. Without augmentation, CNP collapses.
- **Mixup** augmentation: draw `λ ~ Beta(α, α)` with `α = 0.1` (paper value):
  `x̂ = λ·x_i + (1−λ)·x_j`,  `ŷ = λ·y_i + (1−λ)·y_j`.
  `α` lives in config.

### Validation / coverage criterion
A trained MFGP is "good" if its predictive band covers held-out HF samples at standard-normal rates:
- 1σ → ≈68.27% (paper achieved 69%)
- 2σ → ≈95.45% (paper achieved 95%)
- 3σ → ≈99.73% (paper achieved 100%)

This is the project's gold-standard test. The ablation without `y_CNP` got 12% / 24% / 47% — **dropping the CNP scores breaks the model**, this is the empirical justification for the three-fidelity design.

### Implementation gotchas these formulas imply
- CNP output activation must keep `β ∈ [0,1]` — `μ_NN` should be passed through sigmoid (or use a bounded distribution).
- CNP loss is **not** BCE on `X`; it's Bernoulli NLL with `p` sampled from a Gaussian over `β`. Test this on synthetic data where ground-truth `t(θ,φ)` is known.
- Aggregator is mean over the event axis; don't accidentally reduce over batch.
- MFGP must accept `(θ_k, y_CNP^LF_k)`, `(θ_k, y_CNP^HF_k)`, `(θ_k, y_Raw^HF_k)` as three separate fidelity datasets, not stacked — Emukit's API handles this explicitly.
- IVR acquisition needs to evaluate the GP posterior fast and many times — keep the GP backend (GPy) hot, don't re-instantiate per call.

## Tooling

- **Configuration**: `pydantic` + `pydantic-yaml`. Every config object is a `pydantic.BaseModel`; YAML files load into and validate against these models.
- **Testing**: `pytest`. Each module should have a corresponding test file. Aim to test pure logic without requiring real HDF5 access where possible (use small synthetic fixtures).
- **Lint + format**: `ruff` (both `ruff check` and `ruff format`).
- **Type checking**: not enforced for now; revisit if the project grows.
- **Experiment tracking**: TBD. Decide before the first real training run.

## Coding Standards

- **Logic First**: Before every modification, review the current structure of the whole project and consider whether the structure itself is appropriate, rather than just bolting on the desired feature. Keep the code clean and elegant.
- **Slow is Smooth, Smooth is Fast**: Never be greedy when adding a feature or refactoring. Fix the bug first, then improve structure. Focus on the current problem at each step; do not over-optimize.
- **Clear docstrings and comments**: write correct types for inputs and outputs. `pydantic` validation and informative error messages are strongly encouraged for every function and class.
- **Avoid deep coupling between modules**: each module should be testable in isolation, pluggable, and decoupled.
- **Always think about what test we can add for each single module**: pytest is powerful — use it.
- **Be humble and curious**: if you are unsure about something — feature details, data format, intent — do not guess. Ask the user explicitly.
- **Be strict with the user and double-check**: what the user says is not always correct. If a statement seems wrong or an idea seems impractical, ask for clarification and state the objection clearly.
- **Never directly continue right after conversation compression**: stop after compression. The user will re-supply the context, docs, and code to read. Do not start blindly.
