# RESUM_FLEX

A modular refactor of **RESuM** (Rare Event Surrogate Model) — a physics-ML
pipeline for **Rare Event Design (RED)** problems where the design metric
`y = m/N` is a discrete count with high variance. The framework denoises
binary observations with a Conditional Neural Process (CNP), then fuses
multi-fidelity scores with a Multi-Fidelity Gaussian Process (MFGP), and
selects new design points by Integrated Variance Reduction (IVR) or
Expected Improvement (EI).

Reference paper: [RESuM: A Rare Event Surrogate Model](https://openreview.net/pdf?id=lqTILjL6lP).

## Status

| Phase | What | State |
|---|---|---|
| 0 | Pydantic schemas (`StandardBatch`, `ModelPrediction`, `Config`) | ✅ |
| 1 | Pseudo-data generator with analytical `t(θ, φ)` + 8-scenario plots | ✅ |
| 2 | Universal encoder with learnable null embeddings (PyTorch) | ✅ |
| 3 | CNP training, checkpoints, reconstruction & coverage | ✅ |
| 4 | MFGP co-kriging via Emukit/GPy + held-out coverage gate | ✅ |
| 5 | Active learning — IVR (exploration) + EI (exploitation) | ✅ |

158 tests pass (~1 minute on CPU). See `CLAUDE.md` for the full
architecture brief and math reference.

## Install

```bash
git clone <this repo>
cd RESUM_FLEX

uv venv .venv --python 3.12        # or `python -m venv .venv`
source .venv/bin/activate

uv pip install -e ".[gp,dev]"      # core + GPy/Emukit + dev tools
# Or, without uv:
# pip install -e ".[gp,dev]"
```

PyTorch is a hard dependency. Other extras:

* `gp` — `GPy>=1.10`, `emukit>=0.4.10` (required for MFGP / Phase 4 & 5).
* `dev` — `pytest`, `ruff`.

For a CPU-only torch build:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Layout

```
RESUM_FLEX/
├── schemas/           pydantic data contracts (numpy-only, no torch / GPy)
│   ├── data_models.py     StandardBatch, ModelPrediction (+ DesignPoint, EventBatch)
│   └── config.py          Typed YAML config tree (load_config)
├── core/              compute modules
│   ├── networks.py        Universal encoder + null embeddings (PyTorch)
│   ├── surrogate_cnp.py   CNP forward, Bernoulli-NLL loss, ctx/target split
│   ├── surrogate_mfgp.py  3-fidelity recursive co-kriging via Emukit/GPy
│   ├── mfgp_pipeline.py   CNP → MFGP bridge: prepare datasets, fit, coverage eval
│   ├── optimizer.py       IVR + EI acquisitions, active-learning loop
│   └── training.py        train_cnp, evaluate_mae, cnp_trial_predictive, checkpoints
├── data/              synthetic data
│   └── pseudo_generator.py    GaussianBumpTruth + PseudoDataGenerator + for_scenario()
├── viz/               plotting primitives
│   └── dispatch.py        plot_field, plot_comparison_1d/2d, plot_coverage_test
├── scripts/           runnable end-to-end demonstrations on synthetic data
│   ├── phase{1..5}_*.py
├── tests/             pytest (no real-data fixtures; everything synthetic)
├── config.yaml        canonical hyperparameter file (+ pydantic-validated)
├── pyproject.toml     deps + ruff/pytest config
└── CLAUDE.md          authoritative agent brief; deep architecture & math
```

**Decoupling rule:** PyTorch lives only in `core/networks.py`,
`core/surrogate_cnp.py`, `core/training.py`, and `core/optimizer.py`. The
schema layer is numpy-native; `core/surrogate_mfgp.py` (GPy/Emukit) consumes
those arrays directly. `core/mfgp_pipeline.py` is the only file where
torch and GPy coexist — and only at the boundary (CNP → numpy → GP).

---

## I/O schemas

Everything the package consumes or emits is one of these typed objects.
All array fields are `numpy.ndarray`.

### Input — `StandardBatch`

The primary pipeline carrier. Three modalities are supported via the `mode` flag:

| `mode` (`InputMode`) | `theta` | `phi` | `labels` (X) |
|---|---|---|---|
| `InputMode.FULL` | `[B, D_θ]` | `[B, N, D_φ]` | `[B, N]` binary {0,1} |
| `InputMode.EVENT_ONLY` | `None` | `[B, N, D_φ]` | `[B, N]` binary |
| `InputMode.DESIGN_ONLY` | `[B, D_θ]` | `None` | `[B, N]` binary |

`B` = number of trials; `N` = events per trial; `D_θ`, `D_φ` ≥ 1 (arbitrary).
Optional field `beta: [B, N] ∈ [0, 1]` is filled by the CNP.

```python
from schemas.data_models import StandardBatch, InputMode
batch = StandardBatch(
    mode=InputMode.FULL,
    theta=theta_arr,        # (B, D_θ)
    phi=phi_arr,            # (B, N, D_φ)
    labels=labels_arr,      # (B, N), values in {0, 1}
)
```

The validator cross-checks `mode` against the presence of `theta` / `phi`
and verifies all batch dims agree, so malformed batches fail at construction.

### Output — `ModelPrediction`

What MFGP-style models return at query points:

| Field | Shape | Meaning |
|---|---|---|
| `mean` | `[B]` | Posterior mean μ(θ) |
| `variance` | `[B]` | Posterior variance σ²(θ) (≥ 0) |
| `theta_query` | `[B, D_θ]` | The θ values queried |

Available via `MultiFidelityGP.predict_as_model_prediction(X)`.

### Output — `cnp_trial_predictive` dict

Per-trial predictive distribution from a trained CNP. All values are 1-D arrays of length `B` (number of trials):

| Key | Meaning |
|---|---|
| `y_cnp` | Posterior mean of `y = m/N` on the trial |
| `sigma_total` | √(σ²_epistemic + σ²_aleatoric) |
| `sigma_epistemic` | CNP-decoder uncertainty (model knowledge) |
| `sigma_aleatoric` | √(p(1-p)/N) — irreducible Bernoulli noise |

### Output — MFGP datasets dict

Returned by `prepare_mfgp_datasets_from_batches`. Each entry is `(n_trials, k)` numpy:

| Key | Shape | Meaning |
|---|---|---|
| `X_lf` | `(n_lf, D_θ)` | Per-trial θ for the LF dataset |
| `Y_lf_cnp` | `(n_lf, 1)` | β̄(θ) per LF trial (CNP-aggregated) |
| `X_hf` | `(n_hf, D_θ)` | Per-trial θ for the HF dataset |
| `Y_hf_cnp` | `(n_hf, 1)` | β̄(θ) per HF trial |
| `Y_hf_raw` | `(n_hf, 1)` | `m/N` per HF trial — the GP's target signal |

### Output — coverage dict

Returned by `evaluate_mfgp_coverage_from_batch`:

| Key | Shape / Type | Meaning |
|---|---|---|
| `theta` | `(n_test, D_θ)` | Held-out θ |
| `y_obs` | `(n_test,)` | Observed `m/N` per trial |
| `mu` | `(n_test,)` | MFGP posterior mean |
| `sigma` | `(n_test,)` | MFGP posterior std |
| `1sigma` / `2sigma` / `3sigma` | float | Fraction inside ±kσ band |

Target Gaussian rates: 68.27 / 95.45 / 99.73 %.

---

## Configuration

All hyperparameters live in **`config.yaml`**, validated against the pydantic
models in `schemas/config.py`. Two equivalent ways to use them:

### Option A — load from YAML

```python
from schemas.config import load_config

cfg = load_config("config.yaml")
print(cfg.cnp.n_context_min, cfg.training.learning_rate)
```

The full default `config.yaml`:

```yaml
seed: 42

encoder:                          # MLP encoder (Phase 2)
  type: mlp
  latent_dim: 64
  hidden_dims: [128, 128]
  dropout: 0.0

cnp:                              # CNP (Phase 3)
  n_context_min: 16
  n_context_max: 64
  output_activation: sigmoid
  mixup_alpha: 0.1

mfgp:                             # MFGP (Phase 4)
  kernel: rbf                     # 'rbf' or 'matern52'
  n_fidelities: 3

ivr:                              # IVR optimizer (Phase 5)
  n_mc_samples: 1000

training:                         # CNP training loop (Phase 3)
  n_steps: 1500
  learning_rate: 1.0e-3
  batch_size: 16
  n_events_per_trial: 128
  n_mc_samples: 4
  grad_clip: 1.0
  eval_every: 200
  eval_batch_size: 32
  eval_n_events: 256
  seed: 0

mae_thresholds:                   # Phase 3 acceptance gate per scenario
  s1: 0.05
  s2: 0.08
  s3: 0.08
  s4: 0.12
  s5: 0.05
  s6: 0.08
  s7: 0.05
  s8: 0.08
```

### Option B — build configs in code

```python
from schemas.config import EncoderConfig, CNPConfig, TrainingConfig

enc_cfg = EncoderConfig(type="mlp", latent_dim=64, hidden_dims=[128, 128], dropout=0.0)
cnp_cfg = CNPConfig(n_context_min=16, n_context_max=64,
                    output_activation="sigmoid", mixup_alpha=0.1)
train_cfg = TrainingConfig(n_steps=1500, learning_rate=1.0e-3, batch_size=16,
                           n_events_per_trial=128, n_mc_samples=4, seed=0)
```

### Override a single field

Pydantic models are immutable but support `model_copy(update=...)`:

```python
cfg = load_config("config.yaml")
custom_cnp = cfg.cnp.model_copy(update={"n_context_min": 64, "n_context_max": 256})
custom_train = cfg.training.model_copy(update={"n_steps": 3000, "learning_rate": 5e-4})
```

The downstream API takes typed configs directly — no string-keyed dicts:
`train_cnp(cnp, generator, cnp_config=custom_cnp, training_config=custom_train)`.

---

## Usage

The user-facing flow is the **bring-your-own-data** path. The synthetic
generator (`for_scenario("S1"..."S8")`) is for validation / unit tests.

### 1. Prepare your data → `StandardBatch`

The package consumes `StandardBatch` objects. Load your simulation data
(HDF5, npz, CSV, ROOT, …) into numpy arrays, then instantiate:

```python
import numpy as np
from schemas.data_models import StandardBatch, InputMode

# Example: load from your file format
arrays = np.load("my_lf_simulation.npz")
theta_lf  = arrays["theta"]      # shape (n_trials, D_θ)
phi_lf    = arrays["phi"]        # shape (n_trials, n_events_per_trial, D_φ)
labels_lf = arrays["X"]          # shape (n_trials, n_events_per_trial), {0, 1}

lf_batch = StandardBatch(
    mode=InputMode.FULL,
    theta=theta_lf,
    phi=phi_lf,
    labels=labels_lf.astype(np.int8),
)
```

For event-only or design-only modalities, set the absent component to
`None` and use the matching `InputMode`:

```python
event_only = StandardBatch(mode=InputMode.EVENT_ONLY, theta=None, phi=phi, labels=X)
design_only = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, phi=None, labels=X)
```

The validator throws on shape / mode mismatches at construction time.

#### Normalize before you build the batch (important)

The CNP encoder is an MLP that fits the **raw numerical input**. When
`θ` (or `φ`) components have ranges that differ by ≥ ~10× — *whatever
the units* — gradient imbalance makes the encoder *scale-blind*: it
locks onto the high-magnitude dimension and ignores the small one
(often by a factor of 100× in effective gradient flow, even after
long training). The MFGP can compensate via ARD lengthscales; the
CNP cannot.

**Always normalize before building the batch when feature ranges
differ.** Use `core.MinMaxScaler` and persist it alongside your CNP /
MFGP checkpoint so predictions can be inverse-transformed back to the
original units later. **Any numerical ranges are supported** — the
scaler simply maps each feature's `[low, high]` linearly onto
`[-1, 1]` (or any target interval you choose).

Three equivalent ways to construct it:

```python
from core import MinMaxScaler

# 1. Preferred — known per-feature bounds (any numerical ranges).
theta_scaler = MinMaxScaler.from_bounds(
    low=theta_low,             # 1-D array of length D_θ
    high=theta_high,            # 1-D array of length D_θ
)

# 2. Fit from data when bounds aren't known a priori.
theta_scaler = MinMaxScaler.fit(theta_train)

# 3. Pick a different output interval (default is [-1, 1]).
theta_scaler = MinMaxScaler.from_bounds(
    low=theta_low, high=theta_high,
    target_low=0.0, target_high=1.0,
)
```

Concrete examples — the same call works for any ranges:

```python
# Physical units that span vastly different magnitudes:
sc = MinMaxScaler.from_bounds(low=[500.0, 0.0],   high=[3000.0, 1.0])

# Negative ranges, mixed scales:
sc = MinMaxScaler.from_bounds(low=[-50, 1e-6],    high=[50, 1e3])

# 4-D design space:
sc = MinMaxScaler.from_bounds(low=[0, 0, 0, -π],  high=[10, 100, 1, π])
```

Apply, build the batch, train, predict, invert:

```python
theta_scaled = theta_scaler.transform(theta_raw)            # forward
lf_batch = StandardBatch(mode=InputMode.FULL,
                          theta=theta_scaled,
                          phi=phi_lf,                        # use a separate scaler
                          labels=labels_lf.astype(np.int8))   #   for φ if needed

# At predict time, scale the query with the *same* scaler:
mu, var = mfgp.predict(theta_scaler.transform(theta_query_raw))

# Inverse-transform any θ-shaped output (e.g. an AL record) back to original units:
theta_next_raw = theta_scaler.inverse_transform(record.theta_next.reshape(1, -1))[0]
```

If you skip normalization on imbalanced inputs, the framework emits a
`ScaleImbalanceWarning` at `StandardBatch` construction. The threshold
is a 10× per-feature range gap — see `schemas.data_models.SCALE_IMBALANCE_THRESHOLD`.

### 2. Train the CNP

`train_cnp` consumes a duck-typed *batch generator* — anything with the
shape

```python
class HasGenerate:
    mode: InputMode
    dim_theta: int | None
    dim_phi:   int | None
    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch: ...
```

For synthetic data this is `PseudoDataGenerator`. For your own fixed
dataset, write a small re-sampling adapter:

```python
import numpy as np
from schemas.data_models import StandardBatch, InputMode

class BatchSampler:
    """Resample sub-batches from a fixed StandardBatch for CNP training."""
    def __init__(self, full: StandardBatch) -> None:
        self.batch = full
        self.mode = full.mode
        self.dim_theta = full.theta.shape[1] if full.theta is not None else None
        self.dim_phi   = full.phi.shape[2]   if full.phi   is not None else None

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch:
        rng = np.random.default_rng(seed)
        ti  = rng.choice(self.batch.batch_size, size=n_trials, replace=True)
        ei  = rng.choice(self.batch.n_events,   size=n_events, replace=False)
        return StandardBatch(
            mode=self.batch.mode,
            theta=self.batch.theta[ti] if self.batch.theta is not None else None,
            phi=self.batch.phi[ti][:, ei] if self.batch.phi is not None else None,
            labels=self.batch.labels[ti][:, ei],
        )

sampler = BatchSampler(full_lf_batch)   # built from your raw arrays in §1
```

Then:

```python
import torch
from core import build_cnp, train_cnp, save_checkpoint
from schemas.config import EncoderConfig, CNPConfig, TrainingConfig

torch.manual_seed(0)
enc_cfg = EncoderConfig(type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0)
cnp = build_cnp(enc_cfg, dim_theta=sampler.dim_theta, dim_phi=sampler.dim_phi)

history = train_cnp(
    cnp, sampler,
    cnp_config=CNPConfig(n_context_min=32, n_context_max=96,
                          output_activation="sigmoid", mixup_alpha=0.1),
    training_config=TrainingConfig(
        n_steps=1500, learning_rate=1e-3,
        batch_size=16, n_events_per_trial=128,
        n_mc_samples=4, eval_every=0,    # ← 0 for real data; see note below
        seed=0,
    ),
)

save_checkpoint("results/cnp.ckpt", cnp,
                encoder_config=enc_cfg,
                dim_theta=sampler.dim_theta, dim_phi=sampler.dim_phi,
                history=history, metadata={"data": "my_simulation_v1"})
```

**Note on `eval_every`:** the in-loop MAE evaluation compares predicted
`β` against analytical `p` from `generator.truth`. Synthetic data has
that; real data does not. Set `eval_every=0` to disable it on real data,
and run your own held-out evaluation (Section 3 / 5) after training.

The CNP loss is **Bernoulli NLL** of the binary `X` under `p = β` — *not*
BCE on `X`. Output `β` is bounded to `[0, 1]` via sigmoid. The aggregator
collapses the **event axis only** (asserted in `forward`).

### 3. CNP-only coverage check (no MFGP)

For pipelines without a fidelity tier (or as a sanity check before MFGP):

```python
from core import cnp_trial_predictive, split_context_target
from viz import plot_coverage_test

# Held-out HF batch
holdout = StandardBatch(mode=InputMode.FULL, theta=hf_theta_test,
                        phi=hf_phi_test, labels=hf_X_test.astype(np.int8))
ctx, tgt = split_context_target(holdout, n_context=64, seed=0)

pred  = cnp_trial_predictive(cnp, ctx, tgt, n_mc_samples=200)
y_raw = tgt.labels.mean(axis=1).astype(float)

coverage = plot_coverage_test(
    y_raw=y_raw,
    y_predicted=pred["y_cnp"],
    sigma_predicted=pred["sigma_total"],
    out_path="results/plots/cnp_only_coverage.png",
    title="CNP-only coverage on held-out HF",
)
print(coverage)  # {"1sigma": 0.68, "2sigma": 0.96, "3sigma": 1.00}
```

### 4. Fit the MFGP on your LF + HF data

```python
from core import (
    prepare_mfgp_datasets_from_batches,
    fit_mfgp_three_fidelity,
    save_mfgp,
)

lf_batch = StandardBatch(mode=InputMode.FULL, theta=lf_theta, phi=lf_phi,
                         labels=lf_X.astype(np.int8))
hf_batch = StandardBatch(mode=InputMode.FULL, theta=hf_theta, phi=hf_phi,
                         labels=hf_X.astype(np.int8))

# Aggregate β̄ via the CNP for both fidelities; collect y_raw = m/N for HF.
data = prepare_mfgp_datasets_from_batches(cnp, lf_batch, hf_batch,
                                           n_mc_samples=50, seed=0)

# Fit the 3-fidelity MFGP (LF β̄, HF β̄, HF y_raw).
mfgp = fit_mfgp_three_fidelity(data, kernel="rbf", n_restarts=5)

save_mfgp("results/mfgp.pkl", mfgp)        # pickle-backed
```

**Predict** at any θ:

```python
import numpy as np
theta_query = np.array([[0.1], [0.2], [0.3]])         # (n, D_θ)
mu, var = mfgp.predict(theta_query)                    # 1-D arrays of length n
# Or get the typed schema:
prediction = mfgp.predict_as_model_prediction(theta_query)
print(prediction.mean, prediction.variance, prediction.theta_query.shape)
```

### 5. Held-out MFGP coverage check

```python
from core import evaluate_mfgp_coverage_from_batch

holdout_batch = StandardBatch(mode=InputMode.FULL, theta=test_theta,
                               phi=test_phi, labels=test_X.astype(np.int8))

result = evaluate_mfgp_coverage_from_batch(mfgp, cnp, holdout_batch, seed=0)

print(f"1σ coverage: {result['1sigma']:.1%}  (target 68.3%)")
print(f"2σ coverage: {result['2sigma']:.1%}  (target 95.5%)")
print(f"3σ coverage: {result['3sigma']:.1%}  (target 99.7%)")

# Plot it the same way as the CNP-only path:
from viz import plot_coverage_test
plot_coverage_test(
    y_raw=result["y_obs"],
    y_predicted=result["mu"],
    sigma_predicted=result["sigma"],
    out_path="results/plots/mfgp_coverage.png",
    title="MFGP coverage on held-out HF",
)
```

### 6. Active learning — IVR (exploration) or EI (exploitation)

Once the MFGP is fit, run an active-learning loop to pick the next θ
to simulate. Two acquisitions are available:

```python
from core import ActiveLearningLoop, BoxBounds

bounds = BoxBounds(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))

loop = ActiveLearningLoop(
    mfgp=mfgp, generator=my_data_generator, cnp=cnp,
    bounds=bounds, data=data,
    n_hf_events=128, n_mc_samples=1000, n_candidates_per_axis=50,
    refit_n_restarts=5,
    acquisition="ei",          # or "ivr"
    target="min",              # for EI: 'min' or 'max'
)
records = loop.run(n_steps=5)

for rec in records:
    print(f"step {rec.step}: θ_next={rec.theta_next}  "
          f"IV {rec.integrated_variance_before:.3e} → {rec.integrated_variance_after:.3e}")
```

* `acquisition="ivr"` — pure exploration. Direction-agnostic;
  `target` is ignored at acquisition time.
* `acquisition="ei"` — Expected Improvement, exploitation-leaning.
  `target="max"` searches for an argmax, `target="min"` for an argmin.
  Stars cluster near the predicted optimum once σ shrinks.

### 7. Save / load + revisit results

| Artifact | Save | Load |
|---|---|---|
| Trained CNP | `save_checkpoint(path, cnp, encoder_config, dim_theta, dim_phi, history, metadata)` | `load_checkpoint(path) -> (cnp, payload)` |
| Fitted MFGP | `save_mfgp(path, mfgp)` | `load_mfgp(path) -> MultiFidelityGP` |
| Plots | matplotlib PNGs at user-chosen `out_path` | open with any image viewer |
| Training history | included in CNP checkpoint payload (`payload["history"]`) | read after `load_checkpoint` |

```python
# Reload and revisualize a saved MFGP without re-fitting:
from core import load_mfgp, load_checkpoint
from viz import plot_coverage_test

cnp_reloaded, payload = load_checkpoint("results/cnp.ckpt")
mfgp_reloaded         = load_mfgp("results/mfgp.pkl")

# Run a fresh held-out coverage check using the reloaded models.
result = evaluate_mfgp_coverage_from_batch(
    mfgp_reloaded, cnp_reloaded, fresh_holdout_batch,
)
plot_coverage_test(
    y_raw=result["y_obs"],
    y_predicted=result["mu"],
    sigma_predicted=result["sigma"],
    out_path="results/plots/coverage_replay.png",
    title="MFGP coverage (replay)",
)
```

### Where results go (recommended layout)

The package itself does not impose an output directory — you pass paths
to every save / plot call. We recommend a flat `results/` tree:

```
results/
├── checkpoints/cnp_<run-tag>.ckpt
├── mfgp/mfgp_<run-tag>.pkl
└── plots/
    ├── cnp_only_coverage.png
    ├── mfgp_coverage.png
    └── ...
```

The `viz_output/phaseN_*/` tree is only used by the demonstration
scripts in `scripts/` — it's gitignored.

---

## Run the synthetic-validation pipelines (Phase 1–5)

For a sanity check on a fresh install, the `scripts/` directory drives
the full 8-scenario synthetic validation. All outputs go to
`viz_output/phaseN_*/` (gitignored):

```bash
python scripts/phase1_plot_ground_truth.py        # pseudo-data plots, S1..S8
python scripts/phase2_plot_latent.py              # encoder null-token PCA
python scripts/phase3_plot_reconstruction.py      # CNP train + reconstruction + coverage
python scripts/phase4_plot_mfgp.py                # MFGP posterior + coverage + Q-Q
python scripts/phase5_plot_optimizer.py           # IVR active learning + trajectory
python scripts/phase5_plot_optimizer.py \
    --acquisition ei --target min --n-initial-hf 4 \
    --scenarios S1,S8 \
    --out-dir viz_output/phase5_optimizer/stress_ei_min
```

Every Phase 5 run also emits an `optimizer_{S}_trajectory.png` and an
`optimizer_{S}_metrics.png` showing how the optimizer reaches the optimum.

## Tests

```bash
pytest                     # 158 tests, ~1 minute on CPU
pytest tests/test_cnp_recovery.py  -v   # the 8-scenario MAE gate (~17s)
pytest tests/test_mfgp_recovery.py -v   # MFGP coverage gate, ~2 min
```

## Further reading

* `CLAUDE.md` — full architecture brief, math reference, validation matrix,
  visualization plan, and live progress checklist.
* The reference paper for the RED problem formulation, CNP loss derivation,
  and MFGP / IVR details: [openreview lqTILjL6lP](https://openreview.net/pdf?id=lqTILjL6lP).
