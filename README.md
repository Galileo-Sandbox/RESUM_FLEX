# RESUM_FLEX

A modular refactor of **RESuM** (Rare Event Surrogate Model) — a physics-ML
pipeline for **Rare Event Design (RED)** problems where the design metric
`y = m/N` is a discrete count with high variance. The framework denoises
binary observations with a Conditional Neural Process (CNP), then fuses
multi-fidelity scores with a Multi-Fidelity Gaussian Process (MFGP), and
selects new design points by Integrated Variance Reduction (IVR).

Reference paper: [RESuM: A Rare Event Surrogate Model](https://openreview.net/pdf?id=lqTILjL6lP).
The original RESuM repo is `/home/yuema137/resum`; this project ports the
ideas into a modular, dimension-flexible, universal-input architecture.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Pydantic schemas (`StandardBatch`, `DesignPoint`, `ModelPrediction`, `Config`) | ✅ |
| 1 | Pseudo-data generator with analytical `t(θ, φ)` + 8-scenario plots | ✅ |
| 2 | Universal encoder with learnable null embeddings (PyTorch) | ✅ |
| 3 | CNP training, evaluation, checkpoints, reconstruction & coverage plots | ✅ |
| 4 | MFGP co-kriging via Emukit/GPy | ⏳ |
| 5 | IVR active-learning optimizer | ⏳ |

132 tests pass. See `CLAUDE.md` for the full architecture brief, math
reference, and the per-phase visualization plan.

## Layout

```
RESUM_FLEX/
├── schemas/           pydantic data contracts (numpy-only, no torch / GPy)
│   ├── data_models.py    StandardBatch, DesignPoint, EventBatch, ModelPrediction
│   └── config.py         Typed YAML config tree (load_config)
├── core/              compute modules
│   ├── networks.py       Universal encoder + null embeddings (PyTorch)
│   ├── surrogate_cnp.py  CNP forward, Bernoulli-NLL loss, ctx/target split
│   └── training.py       train_cnp, evaluate_mae, cnp_trial_predictive, checkpoints
├── data/              synthetic data
│   └── pseudo_generator.py   GaussianBumpTruth + PseudoDataGenerator + for_scenario()
├── viz/               plotting primitives
│   └── dispatch.py       plot_field, plot_comparison_1d/2d, plot_coverage_test
├── scripts/           runnable end-to-end pipelines
│   ├── phase1_plot_ground_truth.py     pseudo-data plots over S1..S8
│   ├── phase2_plot_latent.py           encoder null-token PCA + shape table
│   └── phase3_plot_reconstruction.py   CNP train + reconstruction + coverage
├── tests/             pytest (no real-data fixtures; everything synthetic)
├── config.yaml        all hyperparameters
├── pyproject.toml     deps + ruff/pytest config
└── CLAUDE.md          authoritative agent brief; deep architecture & math
```

**Decoupling rule:** PyTorch lives only in `core/networks.py`,
`core/surrogate_cnp.py`, and `core/training.py`. The schema layer is
numpy-native so the (eventual) MFGP module in `core/surrogate_mfgp.py`
can consume `StandardBatch` arrays without dragging torch into the
GP/Emukit stack.

## Install

```bash
git clone <this repo>
cd RESUM_FLEX

# project-local venv (uv recommended, but python -m venv works too)
uv venv .venv --python 3.12
source .venv/bin/activate

# core deps + dev tooling
uv pip install pyyaml pydantic numpy pytest matplotlib
uv pip install --index-url https://download.pytorch.org/whl/cpu torch  # CPU build
```

Phase 4 will add `GPy` and `emukit` (already declared as the `gp` extra in
`pyproject.toml`). Until then, those are optional.

## Data format: `StandardBatch`

The pipeline carries `StandardBatch` objects. All array fields are
`numpy.ndarray`. Three input modalities are supported via the `mode` flag,
and the encoder handles each transparently with learnable null embeddings:

| `mode` | `theta` | `phi` | `labels` (X) |
|---|---|---|---|
| `FULL` | `[B, D_θ]` | `[B, N, D_φ]` | `[B, N]` binary |
| `EVENT_ONLY` | `None` | `[B, N, D_φ]` | `[B, N]` binary |
| `DESIGN_ONLY` | `[B, D_θ]` | `None` | `[B, N]` binary |

`D_θ` and `D_φ` are arbitrary (≥ 1). The validator cross-checks that the
declared `mode` matches the presence of `theta` / `phi` and that all batch
dims agree, so malformed batches fail fast at construction.

Optional fields: `beta` (`[B, N]` in `[0, 1]`, populated by the CNP).

The 8-scenario validation matrix S1..S8 covers the cross-product of
modalities × `dim(θ)` × `dim(φ)` — see `CLAUDE.md` for the full table.

## Usage

### 1. Generate synthetic data

```python
from data import for_scenario

gen = for_scenario("S1")           # FULL, dim_θ=1, dim_φ=1
batch = gen.generate(n_trials=4, n_events=64)
print(batch.theta.shape, batch.phi.shape, batch.labels.shape)
# (4, 1)  (4, 64, 1)  (4, 64)

# Analytical truth at any (θ, φ) — for validation
p = gen.truth.evaluate(theta=batch.theta[:, None, :], phi=batch.phi)  # [4, 64]
```

`for_scenario(name)` covers `"S1"..."S8"`. For ad-hoc setups, build a
`GaussianBumpTruth` and a `PseudoDataGenerator` directly.

### 2. Build & train a CNP

```python
import torch
from core import build_cnp, train_cnp, evaluate_mae
from data import for_scenario
from schemas.config import EncoderConfig, CNPConfig, TrainingConfig

torch.manual_seed(0)
gen = for_scenario("S5")           # EVENT_ONLY, 1D φ

cnp = build_cnp(
    EncoderConfig(type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0),
    dim_theta=gen.dim_theta, dim_phi=gen.dim_phi,
)

history = train_cnp(
    cnp, gen,
    cnp_config=CNPConfig(
        n_context_min=32, n_context_max=96,
        output_activation="sigmoid", mixup_alpha=0.1,
    ),
    training_config=TrainingConfig(
        n_steps=600, learning_rate=1e-3, batch_size=16,
        n_events_per_trial=128, n_mc_samples=4, eval_every=200, seed=0,
    ),
)

mae = evaluate_mae(cnp, gen, batch_size=64, n_events=256, n_context=128, seed=999)
print(f"Final MAE(β, p) = {mae:.4f}")
```

CNP loss is **Bernoulli NLL** of the target binary X under `p = β` — *not*
BCE on X. Output `β` is bounded to `[0, 1]` via sigmoid. The aggregator
collapses the **event axis only**, never the batch axis (asserted in
`forward`).

### 3. Save / load checkpoints

```python
from core import save_checkpoint, load_checkpoint

save_checkpoint(
    "runs/cnp_S5.ckpt", cnp,
    encoder_config=EncoderConfig(type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0),
    dim_theta=gen.dim_theta, dim_phi=gen.dim_phi,
    history=history, metadata={"scenario": "S5"},
)

cnp_loaded, payload = load_checkpoint("runs/cnp_S5.ckpt")
```

### 4. Trial-level predictive distribution & coverage

```python
from core import cnp_trial_predictive, split_context_target
from viz import plot_coverage_test

test = gen.generate(n_trials=100, n_events=128, seed=98765)
ctx, tgt = split_context_target(test, n_context=64, seed=98766)

pred = cnp_trial_predictive(cnp, ctx, tgt, n_mc_samples=200, include_aleatoric=True)
y_raw = tgt.labels.mean(axis=1).astype(float)

coverage = plot_coverage_test(
    y_raw=y_raw,
    y_predicted=pred["y_cnp"],
    sigma_predicted=pred["sigma_total"],
    out_path="viz_output/cnp_coverage_S5.png",
    title="S5 — Pre-MFGP coverage (CNP only, 100 held-out trials)",
)
print(coverage)   # {"1sigma": 0.68, "2sigma": 0.96, "3sigma": 1.00}
```

`pred` returns `y_cnp`, `sigma_total`, `sigma_epistemic`, `sigma_aleatoric`
separately. Use `include_aleatoric=False` when you want a pure
decoder-calibration diagnostic instead of the m/N-comparison.

### 5. Run a full phase as a script

```bash
python scripts/phase1_plot_ground_truth.py        # writes viz_output/pseudo_ground_truth_*.png
python scripts/phase2_plot_latent.py              # writes encoder_latent_*.png + shape table
python scripts/phase3_plot_reconstruction.py      # trains CNPs + writes 3 plots × 8 scenarios
```

`viz_output/` is in `.gitignore` — plots are build products, not source.

## Configuration

Everything tunable lives in `config.yaml`, validated against pydantic models
in `schemas/config.py`:

```yaml
encoder:        # MLP layer sizes, latent dim, dropout
cnp:            # context-size range, output activation, mixup α
mfgp:           # kernel type, n_fidelities (Phase 4)
ivr:            # n_mc_samples for IVR (Phase 5)
training:       # n_steps, learning rate, batch size, events/trial, …
mae_thresholds: # per-scenario S1..S8 acceptance thresholds for Phase 3 gate
```

Load via `from schemas.config import load_config; cfg = load_config("config.yaml")`.

## Tests

```bash
pytest tests/             # 132 tests, ~20s on CPU
pytest tests/test_cnp_recovery.py -v   # the 8-scenario MAE gate (~17s)
```

## Further reading

* `CLAUDE.md` — full architecture brief, math reference, validation matrix,
  visualization plan, and live progress checklist.
* The reference paper for the RED problem formulation, CNP loss derivation,
  and MFGP / IVR details: [openreview lqTILjL6lP](https://openreview.net/pdf?id=lqTILjL6lP).
