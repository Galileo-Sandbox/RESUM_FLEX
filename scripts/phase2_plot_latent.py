"""Phase 2 visual evidence: encoder latent space and shape table.

Produces two artifacts under ``viz_output/phase2_encoder/``:

* ``encoder_shape_table.txt`` — a row per scenario S1..S8 showing the
  declared input dims and the resulting ``z_θ`` / ``z_φ`` tensor shapes.
  This is the "dimension matrix" gate from the validation plan.
* ``encoder_latent_S1_vs_S5.png`` — PCA(2) scatter of ``z_θ`` for an S1
  batch (θ provided) and an S5 batch (θ=None). The S5 points must
  collapse to a single point (the null token). Visual proof that null
  inputs map to *the same* learnable embedding by construction.

The encoder is randomly initialized; both plots show structural
properties that hold without training.

Run from the repo root:

    python scripts/phase2_plot_latent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.networks import build_encoder  # noqa: E402
from data.pseudo_generator import PseudoDataGenerator, for_scenario  # noqa: E402
from schemas.config import EncoderConfig  # noqa: E402

OUT_DIR = Path("viz_output/phase2_encoder")


def _make_config(latent_dim: int = 16) -> EncoderConfig:
    return EncoderConfig(
        type="mlp",
        latent_dim=latent_dim,
        hidden_dims=[32, 32],
        dropout=0.0,
    )


def write_shape_table(out_path: Path) -> None:
    """Run an encoder forward for every scenario and tabulate I/O shapes."""
    config = _make_config()
    header = (
        f"{'scenario':<10}{'mode':<14}{'dim_θ':<8}{'dim_φ':<8}"
        f"{'z_θ.shape':<18}{'z_φ.shape':<18}"
    )
    rows = [header, "-" * len(header)]
    for name in PseudoDataGenerator.SCENARIOS:
        gen = for_scenario(name)
        batch = gen.generate(n_trials=4, n_events=16)
        enc = build_encoder(config, gen.dim_theta, gen.dim_phi)
        enc.eval()
        with torch.no_grad():
            z_t, z_p = enc(batch)
        rows.append(
            f"{name:<10}{batch.mode.value:<14}"
            f"{str(gen.dim_theta):<8}{str(gen.dim_phi):<8}"
            f"{str(tuple(z_t.shape)):<18}{str(tuple(z_p.shape)):<18}"
        )
    out_path.write_text("\n".join(rows) + "\n")
    print(f"  wrote {out_path}")


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project rows of X onto the top-2 principal directions.

    Returns
    -------
    coords : (N, 2) projected coordinates
    explained : (2,) fraction of total variance captured by each axis
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var = (S**2) / max(len(X) - 1, 1)
    explained = var[:2] / var.sum()
    return coords, explained


def plot_latent_S1_vs_S5(out_path: Path, *, B: int = 64, seed: int = 0) -> None:
    config = _make_config(latent_dim=16)
    torch.manual_seed(seed)
    encoder = build_encoder(config, dim_theta=1, dim_phi=1)
    encoder.eval()

    s1 = for_scenario("S1", seed=seed).generate(n_trials=B, n_events=8)
    s5 = for_scenario("S5", seed=seed).generate(n_trials=B, n_events=8)

    with torch.no_grad():
        z_s1, _ = encoder(s1)
        z_s5, _ = encoder(s5)
        null_token = encoder.theta_null.detach().cpu().numpy()

    Z = torch.cat([z_s1, z_s5], dim=0).detach().cpu().numpy()
    coords, explained = _pca_2d(Z)
    coords_s1, coords_s5 = coords[:B], coords[B:]

    # Sanity check that the null cluster really collapsed.
    spread = float(np.linalg.norm(coords_s5 - coords_s5[0:1], axis=1).max())
    assert spread == 0.0, f"S5 cluster did not collapse to a point (spread={spread})"

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(
        coords_s1[:, 0], coords_s1[:, 1],
        s=22, alpha=0.7, c="C0",
        label=f"S1: θ provided, n={B} (spread)",
    )
    ax.scatter(
        coords_s5[:, 0], coords_s5[:, 1],
        s=160, alpha=0.45, c="C3", marker="*",
        label=f"S5: θ=None → null token, n={B} (collapsed)",
    )
    ax.set_title(
        "Encoder z_θ in PCA(2) — null inputs collapse to a single point\n"
        "(encoder is randomly initialized; structural property only)"
    )
    ax.set_xlabel(f"PC1  ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({explained[1] * 100:.1f}% var)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}  (S5 cluster spread = {spread:.2e})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_shape_table(OUT_DIR / "encoder_shape_table.txt")
    plot_latent_S1_vs_S5(OUT_DIR / "encoder_latent_S1_vs_S5.png")


if __name__ == "__main__":
    main()
