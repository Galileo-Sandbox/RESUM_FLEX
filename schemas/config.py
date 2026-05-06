"""Static configuration schema for RESUM_FLEX.

Loads ``config.yaml`` into a typed pydantic tree so hyperparameters,
kernel choices, and per-scenario MAE thresholds are validated up-front
rather than raising deep inside training. Each subsection corresponds to
one phase of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class EncoderConfig(BaseModel):
    type: Literal["mlp", "transformer"] = "mlp"
    latent_dim: int = Field(gt=0)
    hidden_dims: list[int]
    dropout: float = Field(ge=0.0, lt=1.0, default=0.0)

    @field_validator("hidden_dims")
    @classmethod
    def _check_hidden(cls, v: list[int]) -> list[int]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("hidden_dims must be a non-empty list of positive ints")
        return v


class CNPConfig(BaseModel):
    n_context_min: int = Field(gt=0)
    n_context_max: int = Field(gt=0)
    output_activation: Literal["sigmoid"] = "sigmoid"
    mixup_alpha: float = Field(gt=0.0)

    @field_validator("n_context_max")
    @classmethod
    def _check_range(cls, v: int, info) -> int:
        n_min = info.data.get("n_context_min")
        if n_min is not None and v < n_min:
            raise ValueError("n_context_max must be >= n_context_min")
        return v


class MFGPConfig(BaseModel):
    kernel: Literal["rbf", "matern"] = "rbf"
    n_fidelities: int = Field(ge=2, default=3)


class IVRConfig(BaseModel):
    n_mc_samples: int = Field(gt=0, default=1000)


class ScenarioThresholds(BaseModel):
    """Per-scenario MAE thresholds for the Phase 3 acceptance gate."""

    s1: float = Field(gt=0)
    s2: float = Field(gt=0)
    s3: float = Field(gt=0)
    s4: float = Field(gt=0)
    s5: float = Field(gt=0)
    s6: float = Field(gt=0)
    s7: float = Field(gt=0)
    s8: float = Field(gt=0)


class Config(BaseModel):
    seed: int = 42
    encoder: EncoderConfig
    cnp: CNPConfig
    mfgp: MFGPConfig
    ivr: IVRConfig
    mae_thresholds: ScenarioThresholds


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
