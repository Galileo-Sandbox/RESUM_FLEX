from core.networks import MLPEncoder, UniversalEncoder, build_encoder
from core.surrogate_cnp import (
    CnpDecoder,
    CnpOutput,
    CnpObjective,
    ConditionalNeuralProcess,
    ContextPointEncoder,
    build_cnp,
    cnp_loss,
    practice_truth_loss,
    resum_binary_moments,
    split_context_target,
    theory_truth_loss,
)
from core.scaling import MinMaxScaler
from core.training import (
    TrainingHistory,
    cnp_trial_predictive,
    evaluate_mae,
    load_checkpoint,
    save_checkpoint,
    train_cnp,
)

_LAZY_GP_EXPORTS = {
    "ActiveLearningLoop": ("core.optimizer", "ActiveLearningLoop"),
    "ActiveLearningStep": ("core.optimizer", "ActiveLearningStep"),
    "BoxBounds": ("core.optimizer", "BoxBounds"),
    "ExpectedImprovementAcquisition": (
        "core.optimizer",
        "ExpectedImprovementAcquisition",
    ),
    "HighFidelityObservation": ("core.optimizer", "HighFidelityObservation"),
    "IvrAcquisition": ("core.optimizer", "IvrAcquisition"),
    "MultiFidelityGP": ("core.surrogate_mfgp", "MultiFidelityGP"),
    "ObservationProvider": ("core.optimizer", "ObservationProvider"),
    "SyntheticObservationProvider": (
        "core.optimizer",
        "SyntheticObservationProvider",
    ),
    "evaluate_mfgp_coverage": ("core.mfgp_pipeline", "evaluate_mfgp_coverage"),
    "evaluate_mfgp_coverage_from_batch": (
        "core.mfgp_pipeline",
        "evaluate_mfgp_coverage_from_batch",
    ),
    "fit_mfgp_three_fidelity": ("core.mfgp_pipeline", "fit_mfgp_three_fidelity"),
    "integrated_variance": ("core.optimizer", "integrated_variance"),
    "load_mfgp": ("core.surrogate_mfgp", "load_mfgp"),
    "posterior_covariance": ("core.optimizer", "posterior_covariance"),
    "prepare_mfgp_datasets": ("core.mfgp_pipeline", "prepare_mfgp_datasets"),
    "prepare_mfgp_datasets_from_batches": (
        "core.mfgp_pipeline",
        "prepare_mfgp_datasets_from_batches",
    ),
    "save_mfgp": ("core.surrogate_mfgp", "save_mfgp"),
    "simulate_at_theta": ("core.optimizer", "simulate_at_theta"),
}


def __getattr__(name: str):
    """Load GP-dependent exports only when they are requested."""
    target = _LAZY_GP_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute = target
    try:
        value = getattr(import_module(module_name), attribute)
    except ModuleNotFoundError as exc:
        if exc.name in {"GPy", "emukit"}:
            raise ImportError(
                f"{name} requires the optional GP dependencies; "
                'install them with `pip install -e ".[gp]`.'
            ) from exc
        raise
    globals()[name] = value
    return value

__all__ = [
    "ActiveLearningLoop",
    "ActiveLearningStep",
    "BoxBounds",
    "CnpDecoder",
    "CnpOutput",
    "CnpObjective",
    "ConditionalNeuralProcess",
    "ContextPointEncoder",
    "ExpectedImprovementAcquisition",
    "HighFidelityObservation",
    "IvrAcquisition",
    "MLPEncoder",
    "MinMaxScaler",
    "MultiFidelityGP",
    "ObservationProvider",
    "SyntheticObservationProvider",
    "TrainingHistory",
    "UniversalEncoder",
    "build_cnp",
    "build_encoder",
    "cnp_loss",
    "practice_truth_loss",
    "resum_binary_moments",
    "cnp_trial_predictive",
    "evaluate_mae",
    "evaluate_mfgp_coverage",
    "evaluate_mfgp_coverage_from_batch",
    "fit_mfgp_three_fidelity",
    "integrated_variance",
    "load_checkpoint",
    "load_mfgp",
    "posterior_covariance",
    "prepare_mfgp_datasets",
    "prepare_mfgp_datasets_from_batches",
    "save_checkpoint",
    "save_mfgp",
    "simulate_at_theta",
    "split_context_target",
    "theory_truth_loss",
    "train_cnp",
]
