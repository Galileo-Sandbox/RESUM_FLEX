from __future__ import annotations

import numpy as np
import pytest

from data import FixedBatchSource, for_scenario


def test_fixed_batch_source_is_reproducible_and_preserves_pairs() -> None:
    full = for_scenario("S1").generate(6, 20, seed=7)
    source = FixedBatchSource(full)
    first = source.generate(4, 8, seed=11)
    second = source.generate(4, 8, seed=11)
    np.testing.assert_array_equal(first.theta, second.theta)
    np.testing.assert_array_equal(first.phi, second.phi)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert source.dim_theta == 1 and source.dim_phi == 1


def test_fixed_batch_source_checks_sampling_limits() -> None:
    full = for_scenario("S7").generate(3, 5, seed=0)
    with pytest.raises(ValueError, match="events"):
        FixedBatchSource(full).generate(2, 6, seed=0)
    with pytest.raises(ValueError, match="trials"):
        FixedBatchSource(full, replace_trials=False).generate(4, 2, seed=0)
