"""
The batched scan must agree with the unbatched reference scan.

Same seed, same num_samples: identical RNG stream per policy, so the
batched solver selects the same optimal policy and its expected utility
matches the unbatched estimate to float rounding (float32 mean vs the
unbatched float64 accumulation).
"""

from __future__ import annotations

import jax
import pytest

from pylimid.inference.numpyro.solvers import batched_solve
from pylimid.inference.numpyro.solvers.intervention_scan import solve as scan_solve

from .._limid_fixtures import (
    EmptyInfoDecision,
    IrrelevantInfoDecision,
    NestedTwoDecisions,
    SingleDecisionTreat,
    ThreeStateClimate,
    TwoDecisionsIndependent,
)

FIXTURES = [
    SingleDecisionTreat,
    TwoDecisionsIndependent,
    EmptyInfoDecision,
    IrrelevantInfoDecision,
    NestedTwoDecisions,
    ThreeStateClimate,
]

NUM_SAMPLES = 500


@pytest.mark.parametrize("fixture", FIXTURES, ids=[type(f).__name__ for f in FIXTURES])
def test_batched_matches_scan(fixture):
    snapshot = fixture.pylimid().snapshot()
    rng_key = jax.random.PRNGKey(0)
    expected = scan_solve(snapshot, num_samples=NUM_SAMPLES, rng_key=rng_key)
    actual = batched_solve(snapshot, num_samples=NUM_SAMPLES, rng_key=rng_key)
    assert actual.policy == expected.policy
    assert actual.expected_utility == pytest.approx(
        expected.expected_utility, rel=1e-3, abs=1e-3
    )
