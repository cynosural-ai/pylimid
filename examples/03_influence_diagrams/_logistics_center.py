"""
The Automated Logistics Center as a mixed continuous/discrete LIMID.

An e-commerce company may run an expensive feasibility test on an advanced
robotics system before deciding whether to invest in it or in a conventional
system. This is the mixed-type variant of the discrete logistics problem: the
test no longer reports one of three categories directly, it measures a
*continuous* performance score, and the report the company acts on is the
score binned into ``bad`` / ``good`` / ``excellent``.

Nodes:

- ``advanced_state`` (chance, discrete): ``smooth`` / ``minor`` / ``major``.
- ``conventional_state`` (chance, discrete): ``smooth`` / ``failure``.
- ``test`` (decision): ``test`` / ``no_test`` / ``nothing``.
- ``score`` (chance, continuous): the performance score, Gamma distributed with
  a per-state mean — the only continuous variable.
- ``report`` (chance, discrete): the binned score, ``bad`` / ``good`` /
  ``excellent``; a flat filler when the test is not run.
- ``invest`` (decision): ``advanced`` / ``conventional``; observes the report
  and remembers whether the test was run.
- ``utility`` (utility): payoff in € million, a function of ``test``,
  ``invest``, ``advanced_state`` and ``conventional_state``.

The continuous ``score`` never enters a decision's information set — the
decision observes the discrete ``report`` — so the diagram stays solvable by
the discrete-decision solver family. The companion reference solves the same
model in pyAgrum after collapsing ``score`` into the report distribution it
induces, ``P(report | advanced_state)``.

The numbers follow the discrete problem's ``problem.yaml``; the score's Gamma
parameters are a fresh (not fitted) parameterization chosen so the induced
report probabilities tell the same story.
"""

from __future__ import annotations

from itertools import product

import jax.numpy as jnp
import numpyro.distributions as dist
import pyagrum as gum

from pylimid.graph import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode

__all__ = [
    "induced_report_probs",
    "pyagrum_diagram",
    "pyagrum_solution",
    "pylimid_diagram",
]

# --- problem constants ------------------------------------------------------

ADVANCED_STATES = ("smooth", "minor", "major")
ADVANCED_PRIOR = (0.70, 0.20, 0.10)

CONVENTIONAL_STATES = ("smooth", "failure")
CONVENTIONAL_PRIOR = (0.95, 0.05)

TEST_STATES = ("test", "no_test", "nothing")
REPORT_STATES = ("bad", "good", "excellent")
INVEST_STATES = ("advanced", "conventional")

#: Mean performance score for each advanced-system state (higher is better).
SCORE_MEANS = (8.0, 5.0, 3.0)
#: Gamma concentration shared by the states; the rate is concentration / mean.
SCORE_CONCENTRATION = 25.0
#: Score cut points: below the first is ``bad``, above the second ``excellent``.
SCORE_THRESHOLDS = (4.0, 7.0)

#: Payoff by ``[test][advanced_state]`` when investing in the advanced system.
_ADVANCED_PAYOFF = (
    (140.0, -50.0, -130.0),  # test
    (150.0, -40.0, -120.0),  # no_test
    (0.0, 0.0, 0.0),  # nothing
)
#: Payoff by ``[test][conventional_state]`` when investing in the conventional.
_CONVENTIONAL_PAYOFF = (
    (50.0, -40.0),  # test
    (60.0, -30.0),  # no_test
    (0.0, 0.0),  # nothing
)

_SCORE_MEANS = jnp.array(SCORE_MEANS)
_ADVANCED = jnp.array(_ADVANCED_PAYOFF)
_CONVENTIONAL = jnp.array(_CONVENTIONAL_PAYOFF)

# --- model builders ---------------------------------------------------------


def pylimid_diagram() -> InfluenceDiagram:
    """Build the mixed continuous/discrete logistics influence diagram."""
    diag = InfluenceDiagram()
    diag.add_node(
        ChanceNode(
            name="advanced_state",
            states=ADVANCED_STATES,
            dist=lambda: dist.Categorical(probs=jnp.array(ADVANCED_PRIOR)),
        )
    )
    diag.add_node(
        ChanceNode(
            name="conventional_state",
            states=CONVENTIONAL_STATES,
            dist=lambda: dist.Categorical(probs=jnp.array(CONVENTIONAL_PRIOR)),
        )
    )
    diag.add_node(DecisionNode(name="test", states=TEST_STATES))
    diag.add_node(
        ChanceNode(
            name="score",
            parents=("advanced_state",),
            dist=_score_dist,
        )
    )
    diag.add_node(
        ChanceNode(
            name="report",
            parents=("score", "test"),
            states=REPORT_STATES,
            dist=_report_dist,
        )
    )
    diag.add_node(
        DecisionNode(name="invest", parents=("test", "report"), states=INVEST_STATES)
    )
    diag.add_node(
        UtilityNode(
            name="utility",
            parents=("test", "invest", "advanced_state", "conventional_state"),
            values=_utility,
        )
    )
    return diag


def induced_report_probs():
    """
    ``P(report | advanced_state)`` induced by binning the Gamma score.

    Returns:
        A ``(len(ADVANCED_STATES), len(REPORT_STATES))`` array of rows
        ``(bad, good, excellent)``.
    """
    low, high = SCORE_THRESHOLDS
    rows = []
    for index in range(len(ADVANCED_STATES)):
        cdf = _score_dist(index).cdf
        p_bad = cdf(jnp.array(low))
        p_good = cdf(jnp.array(high)) - p_bad
        rows.append(jnp.stack([p_bad, p_good, 1.0 - p_bad - p_good]))
    return jnp.stack(rows)


def pyagrum_diagram() -> gum.InfluenceDiagram:  # ty: ignore[possibly-missing-attribute]
    """
    Build the collapsed discrete pyAgrum diagram.

    ``score`` is integrated out: ``report`` becomes a chance node whose CPD is
    ``P(report | advanced_state, test)`` — the binned distribution when the
    test runs, uniform otherwise. Everything else mirrors `pylimid_diagram`.
    """
    edges = [
        ("advanced_state", "report"),
        ("test", "report"),
        ("test", "invest"),
        ("report", "invest"),
        ("test", "utility"),
        ("invest", "utility"),
        ("advanced_state", "utility"),
        ("conventional_state", "utility"),
    ]
    g = gum.InfluenceDiagram()  # ty: ignore[possibly-missing-attribute]
    g.addChanceNode(
        gum.LabelizedVariable("advanced_state", "advanced_state", list(ADVANCED_STATES))
    )
    g.addChanceNode(
        gum.LabelizedVariable(
            "conventional_state", "conventional_state", list(CONVENTIONAL_STATES)
        )
    )
    g.addDecisionNode(gum.LabelizedVariable("test", "test", list(TEST_STATES)))
    g.addChanceNode(gum.LabelizedVariable("report", "report", list(REPORT_STATES)))
    g.addDecisionNode(gum.LabelizedVariable("invest", "invest", list(INVEST_STATES)))
    g.addUtilityNode(gum.LabelizedVariable("utility", "utility", ["pay"]))
    for parent, child in edges:
        g.addArc(g.idFromName(parent), g.idFromName(child))

    g.cpt("advanced_state")[{}] = list(ADVANCED_PRIOR)
    g.cpt("conventional_state")[{}] = list(CONVENTIONAL_PRIOR)

    induced = induced_report_probs()
    uniform = [1.0 / len(REPORT_STATES)] * len(REPORT_STATES)
    for a, t in product(range(len(ADVANCED_STATES)), range(len(TEST_STATES))):
        probs = induced[a].tolist() if t == 0 else uniform
        g.cpt("report")[{"advanced_state": a, "test": t}] = probs

    for t, i, a, c in product(
        range(len(TEST_STATES)),
        range(len(INVEST_STATES)),
        range(len(ADVANCED_STATES)),
        range(len(CONVENTIONAL_STATES)),
    ):
        value = _ADVANCED_PAYOFF[t][a] if i == 0 else _CONVENTIONAL_PAYOFF[t][c]
        g.utility("utility")[
            {"test": t, "invest": i, "advanced_state": a, "conventional_state": c}
        ] = value
    return g


def pyagrum_solution() -> tuple[float, dict[str, dict[tuple[int, ...], int]]]:
    """
    Solve the collapsed discrete diagram exactly with pyAgrum.

    Returns:
        ``(meu, policy)`` — the mean expected utility and the per-decision
        optimal policy in pylimid's format (decision name to
        ``{info_set_assignment: action}``).
    """
    ie = gum.ShaferShenoyLIMIDInference(pyagrum_diagram())  # ty: ignore[possibly-missing-attribute]
    ie.makeInference()
    meu = float(ie.MEU()["mean"])

    info_order = {
        "test": (),
        "invest": ("test", "report"),
    }
    cards = {
        "test": len(TEST_STATES),
        "report": len(REPORT_STATES),
    }
    policy: dict[str, dict[tuple[int, ...], int]] = {}
    for name, info in info_order.items():
        decisions = ie.optimalDecision(name)
        sub: dict[tuple[int, ...], int] = {}
        for assignment in product(*[range(cards[p]) for p in info]):
            entry = dict(zip(info, assignment, strict=True))
            for action in range(2):
                if float(decisions[{**entry, name: action}]) >= 0.999999:
                    sub[assignment] = action
                    break
        policy[name] = sub
    return meu, policy


# --- pylimid distribution / utility callables -------------------------------


def _score_dist(advanced_state):
    """Gamma performance score for an advanced-system state index."""
    rate = SCORE_CONCENTRATION / _SCORE_MEANS[advanced_state]
    return dist.Gamma(concentration=SCORE_CONCENTRATION, rate=rate)


def _report_dist(score, test):
    """Bin the continuous score; a flat filler when the test is not run."""
    low, high = SCORE_THRESHOLDS
    binned = jnp.stack(
        [score < low, (score >= low) & (score < high), score >= high], axis=-1
    ).astype(jnp.float32)
    uniform = jnp.full_like(binned, 1.0 / len(REPORT_STATES))
    return dist.Categorical(
        probs=jnp.where(jnp.expand_dims(test == 0, -1), binned, uniform)
    )


def _utility(test, invest, advanced_state, conventional_state):
    """Payoff: advanced-system table when investing, conventional-table else."""
    advanced = _ADVANCED[test, advanced_state]
    conventional = _CONVENTIONAL[test, conventional_state]
    return jnp.where(invest == 0, advanced, conventional)
