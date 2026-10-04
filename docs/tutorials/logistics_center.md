---
file_format: mystnb
kernelspec:
  name: python3
---

# The logistics center

An e-commerce company is standing in front of a very expensive decision. Its
logistics center is at capacity, and there are three ways forward: upgrade to a
**conventional** robotics system, bet on an **advanced** AI-powered system, or
**do nothing** and keep the current operation.

The advanced system is the prize. If it works, it changes the economics of the
company. If it fails, it fails hard. Nobody knows which before the robots are
installed — but there is a **feasibility test**, and for €10M it might tell us
something before we commit.

```{figure} /_static/figures/logistics_center/logistics_center.jpg
:alt: Robots sorting packages in an automated logistics center
:width: 80%

An automated logistics center ([image source](https://www.xataka.com/robotica-e-ia/primer-almacen-gestionado-integramente-robots-sigue-necesitando-a-humanos-5-tecnicos-para-tareas-mantenimiento)).
```

This is the library's **mixed-type** showcase. The test does not return one of
three neat categories; it returns a **continuous performance score**, and the
company acts on that score binned into ``bad`` / ``good`` / ``excellent``. A
continuous chance variable, discrete decisions, and an answer we can check
against exact arithmetic. It is the companion to [The oil field](oil_field.md),
which is fully discrete.

## The numbers

Two things are uncertain: how the advanced system behaves, and how the
conventional one would. The test adds a continuous measurement of the first.

The **advanced system** can behave in one of three ways:

| Advanced state | Probability |
| --- | --- |
| smooth | 0.70 |
| minor failure | 0.20 |
| major failure | 0.10 |

The **conventional system** can behave in one of two ways:

| Conventional state | Probability |
| --- | --- |
| smooth | 0.95 |
| failure | 0.05 |

And the payoffs, in € million, depend on the choice and on how the chosen
system performs. The test's €10M cost is not a separate line item — it is
already baked into the numbers on the ``test`` rows (a tested advanced system
that runs smoothly earns 140 rather than 150, and so on).

| | test | no test |
| --- | --- | --- |
| **Advanced**, smooth | 140 | 150 |
| **Advanced**, minor failure | −50 | −40 |
| **Advanced**, major failure | −130 | −120 |
| **Conventional**, smooth | 50 | 60 |
| **Conventional**, failure | −40 | −30 |

Doing nothing pays **0**, whatever happens.

```{admonition} Reading the decision sequence
:class: note
First decide whether to test; if you test, see the report; then choose the
system. The choice of system is not frozen when the test is ordered — that is
the whole point of paying for information. In the diagram, this is a **memory
arc**: the investment decision must remember whether a test was actually run,
because "no report" and "a bad report" are very different situations.
```

## A test that measures, not labels

In a textbook version of this problem the test would flip a coin weighted by
the true state and print ``bad``, ``good``, or ``excellent``. But real
diagnostics are not three-sided coins. Our test measures a **performance
score**: a continuous number where higher is better. The score depends on the
true state of the advanced system:

| Advanced state | Mean score |
| --- | --- |
| smooth | 8 |
| minor failure | 5 |
| major failure | 3 |

The score is positive and skewed — a natural fit for a Gamma distribution —
and its spread is controlled by a concentration parameter. With concentration
25 the standard deviation is one fifth of the mean, so the three states
overlap a little but are clearly distinguishable.

The company does not get to act on the raw number. It commits to a reporting
rule ahead of time: scores **below 4** are reported ``bad``, scores **between
4 and 7** are ``good``, and scores **above 7** are ``excellent``. That keeps
the decision table finite while the uncertainty stays continuous.

```{admonition} Why bin a continuous measurement?
:class: note
The investment decision has two actions, so it needs a finite information set.
Binning is the honest way to model "we will look at the score and respond on
these thresholds" while still letting the score carry more information than
three hand-written probabilities ever could.
```

## Modelling it in pylimid

Chance distributions, decisions and utilities in `pylimid` are ordinary Python
callables backed by NumPyro. Let us build the model from the numbers above.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, infer, solve
```

First the labels, the priors, and the two payoff tables. Each payoff row is
indexed by the test decision (``0 = test``, ``1 = no_test``, ``2 = nothing``)
and then by the relevant system's state.

```{code-cell} ipython3
ADVANCED_STATES = ("smooth", "minor", "major")
CONVENTIONAL_STATES = ("smooth", "failure")
TEST_STATES = ("test", "no_test", "nothing")
REPORT_STATES = ("bad", "good", "excellent")
INVEST_STATES = ("advanced", "conventional")

advanced_prior = jnp.array([0.70, 0.20, 0.10])
conventional_prior = jnp.array([0.95, 0.05])

advanced_payoff = jnp.array([
    [140.0, -50.0, -130.0],  # test
    [150.0, -40.0, -120.0],  # no test
    [0.0, 0.0, 0.0],         # do nothing
])
conventional_payoff = jnp.array([
    [50.0, -40.0],  # test
    [60.0, -30.0],  # no test
    [0.0, 0.0],     # do nothing
])
```

Now the shape of the test. The score mean comes from the state; the Gamma
concentration is shared; the thresholds turn the score into the report.

```{code-cell} ipython3
score_means = jnp.array([8.0, 5.0, 3.0])
score_concentration = 25.0
score_thresholds = (4.0, 7.0)
```

The three callables below are the heart of the model. Read them as the arrows
of the diagram: ``advanced_state -> score``, ``score + test -> report``, and
the payoff function over all four parents.

```{code-cell} ipython3
def score_dist(advanced_state):
    """The performance score: a Gamma whose mean is set by the state."""
    return dist.Gamma(
        concentration=score_concentration,
        rate=score_concentration / score_means[advanced_state],
    )


def report_dist(score, test):
    """Bin the continuous score; a flat filler when the test is not run."""
    low, high = score_thresholds
    binned = jnp.stack(
        [score < low, (score >= low) & (score < high), score >= high], axis=-1
    ).astype(jnp.float32)
    uniform = jnp.full_like(binned, 1.0 / len(REPORT_STATES))
    return dist.Categorical(
        probs=jnp.where(jnp.expand_dims(test == 0, -1), binned, uniform)
    )


def utility(test, invest, advanced_state, conventional_state):
    """Investing pays off per the advanced table, or per the conventional one."""
    advanced = advanced_payoff[test, advanced_state]
    conventional = conventional_payoff[test, conventional_state]
    return jnp.where(invest == 0, advanced, conventional)
```

```{admonition} The report when no test runs
:class: note
A chance node needs a distribution for every combination of its parents, so
``report`` must take a value even when ``test`` is ``no_test`` or ``nothing``.
We give it a flat filler — one third to each report — and because ``invest``
observes ``test`` (the memory arc), it knows the filler is meaningless and
ignores it. The filler's exact probabilities never move the policy.
```

With the pieces in place, the diagram is a short, readable list of nodes. Note
that ``score`` declares no ``states``: ``states=None`` is how `pylimid` marks
a continuous variable.

```{code-cell} ipython3
diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="advanced_state",
        states=ADVANCED_STATES,
        dist=lambda: dist.Categorical(probs=advanced_prior),
    )
)
diagram.add_node(
    ChanceNode(
        name="conventional_state",
        states=CONVENTIONAL_STATES,
        dist=lambda: dist.Categorical(probs=conventional_prior),
    )
)
diagram.add_node(DecisionNode(name="test", states=TEST_STATES))
diagram.add_node(
    ChanceNode(name="score", parents=("advanced_state",), dist=score_dist)
)
diagram.add_node(
    ChanceNode(
        name="report", parents=("score", "test"), states=REPORT_STATES, dist=report_dist
    )
)
diagram.add_node(
    DecisionNode(name="invest", parents=("test", "report"), states=INVEST_STATES)
)
diagram.add_node(
    UtilityNode(
        name="utility",
        parents=("test", "invest", "advanced_state", "conventional_state"),
        values=utility,
    )
)
print(diagram.validate())
```

A diagram is a mutable workspace, so inference is gated behind an explicit
validation checkpoint. With a valid diagram, it also draws itself: circles for
chance nodes, rectangles for decisions, a diamond for the payoff. The
continuous ``score`` is the circle between ``advanced_state`` and ``report``.

```{code-cell} ipython3
diagram
```

## From a number to a category

Binning the score induces the report probabilities. There is no table to write
by hand: each entry is just the Gamma probability mass between two thresholds,
which we can read off the CDF.

```{code-cell} ipython3
def induced_report_probs(concentration):
    """P(report | advanced_state): the Gamma mass inside each score bin."""
    low, high = score_thresholds
    rows = []
    for mean in score_means:
        gamma = dist.Gamma(concentration=concentration, rate=concentration / mean)
        p_bad = gamma.cdf(jnp.array(low))
        p_good = gamma.cdf(jnp.array(high)) - p_bad
        rows.append(jnp.stack([p_bad, p_good, 1.0 - p_bad - p_good]))
    return jnp.stack(rows)


report_probs = induced_report_probs(score_concentration)

print(f"{'state':<8}{'bad':>8}{'good':>8}{'excellent':>11}")
for state, row in zip(ADVANCED_STATES, report_probs, strict=True):
    print(f"{state:<8}{row[0]:8.3f}{row[1]:8.3f}{row[2]:11.3f}")
```

The picture is easier to read than the table. The score distributions live on
one axis, and the dashed lines are the reporting thresholds.

```{code-cell} ipython3
xs = jnp.linspace(0.0, 13.0, 400)
fig, ax = plt.subplots(figsize=(8, 4))
colors = ("#2a9d8f", "#e9c46a", "#e76f51")
for mean, color, state in zip(score_means, colors, ADVANCED_STATES, strict=True):
    gamma = dist.Gamma(concentration=score_concentration, rate=score_concentration / mean)
    ax.plot(xs, jnp.exp(gamma.log_prob(xs)), color=color, label=state)
for threshold in score_thresholds:
    ax.axvline(threshold, color="0.4", linestyle="--", linewidth=1)
ax.set_xlabel("performance score")
ax.set_ylabel("density")
ax.set_title("The continuous score, and the two reporting thresholds")
ax.legend(title="advanced system")
```

The same information as bars:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 3.5))
positions = jnp.arange(len(ADVANCED_STATES))
width = 0.25
report_colors = ("#e76f51", "#e9c46a", "#2a9d8f")  # bad, good, excellent
for index, (report, color) in enumerate(zip(REPORT_STATES, report_colors, strict=True)):
    ax.bar(positions + (index - 1) * width, report_probs[:, index], width, label=report, color=color)
ax.set_xticks(positions)
ax.set_xticklabels(ADVANCED_STATES)
ax.set_ylabel("probability")
ax.set_title("P(report | advanced_state)")
ax.legend()
```

A smooth system is almost never reported ``bad``; a major-failure-prone system
is almost never reported ``excellent``. The test has real signal.

## Solving it

`solve()` searches the policies and returns the best one with its expected
utility. Because this diagram is *solvable* — each decision can be resolved
once the later ones are fixed — the solver uses backward induction
automatically; the exhaustive scan is the fallback for harder diagrams. Both
handle the continuous ``score`` by Monte-Carlo, exactly as they handle any
other chance node.

```{code-cell} ipython3
solution = solve(diagram, num_samples=400_000)
print("method:", solution.method)
print("expected utility: €{:.1f}M".format(solution.expected_utility))
print("run the test?", TEST_STATES[solution.policy["test"][()]])
for info, action in solution.policy["invest"].items():
    if info[0] == 0:  # the branch where the test actually ran
        print(f"  report = {REPORT_STATES[info[1]]:<9} -> invest {INVEST_STATES[action]}")
```

The strategy: **run the test**, invest in the **advanced** system on a
``good`` or ``excellent`` report, and fall back to the **conventional** system
on a ``bad`` one. The expected utility is a Monte-Carlo estimate, so it
wobbles a little around the exact answer we compute next.

## Reading the recommendation

Why this policy? The solver already found it, but the arithmetic is small
enough to do by hand — and doing it is the best way to trust the solver. Bayes
gives the posterior over the advanced state for each report, and the posterior
mixes the payoff table into an expected utility per action.

```{code-cell} ipython3
report_marginal = advanced_prior @ report_probs
state_posterior = report_probs * advanced_prior[:, None] / report_marginal[None, :]

tested_advanced = state_posterior.T @ advanced_payoff[0]
tested_conventional = conventional_prior @ conventional_payoff[0]
best_after_test = jnp.maximum(tested_advanced, tested_conventional)
value_of_testing = float((report_marginal * best_after_test).sum())

value_of_skipping = float((advanced_prior * advanced_payoff[1]).sum())

print("P(report):                    ", jnp.round(report_marginal, 3))
print("EU | advanced, by report:     ", jnp.round(tested_advanced, 1))
print("EU | conventional (constant): ", round(float(tested_conventional), 1))
print("value of testing:             ", round(value_of_testing, 2))
print("value of skipping the test:   ", round(value_of_skipping, 2))
```

Three facts fall out of the table:

- A ``bad`` report is bad news: the posterior piles onto major failure, the
  advanced system is worth about **−108**, and the conventional system at
  **45.5** wins.
- A ``good`` report makes the advanced system worth about **50.7** — just past
  the conventional system. The score is sharp enough that even a middling
  report flips the decision.
- Reporting ``excellent`` is decisive: the posterior is almost surely smooth,
  and the advanced system is worth about **137.6**.

Weighting those branches by how often the report appears gives **94.46**: the
value of testing. Skipping the test and investing in the advanced system is
worth **85**, and doing nothing is worth **0**. So the €10M test earns its
keep many times over.

This exact arithmetic is the reference the Monte-Carlo solver is measured
against; with enough samples its estimate lands right on 94.46.

## Is the test worth €10M?

A sharper test gives a better report, but sharpness is not free in reality —
and the model makes the trade-off easy to explore. We can rerun the exact
calculation above for a range of concentrations: higher concentration means a
tighter score around each state's mean, hence a more informative report.

```{code-cell} ipython3
concentrations = jnp.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])


def value_of_testing_at(concentration):
    probs = induced_report_probs(concentration)
    marginal = advanced_prior @ probs
    posterior = probs * advanced_prior[:, None] / marginal[None, :]
    advanced = posterior.T @ advanced_payoff[0]
    return float((marginal * jnp.maximum(advanced, tested_conventional)).sum())


test_values = jnp.array([value_of_testing_at(c) for c in concentrations])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(concentrations, test_values, "o-", color="#e76f51", label="value of testing")
ax.plot(
    concentrations,
    jnp.maximum(test_values, value_of_skipping),
    "-",
    color="#1d3557",
    linewidth=2,
    label="best expected utility",
)
ax.axhline(value_of_skipping, color="0.4", linestyle="--", linewidth=1)
ax.text(0.6, value_of_skipping + 1.0, "skip the test: invest advanced (85)", color="0.3")
ax.set_xscale("log")
ax.set_xlabel("score concentration (sharper test →)")
ax.set_ylabel("€ million")
ax.set_title("When is the test worth its €10M?")
ax.legend()
```

Two regimes. When the score is fuzzy (concentration below about 3.5), the
test cannot separate smooth from failing, so the best it can do is value the
advanced system at 75 — and since the test itself costs €10M, testing is
*worse* than committing blind at 85. The company would skip it. Once the score
is sharp enough, testing pulls ahead, and the gap keeps growing: a truly
precise test is worth well over €100M.

That is the kind of question the continuous model answers and a three-row
report table cannot: not just *should we pay for information*, but *how good
does the information have to be*.

## What a bad report tells us about the score

The last piece is the one a decision-only solver would hide: the mixed
inference. Once the decisions are bound, `infer()` runs on the same graph.
Looking at the score behind a ``bad`` report mixes the continuous posterior
with the discrete one.

```{code-cell} ipython3
belief = infer(
    diagram,
    ["advanced_state", "score"],
    observed={"report": 0},
    policy={"test": 0, "invest": 1},
)
print("P(advanced_state | report = bad):", [round(p, 3) for p in belief["advanced_state"].marginal()])
print("score | report = bad: mean {:.2f}, std {:.2f}".format(belief["score"].mean(), belief["score"].std()))
```

The draws for the continuous score can be plotted directly, against the
posterior mixture they came from:

```{code-cell} ipython3
draws = belief["score"].values
weights = belief["advanced_state"].marginal()

grid = jnp.linspace(0.0, 7.0, 400)
mixture = jnp.zeros_like(grid)
for weight, mean in zip(weights, score_means, strict=True):
    gamma = dist.Gamma(concentration=score_concentration, rate=score_concentration / mean)
    mixture = mixture + weight * jnp.exp(gamma.log_prob(grid))
# A bad report is exactly the event score < 4, so the posterior is the
# mixture truncated there (and renormalized).
mixture = jnp.where(grid < score_thresholds[0], mixture, 0.0)
mixture = mixture / (mixture.sum() * (grid[1] - grid[0]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(draws, bins=60, density=True, color="#457b9d", alpha=0.5, label="posterior draws")
ax.plot(grid, mixture, color="#1d3557", label="posterior density")
ax.axvline(score_thresholds[0], color="0.4", linestyle="--", linewidth=1)
ax.set_xlabel("performance score")
ax.set_ylabel("density")
ax.set_title("The score behind a bad report")
ax.legend()
```

A ``bad`` report concentrates the state posterior on ``major`` and drags the
score down to a mean of about 3 — the low-performing region. The draws stop
at 4 because a ``bad`` report *is* the event ``score < 4``; the posterior is
just the score conditioned on it. Discrete and continuous chance variables
inferred together, from one graph.

```{admonition} Next
:class: seealso
The oil-field tutorial covers a fully discrete diagram and the memory-arc
mechanics in more depth. [The newsvendor](newsvendor.md) shows another
mixed-type pattern, and [Bayesian networks](bayesian_networks.md) covers
inference without decisions. For the exact reference here: pyAgrum cannot
solve a mixed LIMID, but integrating the continuous score out into its
induced report distribution leaves a discrete model it can solve exactly.
```
