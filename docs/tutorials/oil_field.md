---
file_format: mystnb
kernelspec:
  name: python3
---

# The oil field

This tutorial takes a classic decision problem and models it as a **limited-memory influence diagram (LIMID)** with `pylimid`. We start with the story, write down the numbers, draw the diagram, and let `solve()` find the best strategy — then check the answer against the arithmetic.

## The problem

Imagine you are the CEO of an oil company and you are offered the opportunity to purchase a new oil field. Like any diligent executive, you gather your team to examine satellite images, geological surveys, and consult the team's experience with similar fields in the region.

This research helps form your initial assessment of the field's potential. However, until you actually start drilling, there's no way to know for sure how much oil is down there. This leaves you with a classic dilemma: play it safe and walk away, or make a calculated bet based on the evidence you've gathered.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
```{image} /_static/figures/oil_field/oil_field_image.jpg
:alt: Satellite image of an oil field
:width: 100%
```
:::

:::{grid-item}
```{image} /_static/figures/oil_field/oil_field_heatmap.jpg
:alt: Heatmap of an oil field
:width: 100%
```
:::

::::

*Satellite imagery and heatmap of the oil field ([source](https://www.satimagingcorp.com/applications/energy/exploration/oil-exploration/)).*

This tension between what you know, what you don't know, and what you truly value is at the heart of every important decision. Decision theory offers a framework to answer exactly these questions: it looks at the likelihood of different scenarios and weighs them against how much you value each potential result, guiding you toward the action that is most likely to give you the best overall outcome.

Suppose you have sized up the field and think it could fall into one of three quality levels:

- **High**, promising oil reserves.
- **Medium**, a decent output but nothing groundbreaking.
- **Low**, where you would barely break even or could even end up losing money.

The dilemma becomes clear. If you buy and the field turns out to be high quality, you have struck gold. If you buy and it is low quality, you lose your investment. But if you choose not to buy, you avoid the loss — though you might also miss out on a good opportunity.

## What we know

At its core, any decision problem breaks down into a few components: the **actions** available, the uncertain conditions that influence the outcome (the **states of nature**), the **probabilities** assigned to those conditions, and the **outcomes** for each action under each condition.

- **Actions:** buy the field, investing capital now in the hope of future profits, or do not buy, avoiding the risk but potentially missing the gains.
- **States of nature:** the true quality of the field, which we have simplified into *high*, *medium*, and *low*.
- **Probabilities:** based on the geological data and expert opinions, a 35% chance of high quality, 45% for medium, and 20% for low.
- **Outcomes:** the financial implications, in millions of dollars, for every scenario.

| Action | Field Quality | Monetary Outcome |
| --- | --- | --- |
| Buy | High | +$1250M |
| Buy | Medium | +$630M |
| Buy | Low | +$0M |
| Do not buy | Any | +$350M |

```{admonition} Why is there a profit for not buying?
:class: note
This represents a baseline alternative: investing the capital elsewhere for a guaranteed return of $350M.
```

A common way to weigh these possibilities is the **expected monetary value** (EMV). For "buy", each outcome is weighted by its likelihood:

$$
\mathbb{E}[MV(\text{Buy})] = 0.35 \cdot 1250 + 0.45 \cdot 630 + 0.20 \cdot 0 = 721.
$$

For "do not buy" it is simply $350M. Purely from an EMV perspective, buying looks like the stronger choice, $721M versus $350M.

For the rest of this tutorial we assume the decision maker is **risk neutral**, so monetary values and utilities coincide: a utility of 1000 corresponds to $1000M. Accounting for risk preferences is a subject for another day — the mechanics of `pylimid` do not change, only the numbers you feed the utility node.

## The option to test

Just as the company is about to move forward, a new opportunity arises: it can perform a geological test before deciding whether to purchase the field. The test does not reveal the true quality — it measures the rock's **porosity**, the empty space that could hold oil — and reports either **pass** (porosity ≥ 15%) or **fail** (porosity < 15%).

```{figure} /_static/figures/oil_field/rock_porosity.jpg
:alt: Reservoir quality illustrated through porosity and permeability
:width: 70%

Reservoir quality illustrated through porosity and permeability characteristics ([source](https://www.parliament.wa.gov.au/publications/tabledpapers.nsf/displaypaper/3913541ae03e783bf52cf5b948257ee5000a9d20/$file/3541.pdf)).
```

In an ideal world the test would be perfectly accurate. Real tests are not. The table below gives the probability of each result for each true quality:

|  | high | medium | low |
| --- | --- | --- | --- |
| pass | 0.95 | 0.70 | 0.15 |
| fail | 0.05 | 0.30 | 0.85 |

**The test costs $30M**, and after seeing the result the company still has to decide whether to buy. There are now two decisions: first whether to test, then whether to purchase on the basis of the result. This is a sequential problem, and to represent it well we will use an influence diagram.

## Modelling the problem

An **influence diagram** extends a Bayesian network with two node types: **decision nodes** (squares) and **utility nodes** (diamonds), alongside the **chance nodes** (circles). Arcs into decision nodes are *informational* — they say what is known when the choice is made — while arcs into chance and utility nodes express probabilistic or functional dependence.

The traditional influence diagram assumes **perfect recall**: a later decision implicitly remembers every earlier decision and observation. A **LIMID** drops that assumption and makes memory explicit. Here the buy decision must know whether the test was actually run, so we add a **memory arc** from the test decision to the buy decision:

```{figure} /_static/figures/oil_field/5_limid_oil.png
:alt: LIMID for the oil field decision problem
:width: 60%

LIMID for the oil field decision problem. Informational arcs are dashed; the green arc is the memory arc. The buy decision observes the test result *and* remembers whether the test was run ([source](https://ferjorosa.github.io/blog/2025/07/04/decision-theory-II.html)).
```

The diagram has four nodes plus a utility:

- **Quality** (`quality`): chance, the field's true quality, with the prior $(0.35, 0.45, 0.20)$.
- **Test** (`test`): decision, *do* or *not do*, with no parents. Nothing is observed before this choice.
- **Result** (`result`): chance, *pass* or *fail*, depending on `quality` and on whether the test ran.
- **Buy** (`buy`): decision, *buy* or *not buy*, observing `result` and remembering `test`.
- **Utility** (`utility`): the payoff, a function of `quality`, `test`, and `buy`.

The quantitative part is small enough to write as tables. The prior over quality, the conditional table for the result, and the utility:

| P(quality) | high | medium | low |
| --- | --- | --- | --- |
|  | 0.35 | 0.45 | 0.20 |

| P(result \| quality, test) | do, high | do, medium | do, low | not do, high | not do, medium | not do, low |
| --- | --- | --- | --- | --- | --- | --- |
| pass | 0.95 | 0.70 | 0.15 | 0.50 | 0.50 | 0.50 |
| fail | 0.05 | 0.30 | 0.85 | 0.50 | 0.50 | 0.50 |

| test | buy | quality | utility |
| --- | --- | --- | --- |
| do | buy | high | 1220 |
| do | buy | medium | 600 |
| do | buy | low | -30 |
| do | not buy | any | 320 |
| not do | buy | high | 1250 |
| not do | buy | medium | 630 |
| not do | buy | low | 0 |
| not do | not buy | any | 350 |

```{admonition} The result when the test isn't run
:class: note
A chance node needs a distribution for *every* combination of its parents, so `result` must take a value even when `test = not_do`. We give it the same flat pass/fail filler for every quality — independent of `quality`, so it leaks nothing — and since `buy` observes `test` (the memory arc), it knows the filler is meaningless and ignores it. The chosen action is identical for `pass` and `fail`.

That full table also keeps all four `(test, result)` information sets reachable. Backward induction estimates one continuation value per information-set cell by forward sampling, and a cell that can never occur leaves it with nothing to estimate. The common pyAgrum idiom — a `no_results` state that is certain when `test = not_do` — makes half the cells impossible, so `solve()` would fall back to the scan. A well-defined filler keeps the automatic path, and the filler's exact probabilities never affect the policy.
```

The whole model is a few lines of `pylimid`. Chance distributions are plain NumPyro callables, and the utility is an ordinary function of its parents:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, solve

quality_prior = jnp.array([0.35, 0.45, 0.20])
reward = jnp.array([1250.0, 630.0, 0.0])
# Rows: test = do, not_do. Columns: quality = high, medium, low.
result_probs = jnp.array([
    [[0.95, 0.05], [0.70, 0.30], [0.15, 0.85]],
    [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]],
])
```

```{code-cell} ipython3
diag = InfluenceDiagram()
diag.add_node(
    ChanceNode(
        name="quality",
        states=("high", "medium", "low"),
        dist=lambda: dist.Categorical(probs=quality_prior),
    )
)
diag.add_node(DecisionNode(name="test", states=("do", "not_do")))
diag.add_node(
    ChanceNode(
        name="result",
        parents=("quality", "test"),
        states=("pass", "fail"),
        dist=lambda quality, test: dist.Categorical(probs=result_probs[test, quality]),
    )
)
diag.add_node(
    DecisionNode(name="buy", parents=("test", "result"), states=("buy", "not_buy"))
)


def utility(quality, test, buy):
    """Payoff: the field's reward, minus the test cost when the test runs."""
    payoff = jnp.where(buy == 0, reward[quality], 350.0)
    cost = jnp.where(test == 0, 30.0, 0.0)
    return payoff - cost


diag.add_node(
    UtilityNode(name="utility", parents=("quality", "test", "buy"), values=utility)
)
```

A diagram is a mutable workspace, so inference is gated behind an explicit validation checkpoint:

```{code-cell} ipython3
diag.validate()
```

With a valid diagram we can draw it — chance nodes as circles, decisions as rectangles, the utility as a diamond:

```{code-cell} ipython3
diag
```

## Solving it

`solve()` returns the optimal policy and the expected utility it achieves. Because the diagram is *solvable* — every decision can be resolved from its information set once the later decisions are fixed — `solve` automatically uses backward induction rather than enumerating every policy:

```{code-cell} ipython3
solution = solve(diag, num_samples=20000)
solution.method
```

The policy maps each decision to an action per information-set assignment, and the expected utility is the value of following it:

```{code-cell} ipython3
solution.policy
```

```{code-cell} ipython3
round(solution.expected_utility, 1)
```

Reading the policy: the test key `()` is the empty information set, and `1` selects `not_do`. The buy policy is indexed by `(test, result)` in parent order, so `(0, 0)` is "the test ran and passed" and `(1, 0)` is "the test did not run". The recommended strategy is to **skip the test and buy the field**.

The expected utility is a Monte-Carlo estimate. With the default 2000 samples it lands within roughly ±10 of the truth at this scale, which is why we asked for 20000; here it came out at 716, about 5 below the exact 721. The seed is fixed, so the number is reproducible; the next section computes the exact value.

## Reading the recommendation

Why is testing not worth it? We can evaluate the strategy the solver found by hand. If the company tests and then buys on a pass, does not buy on a fail:

```{code-cell} ipython3
p_pass_given_quality = jnp.array([0.95, 0.70, 0.15])

p_pass = float((quality_prior * p_pass_given_quality).sum())
posterior = quality_prior * p_pass_given_quality / p_pass
e_pass = float((posterior * jnp.array([1220.0, 600.0, -30.0])).sum())
e_test = p_pass * e_pass + (1 - p_pass) * 320.0
e_no_test = float((quality_prior * reward).sum())

round(e_test, 2), round(e_no_test, 2)
```

Testing has an expected value of **696.95** against **721** for buying outright, so the company should not test. The test's information is worth something — a free test would be worth $726.95, about $5.95M of information — but not the $30M it costs. It only changes the decision in the "fail" branch, and that branch is not likely enough, nor the loss it avoids large enough, to pay for the test.

`pylimid` reached the same conclusion without the manual Bayes calculation, and it would reach it for diagrams far too large to evaluate by hand.

```{admonition} Next
:class: seealso
A follow-up tutorial varies the test's accuracy and cost, and the quality prior, to find the thresholds where testing becomes worthwhile — and shows what the Monte-Carlo error looks like along the way.
```
