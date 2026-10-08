# Draft: adding a Bayesian prior to the quickstart

Working notes for extending `docs/getting_started/quickstart.md` with a Bayesian prior. This folder is excluded from the Sphinx build (see `exclude_patterns` in `docs/conf.py`), so nothing here is published. The code blocks are plain `python` fences; turn them into `{code-cell} ipython3` when moving them into the quickstart.

All numbers below come from running this code with `num_samples=50_000` and the default seed.

## The idea

In the current quickstart, the vendor knows the average demand exactly: 40 ice creams on a cloudy day and 120 on a sunny one. In reality those figures come from their experience, and they can't be sure of them. A Bayesian prior lets the model say so: instead of a fixed number, the overall level of demand becomes a chance node of its own, with a distribution that describes the vendor's belief.

This is a good fit for the quickstart because it shows three things at once:

- a prior is just another chance node, so it needs no new API;
- uncertainty about the model changes the best decision, which the reader can see by comparing with the first answer;
- `infer` has a natural job, which is to update the vendor's belief after a day of sales.

## Which prior to use

**Use a prior on demand, not on the weather.** A prior on the weather probabilities, for example a `Beta(3, 7)` on the chance of sun, leaves the answer exactly as it is: with a single forecast, only the average belief matters, and `Beta(3, 7)` averages 0.3, the same as the fixed forecast. It would show a prior that does nothing, so it is the wrong first example.

A prior on demand does change the answer. We add a `busyness` node: a multiplier on both average demands, so `busyness = 1.2` means a season 20% busier than the vendor's figures. The vendor believes their figures are right on average but could be off by about 30% either way, which a log-normal distribution expresses well because the multiplier must be positive.

**Centre the multiplier on 1.** `LogNormal(0, 0.3)` averages about 1.05, not 1, so it would silently raise average demand by 5% and mix two effects. `LogNormal(-0.3**2 / 2, 0.3)` averages exactly 1, so the comparison only shows the effect of being uncertain.

**Widen the stock menu to 300.** With the prior, the best sunny-day stock is 220, which a menu that stops at 200 can't offer: the solver would return 200, the largest option, and the result would look as if the prior changed nothing on sunny days. A menu from 0 to 300 in steps of 20 leaves room above every answer. The first part of the quickstart gives the same answer (60 and 200) with this menu, so it can be widened everywhere.

## Changes to the model

Only the demand node changes, and one node is added:

```python
ORDERS = jnp.arange(0, 301, 20)
BUSYNESS_SD = 0.3


diagram.add_node(
    ChanceNode(
        name="busyness",
        dist=lambda: dist.LogNormal(-(BUSYNESS_SD**2) / 2, BUSYNESS_SD),
    )
)


def demand_dist(weather, busyness):
    mean = busyness * DEMAND_MEANS[weather]
    return dist.Gamma(concentration=4.0, rate=4.0 / mean)


diagram.add_node(
    ChanceNode(name="demand", parents=("weather", "busyness"), dist=demand_dist)
)
```

The decision keeps `weather` as its only parent: the vendor knows the forecast when ordering, but not how busy the season really is. That is also what keeps the diagram solvable, because a decision can only observe discrete parents.

There are two ways to fit this into the page:

1. **A new section after "Solving the model"**, for example "Adding the vendor's experience", which rebuilds the diagram with the two extra lines and compares the answers. This keeps the first model as simple as it is today, and is the option I would pick.
2. **Build it in from the start.** Shorter, but the reader meets the prior before seeing the basic workflow, and loses the before-and-after comparison.

## What the reader sees

Solving both versions with the 0–300 menu:

| Model | Cloudy day | Sunny day | Expected profit |
| --- | --- | --- | --- |
| Known average demand | stock 60 | stock 200 | 507.1 |
| Uncertain average demand | stock 80 | stock 220 | 487.5 |

Being unsure about demand makes the vendor stock more on both kinds of day, because a busier season than expected is exactly when running out costs the most. The expected profit is lower, too: about \$20 a day is the price of not knowing the market precisely.

## Updating the belief with `infer`

After a sunny day, the vendor counts how many ice creams they could have sold. `infer` updates their belief about `busyness` from that one observation:

```python
result = infer(
    diagram,
    ["busyness"],
    observed={"weather": 1, "demand": 180.0},
    policy={"order": 0},
)
result["busyness"].mean()
```

| Evidence | Mean `busyness` | 95% interval |
| --- | --- | --- |
| None (the prior) | 0.99 | 0.51 to 1.57 |
| Sunny day, 180 ice creams demanded | 1.14 | 0.63 to 1.70 |
| Sunny day, 60 ice creams demanded | 0.87 | 0.45 to 1.34 |

A busy sunny day nudges the belief up and a quiet one nudges it down, but one day is weak evidence, so the interval stays wide.

The intervals come from `result["busyness"].hdi()`, whose default covers 95% of the draws.

The `policy` argument needs a sentence of explanation on the page. `infer` requires every decision to be fixed, even though the order has no effect on demand, so `{"order": 0}` works but looks arbitrary. Something like "`infer` needs every decision fixed; the order doesn't affect demand, so any value works" is enough.

## Feeding the update back into the decision

The natural next step, solving again with the updated belief, isn't possible in one call today: `solve` doesn't take observations. The vendor could do it by hand, replacing the prior with a distribution fitted to the posterior draws, but that is too much for a quickstart. Either leave it out, or mention it as a pointer to the user guide.

## Things that don't work yet

**Dirichlet priors.** A `Dirichlet` prior over the weather probabilities, the natural choice with three or more weather states, solves fine but crashes in `infer`: `_infer_numpyro` in `pylimid/inference/engine.py` converts every draw with `float()`, and a Dirichlet draw is a vector. Until that is fixed, use a `Beta` prior on the chance of a single state, and only for two-state variables. This is also tracked in `TODO.md`.

## Checklist before publishing

- [ ] Widen the menu to 0–300 in the quickstart, and check that the first answer is still 60 and 200.
- [ ] Add the new section and both tables, regenerating the numbers from the final code.
- [ ] Add a "Uncertain parameters" or "Bayesian priors" bullet to the Features in `docs/index.md` and `README.md`.
- [ ] Decide whether to fix the Dirichlet crash first, so the feature can be advertised without a caveat.
