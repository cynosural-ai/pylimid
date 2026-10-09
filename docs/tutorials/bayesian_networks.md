---
file_format: mystnb
kernelspec:
  name: python3
---

# Bayesian networks

A **Bayesian network** is an influence diagram with only chance nodes: no decisions, no payoffs, just variables that depend on one another. `pylimid` queries it with `infer()`, the same engine that backs the decision tutorials — exact discrete enumeration when every variable is discrete, NUTS on the continuous ones otherwise.

This tutorial runs two networks end to end. The first is fully discrete and small enough to check against the arithmetic by hand: a diagnosis where two causes compete to explain one symptom. The second is mixed — a discrete cause, a hidden continuous state, and a noisy sensor — and shows what changes when the engine leaves enumeration behind, which is mostly that the answer becomes a Monte-Carlo estimate rather than an exact number.

Both are queried with the same call, `infer(diagram, query, observed=...)`. The [inference guide](../user_guide/inference.md) covers the API in full, and [the quickstart](../getting_started/quickstart.md) shows what happens when a decision enters the diagram.

## A discrete network

### The model

Suppose smoking and air pollution both raise the risk of cancer, and an x-ray may reveal it. Four chance nodes, three arcs: two independent causes, a shared effect with two parents, and one noisy observation of that effect.

The conditional tables come first, one per node. `CANCER` is indexed by both parents — `CANCER[smoking, pollution]` — so it holds one probability row per combination:

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, InfluenceDiagram, infer
```

```{code-cell} ipython3
SMOKING = jnp.array([0.7, 0.3])
POLLUTION = jnp.array([0.9, 0.1])
CANCER = jnp.array(
    [
        [[0.99, 0.01], [0.40, 0.60]],  # smoking = no
        [[0.90, 0.10], [0.50, 0.50]],  # smoking = yes
    ]
)
XRAY = jnp.array([[0.80, 0.20], [0.10, 0.90]])
```

Each node pairs a `states` tuple with a `dist` callable of its parents. The callable index looks up the right row of the table:

```{code-cell} ipython3
diagram = InfluenceDiagram()
diagram.add_node(
    ChanceNode(
        name="smoking",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=SMOKING),
    )
)
diagram.add_node(
    ChanceNode(
        name="pollution",
        states=("low", "high"),
        dist=lambda: dist.Categorical(probs=POLLUTION),
    )
)
diagram.add_node(
    ChanceNode(
        name="cancer",
        parents=("smoking", "pollution"),
        states=("absent", "present"),
        dist=lambda smoking, pollution: dist.Categorical(
            probs=CANCER[smoking, pollution]
        ),
    )
)
diagram.add_node(
    ChanceNode(
        name="xray",
        parents=("cancer",),
        states=("neg", "pos"),
        dist=lambda cancer: dist.Categorical(probs=XRAY[cancer]),
    )
)
print(diagram.validate())
```

A validated diagram draws itself — circles for chance nodes, arrows for dependence:

```{code-cell} ipython3
diagram
```

### The prior and the evidence

One call returns a `Posterior` per query variable; `marginal()` bincounts the draws into a probability vector. With no evidence the draws come straight from the joint distribution:

```{code-cell} ipython3
prior = infer(diagram, ["smoking", "pollution", "cancer", "xray"])
for name, posterior in prior.items():
    print(f"{name:>9}: {[round(p, 3) for p in posterior.marginal()]}")
```

Evidence is a name-to-value map, and a discrete value is its **state index**: `{"xray": 1}` is ``xray = pos``. A positive x-ray raises the belief in cancer and, one step up the arrows, in both of its causes:

```{code-cell} ipython3
posterior = infer(diagram, ["smoking", "pollution", "cancer"], observed={"xray": 1})
for name, p in posterior.items():
    print(f"{name:>9}: {[round(x, 3) for x in p.marginal()]}")
```

### Explaining away

What makes the network more than four independent marginals shows up when a second cause is revealed. Given only the positive x-ray, cancer is likely enough that it lifts the odds on both causes. Now learn that pollution was high: pollution alone can explain the x-ray, so the belief in smoking **drops**. The two causes are independent in the model — nothing connects them — yet they become coupled once their shared effect is observed. This is explaining away, and it is the everyday reason to query a joint distribution instead of the pieces:

```{code-cell} ipython3
for evidence in ({"xray": 1}, {"xray": 1, "pollution": 1}):
    smoking = infer(diagram, ["smoking"], observed=evidence)["smoking"]
    print(f"{str(evidence):<32} -> P(smoking=yes) {smoking.marginal()[1]:.3f}")
```

### The exact answer

Every variable is discrete, so the engine does not approximate the posterior: it enumerates the joint distribution and hands back draws from the exact answer. Here is that enumeration in longhand, in four lines: multiply the tables, slice the evidence, renormalize. The `joint` array is indexed ``[smoking, pollution, cancer, xray]``, so the evidence slices are `axis=3` for `xray` and `axis=1` for `pollution`:

```{code-cell} ipython3
joint = (
    SMOKING[:, None, None, None]
    * POLLUTION[None, :, None, None]
    * CANCER[:, :, :, None]
    * XRAY[None, None, :, :]
)


def exact_posterior(joint, evidence):
    for axis in sorted(evidence, reverse=True):
        joint = jnp.take(joint, evidence[axis], axis=axis)
    return joint / joint.sum()
```

The longhand posterior and the sampled `marginal()` agree to sampling noise, and the exact probability of smoking falls from 0.34 to 0.28 when pollution is added:

```{code-cell} ipython3
positive = exact_posterior(joint, {3: 1})
explained = exact_posterior(joint, {1: 1, 3: 1})
print("sampled P(cancer=present | xray=pos):          ", round(posterior["cancer"].marginal()[1], 3))
print("exact   P(cancer=present | xray=pos):          ", round(float(positive.sum(axis=(0, 1))[1]), 3))
print("exact   P(smoking=yes | xray=pos):             ", round(float(positive.sum(axis=(1, 2))[1]), 3))
print("exact   P(smoking=yes | xray=pos, pollution=hi):", round(float(explained.sum(axis=1)[1]), 3))
```

The small gap between the sampled and exact columns is Monte-Carlo noise from the 2000 draws the engine returns, not approximation in the inference itself.

## A mixed network

### The model

Now a model that cannot be a table. A soil sensor reports one number, and we want to know whether it rained — but the reading is noisy, and what it actually measures is the soil moisture, which is itself continuous. Three nodes: a discrete cause, a continuous hidden state, and a continuous observation of it.

| rained | P | moisture mean | moisture sd |
| --- | --- | --- | --- |
| no | 0.6 | 35 | 10 |
| yes | 0.4 | 55 | 10 |

The sensor adds its own Gaussian noise around the true moisture:

```{code-cell} ipython3
P_RAINED = 0.4
MOISTURE_MEAN = jnp.array([35.0, 55.0])
MOISTURE_SD = 10.0
SENSOR_SD = 5.0

network = InfluenceDiagram()
network.add_node(
    ChanceNode(
        name="rained",
        states=("no", "yes"),
        dist=lambda: dist.Categorical(probs=jnp.array([1.0 - P_RAINED, P_RAINED])),
    )
)
network.add_node(
    ChanceNode(
        name="moisture",
        parents=("rained",),
        dist=lambda rained: dist.Normal(loc=MOISTURE_MEAN[rained], scale=MOISTURE_SD),
    )
)
network.add_node(
    ChanceNode(
        name="sensor",
        parents=("moisture",),
        dist=lambda moisture: dist.Normal(loc=moisture, scale=SENSOR_SD),
    )
)
print(network.validate())
```

The `dist` callables are the same shape as before — the discrete node indexes a mean, the continuous one shifts the location — but the variables are no longer all discrete.

### Inverting the model

The interesting query runs backwards through the arrows: observe the sensor and ask about the rain. A reading of 60 is well above the dry mean of 35 and close to the rainy mean of 55, so the prior probability of rain should rise sharply:

```{code-cell} ipython3
X = 60.0
posterior = infer(network, ["rained", "moisture"], observed={"sensor": X})
print("P(rained=yes | sensor=60):", round(posterior["rained"].marginal()[1], 3))
print("moisture mean:            ", round(posterior["moisture"].mean(), 2))
print("moisture 95% HDI:         ", tuple(round(v, 2) for v in posterior["moisture"].hdi()))
```

One variable changed the engine's behavior. `moisture` is continuous and unobserved, so NUTS samples it; `rained` never leaves the discrete space, so it is enumerated out of NUTS and resampled conditional on the moisture draws. The output is therefore a Monte-Carlo estimate rather than the exact enumeration of the first half — and `rained` cannot be summarized with `mean()`, because averaging state indices is not a probability. That is what `marginal()` is for.

### The exact answer

The mixed model is still small enough to solve with a pen. Sensor given rain is a Gaussian once moisture is marginalized out:

$$
\text{sensor} \mid \text{rained}=r \;\sim\; \mathcal{N}\!\left(m_r,\ \sqrt{\sigma_m^2 + \sigma_s^2}\right).
$$

Weight those two densities by the prior, renormalize, and the posterior probability of rain is exact. The moisture posterior is a two-component Gaussian mixture: given rain, sensor, and moisture are jointly Gaussian, so each component's mean shrinks toward the reading by the factor $k = \sigma_m^2 / (\sigma_m^2 + \sigma_s^2)$:

```{code-cell} ipython3
marginal_sd = jnp.sqrt(MOISTURE_SD**2 + SENSOR_SD**2)
sensor_density = jnp.exp(dist.Normal(MOISTURE_MEAN, marginal_sd).log_prob(X))
exact_rained = jnp.array([1.0 - P_RAINED, P_RAINED]) * sensor_density
exact_rained = exact_rained / exact_rained.sum()

shrink = MOISTURE_SD**2 / (MOISTURE_SD**2 + SENSOR_SD**2)
component_means = MOISTURE_MEAN + shrink * (X - MOISTURE_MEAN)
exact_moisture = float(jnp.sum(exact_rained * component_means))

print("exact P(rained=yes | sensor=60):", round(float(exact_rained[1]), 3))
print("exact moisture mean:            ", round(exact_moisture, 2))
```

The reading lifts P(rained) from the 0.4 prior to about 0.88, and pulls the true moisture to just under 59 — and the NUTS run lands on the same numbers. Sampling noise apart, the two halves of this tutorial are one operation: slice the joint at the observed value, renormalize, read off the query.

```{admonition} Next
:class: seealso
The [inference guide](../user_guide/inference.md) covers the `Posterior` object, evidence under a bound policy, and the engine contract in full. To add a decision to a diagram like these — and call `solve()` instead of `infer()` — continue with [the quickstart](../getting_started/quickstart.md), [the oil field](oil_field.md), or [the newsvendor](newsvendor.md).
```
