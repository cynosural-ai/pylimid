---
file_format: mystnb
kernelspec:
  name: python3
---

# Inference

`infer(diagram, query, observed=..., policy=...)` computes posterior draws for the query variables given evidence. A Bayesian network infers directly. An influence diagram can be inferred once **every** decision is bound by a policy: each bound decision is clamped to its chosen action and behaves like observed evidence, utilities are ignored, and the diagram collapses to a Bayesian network.

## Bayesian networks

```{code-cell} ipython3
import jax.numpy as jnp
import numpyro.distributions as dist

from pylimid import ChanceNode, DecisionNode, InfluenceDiagram, UtilityNode, infer
```

A two-node diagnosis: the disease is `healthy` or `sick`, and a test lights up more often when the patient is sick. The prior comes straight from the model:

```{code-cell} ipython3
diagnosis = InfluenceDiagram()
diagnosis.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
diagnosis.add_node(
    ChanceNode(
        name="test",
        parents=("disease",),
        states=("negative", "positive"),
        dist=lambda disease: dist.Categorical(
            probs=jnp.array([[0.9, 0.1], [0.1, 0.9]])[disease]
        ),
    )
)
infer(diagnosis, ["disease"])["disease"].marginal()
```

Evidence is a name-to-value map, where the value is the **state index** for a discrete variable (or a real number for a continuous one). A positive test moves the posterior toward `sick`:

```{code-cell} ipython3
infer(diagnosis, ["disease"], observed={"test": 1})["disease"].marginal()
```

A query may name chance variables only, and observed values name chance nodes. Asking for a decision node's posterior is an error — a decision is an action, not a random variable.

## Influence diagrams need a policy

With decisions in the diagram, inference is only defined once every decision is bound. Binding is all-or-nothing: a partially bound diagram is still ambiguous, and `infer` raises `InferenceError` listing the unbound decisions. Here the treatment is clamped to `yes` (index `1`), and we ask for the recovery posterior under that fixed action:

```{code-cell} ipython3
medical = InfluenceDiagram()
medical.add_node(
    ChanceNode(
        name="disease",
        states=("healthy", "sick"),
        dist=lambda: dist.Categorical(probs=jnp.array([0.6, 0.4])),
    )
)
medical.add_node(
    DecisionNode(name="treat", parents=("disease",), states=("no", "yes"))
)
medical.add_node(
    ChanceNode(
        name="recovery",
        parents=("disease", "treat"),
        states=("no", "yes"),
        dist=lambda disease, treat: dist.Categorical(
            probs=jnp.array(
                [
                    [[0.10, 0.90], [0.08, 0.92]],  # healthy
                    [[0.60, 0.40], [0.20, 0.80]],  # sick
                ]
            )[disease, treat]
        ),
    )
)
medical.add_node(
    UtilityNode(
        name="benefit", parents=("recovery",), values=lambda recovery: 100.0 * recovery
    )
)

infer(medical, ["recovery"], policy={"treat": 1})["recovery"].marginal()
```

`policy` clamps each decision to a single action — the same action in every realization. It is an interface for *interventions* ("what if everyone were treated?"), not for evaluating the state-dependent rule that `solve` returns. To compare a few fixed policies, call `infer` once per policy; for the conditional rule, the solvers are the tool.

## The Posterior object

Every query returns a `Posterior`: the raw `values` (one draw per entry) plus the variable's `states` (`None` for a continuous variable). The summary methods derive from the draws, and each is valid for exactly one kind of variable:

- `marginal()` — the discrete probability vector, bincounted from the draws.
- `mean()` / `std()` / `hdi()` — continuous summaries. `hdi()` defaults to 95% coverage and takes a `prob` argument.

Calling the wrong kind raises rather than computing something encoding-dependent:

```{code-cell} ipython3
continuous = InfluenceDiagram()
continuous.add_node(ChanceNode(name="x1", dist=lambda: dist.Normal(0.0, 1.0)))
continuous.add_node(
    ChanceNode(
        name="x2",
        parents=("x1",),
        dist=lambda x1: dist.Normal(loc=2.0 + 1.5 * x1, scale=0.5),
    )
)
posterior = infer(continuous, ["x1"], observed={"x2": 3.5})
x1 = posterior["x1"]
(round(x1.mean(), 2), tuple(round(v, 2) for v in x1.hdi()))
```

## The engine and the Monte-Carlo contract

There is one engine, and it chooses how to draw:

- **All-discrete diagrams** use exact discrete enumeration: the joint is enumerated in parallel and the draws are bincounted. The distribution is exact; the returned *draws* are a finite sample.
- **Anything with a continuous latent** uses NUTS on the continuous variables, with the discrete sites enumerated out and resampled conditional on the continuous posterior.

Either way, the result is a Monte-Carlo estimate delivered as draws — there is no exactness flag and no closed-form result type. If you need a smoother estimate, ask for more draws (`infer` does not expose `num_samples`; the underlying `pylimid.inference.numpyro.samples` does).
