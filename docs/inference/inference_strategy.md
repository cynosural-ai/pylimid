# Inference strategy for influence diagrams

This documents the reasoning behind the choice of inference backends for
`decisionpy`, from the current v0 (chance-only Bayesian network) through
the full mixed-type influence diagram with decisions and utilities.

---

## Current state

Two graph-native engines plus the NumPyro bridge: variable elimination
(`decisionpy.inference.exact.categorical`) for exact discrete Bayesian
networks, bucket elimination (same package) for exact discrete influence
diagrams, and the NumPyro translator
(`decisionpy.inference.numpyro.model.to_model`) for forward sampling and
MCMC/SVI — all consuming a validated `Snapshot`.

## Analysis of candidate methods

### Exact: variable elimination

- **Quality:** exact marginals
- **Speed:** exponential in treewidth of the graph
- **Scales:** small-to-medium BNs (typically ~50–100 nodes)
- **Relevance:** the `Snapshot` already carries a topological ordering, which is
  essentially the elimination order. Building VE directly on the graph is
  straightforward and gives reference answers for testing approximate methods.

### NumPyro (NUTS + SVI + enumeration)

- **Discrete unobserved nodes:** SVI with `infer={"enumerate": "parallel"}`
  marginalizes them out analytically *within each factor*. Caveat: variational
  distributions are typically mean-field, so posterior correlations between
  disconnected parents of a common child ("explaining away") are lost.
- **Continuous nodes:** NUTS is the gold standard — auto-tuned, efficient, no
  conjugacy constraints.
- **GPU/TPU:** free from JAX.
- **Gradient-based decision optimization:** once posterior inference and
  reparameterized forward sampling are in place, expected utility is a
  differentiable function of decision parameters. This enables SGD optimization
  over decision rules, which is a qualitative leap for large/continuous decision
  spaces — something VE cannot provide.
- **Discrete HMC:** experimental, unreliable, not recommended.

### Summary table

| Method               | Reliability | Handles discrete | Handles continuous | Gradients | GPU |
|----------------------|-------------|-------------------|---------------------|-----------|-----|
| Variable elimination | Exact       | Yes               | No (discrete only)  | N/A       | No  |
| SVI + enumeration    | Medium      | Yes (via enum)    | Yes (via NUTS)      | Yes       | Yes |
| NUTS (pure)          | High        | No                | Yes                 | Yes       | Yes |

---

## Decision: keep VE, grow NumPyro

### Short term (v0 BN inference)

Add **variable elimination** directly against the graph (`Snapshot` and
`ChanceNode` objects). It requires no PPL:

- VE uses the topological order already produced by `Snapshot`.

VE is graph-native, zero-dependency, and produces reference posteriors
against which approximate methods can be tested.

### Medium term (mixed-type influence diagrams)

Grow the **NumPyro backend** beyond forward sampling:

- Add `observed` support to `to_model` (conditional inference).
- Provide `to_svi` and `to_mcmc` helpers.
- Build gradient-based decision optimization on top of reparameterized sampling.

### Long term

NumPyro is the primary approximate-inference engine. Its strengths align with
the hardest parts of influence diagram solving:

1. **Inference over mixed discrete/continuous latents** — enumeration for
   discrete, NUTS for continuous.
2. **Expected utility estimation** — forward sampling is already built.
3. **Gradient-based policy optimization** — autodiff through the whole diagram
   for decision rules.

VE remains as:
- Exact discrete reference answers.
- A lightweight tool when JAX/NumPyro are not wanted.
- The building block for bucket elimination over utility/decision nodes.
