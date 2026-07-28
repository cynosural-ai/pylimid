# Inference strategy for influence diagrams

This documents the reasoning behind the choice of inference backends for
`decisionpy`, from the current v0 (chance-only Bayesian network) through
the full mixed-type influence diagram with decisions and utilities.

---

## Current state (v0)

We have one backend: `decisionpy.backend.numpyro` — forward (prior-predictive)
sampling for chance-node-only diagrams. It translates a validated `Snapshot`
into a NumPyro model and the caller feeds it to `Predictive`.

## Analysis of candidate methods

### Exact: variable elimination

- **Quality:** exact marginals
- **Speed:** exponential in treewidth of the graph
- **Scales:** small-to-medium BNs (typically ~50–100 nodes)
- **Relevance:** the `Snapshot` already carries a topological ordering, which is
  essentially the elimination order. Building VE directly on the graph is
  straightforward and gives reference answers for testing approximate methods.

### Gibbs sampling

- **Discrete nodes:** conditionals are always tractable (a finite categorical).
  Markov blanket = factors involving the node, trivially computable from the
  graph and CPTs.
- **Continuous conjugate nodes:** conditional stays in a known family (e.g.
  Normal-Normal, Gamma-Poisson, Beta-Binomial). Pure Gibbs step.
- **Non-conjugate continuous nodes:** no closed-form conditional. Fall back to
  Metropolis-within-Gibbs, which requires hand-tuned proposals and often mixes
  slowly.
- **Decision solving:** expected utility and policy optimization are
  sample-based, no gradients.

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
  spaces — something neither VE nor Gibbs can provide.
- **Discrete HMC:** experimental, unreliable, not recommended.

### Summary table

| Method                       | Reliability | Handles discrete | Handles continuous | Gradients | GPU |
|------------------------------|-------------|-------------------|---------------------|-----------|-----|
| Variable elimination         | Exact       | Yes               | No (discrete only)  | N/A       | No  |
| Gibbs (discrete + conjugate) | High        | Yes               | Conjugate families only | No    | No  |
| Gibbs + Metropolis-within    | Medium      | Yes               | Yes (slow)          | No        | No  |
| SVI + enumeration            | Medium      | Yes (via enum)    | Yes (via NUTS)      | Yes       | Yes |
| NUTS (pure)                  | High        | No                | Yes                 | Yes       | Yes |

---

## Decision: keep VE, grow NumPyro

### Short term (v0 BN inference)

Add **variable elimination** and **Gibbs sampling** directly against the graph
(`Snapshot` and `ChanceNode` objects). Neither requires a PPL:

- VE uses the topological order already produced by `Snapshot`.
- Gibbs uses Markov blanket conditionals computed from the node's `dist` and
  the factors of its children.

These are graph-native, zero-dependency, and produce reference posteriors
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

VE and Gibbs remain as:
- Exact discrete reference answers.
- Lightweight tools when JAX/NumPyro are not wanted.
- Building blocks for bucket elimination over utility/decision nodes.

### Why not Gibbs as the primary backend

Gibbs occupies an awkward middle ground: not exact like VE, not scalable or
gradient-friendly like NumPyro. For a library targeting mixed-type decision
optimization, Gibbs doesn't provide enough advantage over the combination of
VE (exact discrete) + NumPyro (approximate mixed-type + gradients) to justify
a third engine.
