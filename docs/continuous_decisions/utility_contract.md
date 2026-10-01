# Writing utilities for continuous decisions

Companion to [`solver_methods.md`](./solver_methods.md). This note pins down the one thing a continuous-decision solver changes about utilities: **nothing structurally**, but a stricter *writing contract*. It is the contract, the failure modes, and how the library should enforce it.

## The utility node does not change

A utility stays exactly what it is today: a `UtilityNode` with `parents` and a `values` callable invoked as `values(**parent_values) -> scalar`, additive across multiple utility nodes. There is no "continuous utility" node and no new field. A payoff was never typed by a domain — it is a deterministic real-valued function. The graph layer's signature gate (`_signature_matches`) already ensures the callable names its parents; that is the *shape* contract. What this note is about is the *regularity* contract.

## Two callable contracts

**Traceable** — the baseline for the vmapped/jitted solvers (`batched_scan` already documents it). The callable works when its inputs are JAX tracers inside `jit`/`vmap`. It may contain integer state indices, comparisons, `jnp.where`, and boolean masks. It does **not** have to be differentiable.

**Differentiable on the decision path** — the extra requirement of the gradient solver. Along the path from a continuous decision's action to this utility, the utility must change smoothly enough that `∂U/∂a` carries signal. The requirement is *per path*: a utility fed only by discrete nodes, or by no decision at all, needs only traceability.

The dangerous middle is **traceable but not differentiable**: JAX runs it and returns zero gradients, so the failure is silent.

## What differentiability means mechanically

The optimizer shifts a policy parameter `θ`, which shifts the action `a`, which shifts the utility through the graph:

```
θ  ->  a  ->  descendants  ->  U
```

If `U` responds smoothly to `a`, `dU/da` carries the learning signal. The thermostat is the clean case — `U = -(T - 20)^2 - a^2`. The indicator is the pathological one:

```python
values=lambda a, D: (a >= D).astype(float)   # zero derivative almost everywhere
```

The expectation `E[1{D <= a}] = P(D <= a)` is a smooth CDF, but the sampled pathwise derivative is zero: the estimator is unusable even though the objective is fine. This is the utility-levels table from [`README.md`](./README.md) seen from the callable's side.

## Patterns

| Instead of | Write | Why |
| --- | --- | --- |
| `float(a)`, `int(a)`, `bool(a)` | keep `a` as a JAX scalar | tracer conversion is a hard error under `jit`/`grad` |
| Python `if a > 0: ...` | `jnp.where(a > 0, x, y)` | control flow on a tracer is a hard error; `where` is differentiable |
| built-in `min(x, a)` | `jnp.minimum(x, a)` | built-in calls `bool()` on a tracer; `jnp.minimum` gives a subgradient |
| `max(0, a)` | `jnp.maximum(0, a)` / `jax.nn.relu(a)` | same |
| `jnp.round(a)`, `jnp.floor(a)`, `.astype(int)` | avoid, or smooth the effect | derivative is zero almost everywhere |
| `(a >= D).astype(float)` | analytic `P(D <= a)`, or `jax.nn.sigmoid((a - D)/τ)` | the indicator's pathwise derivative is zero |
| `jax.lax.stop_gradient(a)` | don't | deliberately severs the gradient |
| `np.sum(...)` | `jnp.sum(...)` | numpy arrays carry no gradients |
| `jnp.clip(a, lo, hi)` | fine | derivative 1 inside, 0 outside — saturation is intended |
| `jnp.abs`, `jnp.minimum`, `jax.nn.relu` | fine | subgradients; workable |

## Failure modes

**Loud by construction.** `float`/`int`/`bool` on a tracer, Python `if` on a traced value, and built-in `min`/`max` raise a JAX tracing error the first time the solver runs. The engine should catch and augment the message ("utilities for continuous decisions must be written with jnp operations"). A non-scalar return or NaN/Inf in the objective or gradients should likewise raise.

**Silent.** Rounding, integer casts, indicator arithmetic, `stop_gradient`, and any utility that is piecewise constant in the action run cleanly and yield zero gradients. Subgradients at kinks (`min`/`max`/`relu`) work but are not classical derivatives. Saturation (a squashed action at its bound) produces vanishing gradients that are indistinguishable from convergence.

## Enforcement: three layers

Documentation alone is not enough, and the repo's own philosophy is "errors should never pass silently" plus "one honest contract". The precedent is already in the tree: `probe_discrete_parents()` is an *opt-in* check that runs user callables to catch the classic silent mistake (a `dist` that ignores its parent) and returns a warning, because it cannot distinguish a bug from a deliberately independent node.

1. **Loud checks (automatic).** Tracer-conversion errors (above), non-scalar utility returns, NaN/Inf. Structural unsupported cases stay with the capability gate.
2. **A first-step gradient smoke check.** After initializing the policy, if *every* policy parameter has exactly zero gradient, warn that the utility may not depend smoothly on the decision. A warning, not an error: zero can occur legitimately at a stationary point or under saturation, so this detects total severance rather than proving anything.
3. **An opt-in probe.** A `probe_decision_gradients()`, mirroring `probe_discrete_parents()`: perturb a continuous decision's action, check the utility responds, and optionally compare autodiff against finite differences (`jax.test_util.check_grads`). Same caveats — kinks and saturation trip it — so it warns.

**What remains an irreducible contract (tests + docs):**

- Whether the expectation should have been written analytically rather than sampled.
- Subgradients at kinks — workable, not exact.
- Vanishing gradients at bounds — indistinguishable from convergence.
- A policy family too weak to represent the optimum — not detectable; yields a valid local optimum.

These are documented, not checked. The tests carry the positive cases: the LQG thermostat (gradients nonzero, coefficients recovered) and, for non-LQG, the newsvendor critical fractile.

## Helpers

A small set of recipes removes most of the footguns: `jax.nn.sigmoid`/`softplus` for smoothing, `jax.scipy.special.logsumexp` for soft maximum, a temperature-annealed smooth indicator, and `jnp.where` in place of branching. Whether to package these as `decisionpy` utility helpers is an open question; the minimum is to use them in the examples so the intended style is visible.
