# Exact inference engines — phased plan

This is the execution plan for the exact continuous-inference roadmap
(REMAINING_WORK.md items 2 and 3) plus the package reorganization that
prepares for it. Each phase is independently shippable, keeps the test
suite green, and lands as its own PR.

## The organizing principle

Exact engines are grouped by **model class**, not by algorithm. Dispatch
(engine.py) classifies a diagram by model class — discrete, all-continuous
linear-Gaussian, mixed — and each exact engine owns the preconditions of
its class and fails loudly when they are violated. NumPyro remains the
default engine for `infer()`; exact engines are opt-in.

Why model class and not algorithm: VE and bucket elimination are the same
model class with different verbs (infer vs solve) and share the factor
machinery in `inference/utils`. LG and CLG are separate model classes with
their own representations (canonical-form Gaussian factors; mixed
potentials). The package names then stay stable even if an engine's
scheduling algorithm later changes (belief propagation vs junction tree).

## Target layout

```
decisionpy/inference/
├── __init__.py                # infer(), solve(), result types
├── engine.py                  # dispatch
├── exact/
│   ├── categorical/
│   │   ├── variable_elim.py   # query()  — exact discrete BN marginals
│   │   └── bucket_elim.py     # solve()  — exact discrete ID policies
│   ├── linear_gaussian/       # phase 2
│   └── clg/                   # phase 4
├── numpyro/                   # approximate engine, any model class
└── utils/                     # shared factor machinery (discrete)
```

## Phase 1 — Restructure exact engines under exact/categorical

Mechanical move, zero behavior change: `ve/elim.py` becomes
`exact/categorical/variable_elim.py`, `id/bucket_elim.py` becomes
`exact/categorical/bucket_elim.py`, the old packages are deleted, and all
call sites (engine.py, tests, examples, notebooks, docs) are updated to
the new paths. The engine names ("ve", "bucket_elim") and their behavior
are unchanged. Done first so LG and CLG land directly in their final
homes.

## Phase 2 — Linear-Gaussian engine (opt-in engine="lg")

An exact all-continuous engine mirroring variable elimination step for
step, but over Gaussian factors:

- `gaussian_cpt` — probe each CPD: call `dist` at a baseline point (all
  parents zero; no parents for a root node) and at one point per parent;
  the returned `Normal`'s `loc` yields the intercept and slopes, `scale`
  the conditional standard deviation. Verify linearity by reconstructing
  `loc` at extra random points and failing loudly (InferenceError) if a
  CPD is not `Normal` or not affine in its parents. The probe doubles as
  parameter extraction, so no structural marker is needed.
- `gaussian_factor` — a factor over a variable set in canonical form
  `exp(-1/2 xᵀJx + hᵀx)`: multiply = add `(J, h)`, marginalize = Schur
  complement, condition = clamp. The LG analog of `utils.factor.Factor`.
- `query()` — same elimination schedule as `variable_elim.query()`.
- Result type `Gaussian(mean, variance)` — a Gaussian posterior is not
  raw draws, so the current `Draws` contract does not fit; the
  `InferenceResult` union grows but the NumPyro default path is untouched.
- Example fills in `examples/02_bayesian_networks/linear_gaussian_bn.py`
  (currently a placeholder); tests cover analytic posteriors on small LG
  chains and cross-validation against NumPyro MCMC draws.

## Phase 3 — NumPyro as the default engine for infer()

`infer()` defaults to `engine="numpyro"`; the "auto" dispatch
(`_choose_engine`) is removed, and the exact engines become explicit
opt-ins that validate their own preconditions. `solve()` is unchanged —
bucket elimination remains the only implemented solver, so it keeps its
auto-dispatch until the NumPyro solver exists. Behavioral consequence to
make visible in the PR: all-discrete diagrams go from exact-by-default to
approximate-by-default. Tests and the inference ADR are updated.

## Phase 4 — Conditional linear-Gaussian engine (opt-in engine="clg")

Lauritzen-style propagation on `exact/clg`, reusing `gaussian_factor`
for the per-discrete-assignment Gaussian potentials and the discrete
machinery in `utils` for the discrete potentials. Marginalizing the
discrete states yields a mixture of Gaussians, so a `Mixture` result type
lands here. Settles the two open design questions from REMAINING_WORK.md
item 3: whether the discrete-parents-only constraint becomes an enforced
validation rule, and how exact continuous results fit the result contract.
