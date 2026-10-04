# Scope and limitations

What `pylimid` is for, and where the edges are.

## What it models

`pylimid` models **limited-memory influence diagrams (LIMIDs)**. A classical influence diagram is the special case where all the memory arcs happen to be drawn; `pylimid` never assumes them. A decision observes exactly its declared parents, and anything it should remember must be an explicit arc.

Chance nodes can be **discrete, continuous, or mixed**, using arbitrary NumPyro distributions. Utilities are arbitrary JAX-traceable callables of their parents. What is discrete is the *decision* layer: decisions and their information sets must be discrete, because the policy is a table over information-set assignments. Continuous decisions are not supported yet.

## What it guarantees

There is one engine, built on NumPyro, and every result is a **Monte-Carlo estimate**:

- `infer` returns posterior draws. All-discrete diagrams use exact discrete enumeration; anything with a continuous latent uses NUTS. The draws are what you get; `Posterior` summarizes them.
- `solve` returns a Monte-Carlo policy and expected utility. On a **solvable** diagram, backward induction is optimal up to sampling noise; on a non-solvable diagram, `auto` falls back to the scan, which is the global optimum of the estimate at an exponential cost.

The solvability gate follows Lauritzen and Nilsson's criterion, which is stricter than pyAgrum's `isSolvable()`. On diagrams where the two disagree, `pylimid` refuses the one-pass answer and the scan remains the reference.

## Comparison with pyAgrum

For fully discrete diagrams, pyAgrum's exact LIMID solver is the right tool: it is exact, fast, and mature. The comparison tests in this repository use it as the external reference for `infer` and `solve`.

`pylimid` is aimed at the cases that exact tabular solvers cannot reach: arbitrary continuous distributions, mixed diagrams, and utility functions that are not tables — anything a forward sample can evaluate but a closed-form propagation cannot.

## Not supported yet

- **Continuous decisions** and continuous information sets.
- **Exact inference.** The exact engines (variable elimination, bucket elimination, linear-Gaussian propagation) were built and retired; validation against exact answers is outsourced to pyAgrum in the test suite.
- **State-dependent policy evaluation with `infer`.** `policy` clamps a decision to one action; the conditional rule that `solve` returns cannot be fed back through `infer`.
- **Sensitivity analysis helpers.** Parameter sweeps and tornado diagrams are a few lines of user code (the upcoming sensitivity tutorial shows the pattern), not library API.

## Where to go next

The `Tutorials` build complete models end to end; the `API reference` has the signatures and docstrings for every public name.
