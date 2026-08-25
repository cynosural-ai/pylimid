# Remaining work

## Engine roadmap

1. **NumPyro intervention-scan solver** — `solve()` on mixed/continuous influence diagrams currently raises a clean `InferenceError`. The planned NumPyro intervention-scan / gradient-optimization path (per `decision_node.md`, Strategy B) makes those diagrams solvable. Deferred from PR B on purpose.

2. **Exact linear-Gaussian inference** — all-continuous BNs (every node Gaussian, linear dependencies) currently route to NumPyro MCMC. A graph-native exact engine (Gaussian belief propagation / junction tree) yields exact Gaussian posteriors — numpy-only, cross-validated against the MCMC draws. The first of two steps toward exact continuous inference.

3. **Exact conditional-linear-Gaussian inference** — mixed BNs (discrete children with discrete parents only; continuous children with either) currently route to NumPyro MCMC. The classic CLG propagation (Lauritzen): conditioned on a discrete assignment the continuous posterior is Gaussian; marginalizing over the discrete states yields a mixture of Gaussians. Builds on the LG engine (item 2). Two design decisions to settle when implementing:
   - whether the discrete-parents-only constraint becomes an enforced validation rule (today it is only a modeling convention that fails loudly at inference),
   - how exact continuous results fit the current contract, where continuous queries return `Draws` (raw samples) with no exactness flag.
