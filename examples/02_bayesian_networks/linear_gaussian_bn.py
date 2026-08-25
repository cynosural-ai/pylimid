"""
Linear-Gaussian Bayesian networks — PLACEHOLDER, not implemented yet.

A linear-Gaussian BN is an all-continuous network: every node is Gaussian and
its mean is a linear function of its parents' values,

    x | parents ~ Normal(loc = a + sum_i b_i * parent_i, scale = sigma)

with every parent continuous too. Example shape: x1 -> x2 -> x3, with
x1 ~ Normal(0, 1), x2 | x1 ~ Normal(2 + 1.5 * x1, 0.5), and so on.

Current status:
- The models themselves can be built with the existing graph layer and run
  through the NumPyro engine (approximate Monte-Carlo draws).
- Exact Gaussian inference — a graph-native engine (Gaussian belief
  propagation / junction tree) returning exact Gaussian posteriors — is not
  implemented yet. This example notebook will land with that engine.
"""
