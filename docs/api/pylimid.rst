Inference and solving
=====================

Use `infer` to query chance and utility nodes, and `solve` to fit decision policies that maximise expected total utility. Both functions validate the diagram before running. Inference returns Monte Carlo draws; solving returns fitted stochastic rules and an independent Monte Carlo estimate of their expected utility.

These functions and their result and error types are available directly from ``pylimid``. Inference is also available from ``pylimid.inference``; optimization and solution results are available from ``pylimid.solver``. See :doc:`graph` for diagram construction and :doc:`policy` for policy families and rule representations.

.. currentmodule:: pylimid

Entry points
------------

`infer` accepts a `Policy` containing fixed actions, complete deterministic tables, deterministic functions or fitted `CategoricalPolicy` rules. Fitted rules support forward evaluation and conditioning on chance-node evidence, including with continuous or mixed information parents. Deterministic functions support forward evaluation only. Only decisions relevant to the query or evidence need a rule.

`solve` always uses policy gradients to jointly fit the `PolicyFamily` declared on every decision. Its ``num_samples`` is the number of scenarios per training update and in the independent final evaluation; ``num_steps`` controls the number of updates, and ``learning_rate`` controls the Adam step size. The returned rules remain stochastic. Training has no convergence or optimality guarantee, and increasing evaluation samples only reduces estimation noise.

Both functions default to ``num_samples=10_000`` and use a fixed default random key. The optimizer defaults to ``num_steps=500`` and ``learning_rate=0.05``. Their generated signatures and parameter descriptions below give the complete contract.

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer
   solve

Results
-------

`Posterior` holds draws for one query variable. Numerical summaries retain the node's shape as JAX arrays; scalar summaries are zero-dimensional arrays. `Posterior.is_categorical` identifies named categories, whose probabilities are read with `Posterior.marginal`. Numerical counts use numerical summaries rather than category probabilities.

`Solution` contains a `FittedPolicy` and its independently estimated expected utility. `Solution.render` uses the labels stored in the fitted rules and takes no diagram argument.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Posterior
   Solution

.. py:data:: InferenceResult
   :canonical: pylimid.inference.posterior.InferenceResult
   :type: dict[str, Posterior]

   One `Posterior` per queried variable. Each posterior stores a JAX array with shape ``(num_samples, *node_shape)``. Entries at the same position in each posterior's ``values`` belong to the same joint draw, so joint events and utility combinations can be computed from aligned draws.

Errors
------

The error classes below are also exported by ``pylimid.errors``. Invalid diagram configuration and optimizer settings raise `ValueError`. Unknown queried or observed node names raise `KeyError`. Relevant deterministic function policies combined with chance-node evidence raise `NotImplementedError`. The entry-point docstrings list the conditions for each error.

.. autosummary::
   :toctree: generated
   :nosignatures:

   InferenceError
   SolveError
