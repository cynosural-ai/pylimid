pylimid
=======

Use `infer` to query chance and utility nodes, and `solve` to find a policy that maximises expected total utility. Both functions validate the diagram before running and return Monte Carlo estimates.

The node classes and `InfluenceDiagram` are also available directly from ``pylimid``; their reference is under :doc:`graph`.

Posterior inference and its result types are also available from ``pylimid.inference``. Solver selection and solution results are available from ``pylimid.solver``. ``pylimid.policy`` defines the shared policy type, and ``pylimid.errors`` defines the error shared by inference and solving.

``Posterior.is_categorical`` replaces ``Posterior.is_discrete``. It distinguishes named categories, summarised with ``marginal()``, from numerical values, summarised with ``mean()``, ``std()`` or ``hdi()``. It does not infer mathematical discreteness from the draws' dtype.

.. currentmodule:: pylimid

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer
   solve
   InferenceError
   Posterior
   Solution
   LinearSoftmax
   SoftmaxPolicy

Data types
----------

These aliases describe the mappings and solver names used by the functions above.

.. py:data:: InferenceResult
   :type: dict[str, Posterior]

   One `Posterior` per query variable. Each posterior stores a JAX array with shape ``(num_samples, *node_shape)``. Entries at the same position in each posterior's ``values`` belong to the same joint draw.

.. py:data:: Policy
   :type: dict[str, dict[tuple[int, ...], int] | collections.abc.Callable[..., int | jax.Array] | SoftmaxPolicy]

   A policy maps each decision name to an action table, a deterministic function, or a fitted stochastic `SoftmaxPolicy`. Each table maps a tuple of parent-state indices, in the decision's ``parents`` order, to an action index in its ``states``. A decision with no parents uses the empty tuple ``()`` as its only key.

   Table solvers return tables and require categorical information sets. ``solve(method="policy_gradient")`` fits exactly one decision declaring ``policy_family=LinearSoftmax()`` and returns a `SoftmaxPolicy`. `infer` accepts these rules and fixed action indices through ``policy``. Tables, functions, and fitted rules can be mixed during evaluation.

   Function rules accept their declared information parents by keyword and return a scalar integer action index. They must be deterministic and JAX-traceable; use ``jnp`` operations rather than Python conditionals on parent values. For example, a decision observing a continuous ``signal`` could use:

   .. code-block:: python

      policy = {"action": lambda signal: jnp.where(signal > 0, 1, 0)}

   This supplies a complete rule for evaluation, rather than asking the solver to learn one. Function rules currently support forward evaluation only: combining a relevant function rule with chance-node evidence raises ``NotImplementedError``. Sampled function actions are checked against the decision's ``states``. Fitted stochastic rules also support forward evaluation only; their actions are sampled from their learned probabilities. Posterior inference through function and fitted stochastic rules remains unsupported.

.. py:data:: SolverName
   :type: Literal["backward_induction", "scan", "policy_gradient"]

   The solver that produced a `Solution`.

.. py:data:: SolveMethod
   :type: Literal["auto", "backward_induction", "scan", "policy_gradient"]

   Solver selection for `solve`.
