pylimid
=======

Use `infer` to query chance and utility nodes, and `solve` to find a policy that maximises expected total utility. Both functions validate the diagram before running and return Monte Carlo estimates.

The node classes and `InfluenceDiagram` are also available directly from ``pylimid``; their reference is under :doc:`graph`.

.. currentmodule:: pylimid

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer
   solve
   InferenceError
   Posterior
   Solution

Data types
----------

These aliases describe the dictionaries and solver names used by the functions above.

.. py:data:: InferenceResult
   :type: dict[str, Posterior]

   One `Posterior` per query variable. Entries at the same position in each posterior's ``values`` belong to the same joint draw.

.. py:data:: Policy
   :type: dict[str, dict[tuple[int, ...], int]]

   A policy maps each decision name to an action table. Each table maps a tuple of parent-state indices, in the decision's ``parents`` order, to an action index in its ``states``. A decision with no parents uses the empty tuple ``()`` as its only key.

   `solve` returns this format, and `infer` accepts it through ``policy``. `infer` also accepts a fixed action index for an individual decision.

.. py:data:: SolverName
   :type: Literal["backward_induction", "scan"]

   The solver that produced a `Solution`.

.. py:data:: SolveMethod
   :type: Literal["auto", "backward_induction", "scan"]

   Solver selection for `solve`.
