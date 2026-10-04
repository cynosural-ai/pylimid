pylimid
=======

The package front door. Build a diagram with the graph layer, then `infer` a posterior or `solve` for the optimal policy.

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

These are type aliases; they carry no signature of their own.

.. py:data:: InferenceResult
   :type: dict[str, Posterior]

   One `Posterior` per query variable.

.. py:data:: Policy
   :type: dict[str, dict[tuple[int, ...], int]]

   Optimal policy: decision name to information-set assignment to chosen action.

.. py:data:: SolverName
   :type: Literal["backward_induction", "scan"]

   The solver that produced a `Solution`.

.. py:data:: SolveMethod
   :type: Literal["auto", "backward_induction", "scan"]

   Solver selection for `solve`.
