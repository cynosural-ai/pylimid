Solvers
=======

.. currentmodule:: pylimid.solver

For most uses, call `pylimid.solve`. It chooses backward induction when the diagram is structurally solvable and the batched policy scan otherwise. Explicitly select ``method="policy_gradient"`` to fit one decision with a declared `LinearSoftmax` family and numerical or mixed information. The :doc:`solving guide <../user_guide/solving>` explains these choices.

The functions below expose the solvers individually. They take a `DiagramSnapshot`, obtained with `InfluenceDiagram.snapshot`, rather than the editable diagram. `batched_solve` and `scan_solve` enumerate the same policies; the batched version evaluates them together with JAX.

`is_solvable` and `solvability_order` check whether the graph admits backward induction. They do not check policy-family support, numerical behaviour, or whether every information-set assignment will appear in the samples. Structural solvability does not give policy-gradient training an optimality guarantee.

.. autosummary::
   :toctree: generated
   :nosignatures:

   backward_induction_solve
   batched_solve
   scan_solve
   policy_gradient_solve
   is_solvable
   solvability_order
