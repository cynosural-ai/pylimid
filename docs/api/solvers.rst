Solvers
=======

.. currentmodule:: pylimid.solver

For most uses, call `pylimid.solve`. It chooses backward induction when the diagram is structurally solvable and the batched policy scan otherwise. The :doc:`solving guide <../user_guide/solving>` explains this choice.

The functions below expose the solvers individually. They take a `DiagramSnapshot`, obtained with `InfluenceDiagram.snapshot`, rather than the editable diagram. `batched_solve` and `scan_solve` enumerate the same policies; the batched version evaluates them together with JAX.

`is_solvable` and `solvability_order` check whether the graph admits backward induction. They do not check numerical behaviour or whether every information-set assignment will appear in the samples.

.. autosummary::
   :toctree: generated
   :nosignatures:

   backward_induction_solve
   batched_solve
   scan_solve
   is_solvable
   solvability_order
