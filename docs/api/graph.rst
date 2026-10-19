Graph layer
===========

Use `ChanceNode`, `DecisionNode` and `UtilityNode` to build an `InfluenceDiagram`. Nodes and arcs can be edited while the model is incomplete. `InfluenceDiagram.validate` reports problems with its structure and configuration; `infer` and `solve` run this check automatically.

The main classes are also available directly from ``pylimid``. The supporting types below describe node configuration, validation problems and the snapshots passed to individual solvers.

``is_categorical`` reports whether a node declares named categories or actions through ``states``. It replaces the label-based ``is_discrete`` property. This check does not evaluate distributions or classify an unconfigured node as continuous; the backend reads mathematical discreteness from each constructed distribution's support. Numerical counts use ``states=None`` and pass their actual values to children. They support forward simulation and observed likelihoods, while posterior inference with unknown counts and count-indexed policy tables remain unsupported.

.. currentmodule:: pylimid.graph

.. autosummary::
   :toctree: generated
   :nosignatures:

   ChanceNode
   DecisionNode
   UtilityNode
   InfluenceDiagram
   DiagramSnapshot
   Node
   NodeKind
   Consistency
   DiagramProblem
   DiagramProblemKind
