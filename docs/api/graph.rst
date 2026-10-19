Graph layer
===========

Use `ChanceNode`, `DecisionNode` and `UtilityNode` to build an `InfluenceDiagram`. Nodes and arcs can be edited while the model is incomplete. `InfluenceDiagram.validate` reports problems with its structure and configuration; `infer` and `solve` run this check automatically.

The main classes are also available directly from ``pylimid``. The supporting types below describe node configuration, validation problems and the snapshots passed to individual solvers.

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
