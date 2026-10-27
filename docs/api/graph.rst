Graph layer
===========

Use `ChanceNode`, `DecisionNode` and `UtilityNode` to build an `InfluenceDiagram`. Nodes and arcs can be edited while the model is incomplete. `InfluenceDiagram.validate` reports problems with its structure and configuration; `infer` and `solve` run this check automatically.

The node classes and `InfluenceDiagram` are also available directly from ``pylimid``. ``pylimid.graph.nodes`` owns the node types, and ``pylimid.graph.diagram`` owns the diagram, snapshots and validation problems. All are re-exported by ``pylimid.graph``.

A decision's ``states`` declare its finite actions. To fit a policy with `solve`, also declare a `PolicyFamily` on ``policy_family``. A decision can have states and pass diagram validation before it has a family; `solve` checks this additional requirement before training. `infer` uses the supplied `Policy` instead of fitting a family.

``is_categorical`` reports whether a node declares named categories or actions through ``states``. It replaces the label-based ``is_discrete`` property. This check does not evaluate distributions or classify an unconfigured node as continuous; the backend reads mathematical discreteness from each constructed distribution's support. Numerical counts use ``states=None`` and pass their actual values to children. Scalar counts with fixed NumPyro-enumerable integer support starting at zero, such as a fixed-trial Binomial, also support posterior inference. Unbounded count latents and count-indexed policy tables remain unsupported; see `ChanceNode` for the complete limits.

Nodes
-----

.. currentmodule:: pylimid.graph.nodes

.. autosummary::
   :toctree: generated
   :nosignatures:

   ChanceNode
   DecisionNode
   UtilityNode
   Node
   NodeKind
   Consistency

Diagrams and validation
-----------------------

`InfluenceDiagram.snapshot` returns nodes in topological order after diagram validation. A `DiagramSnapshot` holds references to the original nodes; take a new snapshot after editing the diagram or its nodes.

.. currentmodule:: pylimid.graph.diagram

.. autosummary::
   :toctree: generated
   :nosignatures:

   InfluenceDiagram
   DiagramSnapshot
   DiagramProblem
   DiagramProblemKind
