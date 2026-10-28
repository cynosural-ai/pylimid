Graph layer
===========

Use `ChanceNode`, `DecisionNode` and `UtilityNode` to build an `InfluenceDiagram`. Nodes and arcs can be edited while the model is incomplete. `InfluenceDiagram.validate` reports problems with its structure and configuration; `infer` and `solve` run this check automatically.

The node classes, continuous action spaces and `InfluenceDiagram` are also available directly from ``pylimid``. ``pylimid.graph.nodes`` owns the node types, ``pylimid.graph.action_space`` owns continuous action domains, and ``pylimid.graph.diagram`` owns the diagram, snapshots and validation problems. All are re-exported by ``pylimid.graph``.

A decision's ``states`` declare its finite categorical actions. Alternatively, ``action_space`` declares a scalar continuous domain using `Real` or `Interval`. Leave both unset for an unconfigured decision. Setting both is a conflict reported as `Consistency.STALE` during validation. Continuous policies must return a scalar continuous density with fixed support contained in the declared domain. `DistributionPolicy` composes such a distribution with a parameter model.

To fit a policy with `solve`, also declare a `PolicyFamily` on ``policy_family``. A decision can pass diagram validation before it has a family; `solve` checks this additional requirement before training. `infer` uses the supplied `Policy` instead of fitting a family.

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

Continuous action spaces
------------------------

`Real` allows an unrestricted real scalar. `Interval` declares fixed, inclusive finite bounds, with ``low < high``. Neither domain selects a policy distribution.

.. code-block:: python

   from pylimid import DecisionNode, Interval, Real

   adjustment = DecisionNode("adjustment", action_space=Real())
   power = DecisionNode("power", action_space=Interval(0.0, 100.0))

.. currentmodule:: pylimid.graph.action_space

.. autosummary::
   :toctree: generated
   :nosignatures:

   Real
   Interval

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
