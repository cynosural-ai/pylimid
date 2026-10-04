Overview
========

pylimid is split into two layers: the **graph layer** for building a diagram, and the **inference layer** for querying it.

- **Graph layer** — `ChanceNode`, `DecisionNode`, `UtilityNode` and the `InfluenceDiagram` they live in.
- **Inference layer** — `infer` for posteriors and `solve` for the optimal policy.
