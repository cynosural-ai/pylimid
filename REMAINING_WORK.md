# Remaining work

## Small fixes

1. **`docs/TODO.md`** — in Spanish and stale. Either rewrite in English reflecting the current state or delete it. (Quick.)

## Engine roadmap

2. **NumPyro intervention-scan solver** — `solve()` on mixed/continuous influence diagrams currently raises a clean `InferenceError`. The planned NumPyro intervention-scan / gradient-optimization path (per `decision_node.md`, Strategy B) makes those diagrams solvable. Deferred from PR B on purpose.
