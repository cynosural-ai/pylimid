# Remaining work

## Small fixes

1. **`docs/TODO.md`** — in Spanish and stale. Either rewrite in English reflecting the current state or delete it. (Quick.)
2. **`_signature_matches` accepts `**kwargs`-only callables as "matching any parent set"** — a `dist` or `values` that silently ignores its parents is `CONSISTENT`. Documented, but it's a silent-wrongness footgun in the exact layer meant to gate inference. (Design decision, graph layer.)

## Engine roadmap

3. **NumPyro intervention-scan solver** — `solve()` on mixed/continuous influence diagrams currently raises a clean `InferenceError`. The planned NumPyro intervention-scan / gradient-optimization path (per `decision_node.md`, Strategy B) makes those diagrams solvable. Deferred from PR B on purpose.
