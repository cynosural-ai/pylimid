# TODO

## Reliability and performance

- [ ] Check chained table indexing (``T[a][b]``) in posterior inference against an exact reference. Add regression coverage and decide whether a targeted diagnostic is feasible. See the [indexing limitation](design/inference/backend_numpyro.md#constraints).
- [ ] Add training diagnostics to `solve` and report Monte Carlo error for independently estimated expected utility.
- [ ] Expose NUTS warmup, chain controls, and sampling diagnostics through `infer`.
- [ ] Profile repeated `solve` and `infer` calls and reduce avoidable recompilation.

## Documentation

- [ ] Review and migrate tutorial drafts before publishing them. See the [documentation plan](design/documentation_plan.md#future-work).
