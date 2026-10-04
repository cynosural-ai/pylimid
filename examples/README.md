# pylimid examples

Interactive notebooks showcasing the library's capabilities. Read in numeric order.

| Notebook | Shows |
| --- | --- |
| `01_model_creation/workspace_and_validation.ipynb` | The mutable workspace: node types, the consistency gate (`UNCONFIGURED` / `STALE` / `CONSISTENT`), `validate()` / `snapshot()`, and live Mermaid rendering |
| `02_bayesian_networks/mechanics.ipynb` | `infer()` on the smallest BN: priors, evidence, and continuous variables returning raw draws |
| `02_bayesian_networks/categorical_bn.ipynb` | A fully-categorical network: multi-query, evidence, explaining away, estimates vs pyAgrum's exact posterior |
| `02_bayesian_networks/linear_gaussian_bn.py` | Linear-Gaussian BNs: an all-continuous chain, posterior draws via NUTS |
| `02_bayesian_networks/conditional_linear_gaussian_bn.py` | Conditional linear-Gaussian BNs — placeholder |
| `02_bayesian_networks/sangiovese/sangiovese.ipynb` | A realistic mixed CLG network (Magrini et al. 2017, from the bnlearn repository, CC BY-SA 3.0): treatment-to-quality queries, including the inverse query (which treatment explains an observed quality profile) |
| `03_influence_diagrams/medical_treatment.ipynb` | `solve()`: optimal policy and expected utility for a treatment decision |
| `03_influence_diagrams/counterfactuals_with_policy.ipynb` | `infer(..., policy=)`: interventional queries, unbound-decision errors, solve + evaluate loop |
| `03_influence_diagrams/solver_comparison.py` | The batched NumPyro intervention scan: optimal policy, Monte-Carlo noise across seeds, mixed/continuous outcomes, and pyAgrum's exact solution as the reference |
| `03_influence_diagrams/mixed_logistics_center.ipynb` | A mixed continuous/discrete LIMID: a continuous Gamma test score binned into a discrete report, solved with the discrete-decision scan and checked against pyAgrum solving the collapsed discrete model; `infer()` on the continuous score given the report |

## Running

```bash
uv sync                          # installs jupyterlab + pyagrum as dev dependencies
uv run jupyter lab examples/
```

Several examples (`categorical_bn.py`, `mixed_logistics_center.py`, and `solver_comparison.py`) build the same models in pyAgrum to show the exact answer next to the Monte-Carlo estimate; pyagrum is in the `dev` dependency group, so `uv sync` installs it. The logistics center shares its model and its `P(report | advanced_state)` collapse (`03_influence_diagrams/_logistics_center.py`), so the comparison is exact engine vs Monte-Carlo scan on identical probabilities.

Each notebook is paired with a `.py` twin (Jupyter percent format, kept in sync with jupytext) so the examples are greppable and reviewable. Regenerate a twin after editing either side with:

```bash
uv run jupytext --sync examples/01_model_creation/workspace_and_validation.ipynb
```

To re-execute a notebook (regenerating its outputs):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_model_creation/workspace_and_validation.ipynb
```

Mermaid diagrams render natively in JupyterLab ≥ 4.1 and Notebook ≥ 7.
