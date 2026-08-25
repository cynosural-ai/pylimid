# decisionpy examples

Interactive notebooks showcasing the library's capabilities. Read in numeric order.

| Notebook | Shows |
| --- | --- |
| `01_model_creation/workspace_and_validation.ipynb` | The mutable workspace: node types, the consistency gate (`UNCONFIGURED` / `STALE` / `CONSISTENT`), `validate()` / `snapshot()`, and live Mermaid rendering |
| `02_bayesian_networks/mechanics.ipynb` | `infer()` on the smallest BN: priors, evidence, and continuous variables returning raw draws |
| `02_bayesian_networks/categorical_bn.ipynb` | A fully-categorical network: multi-query, evidence, explaining away, estimates vs pyAgrum's exact posterior |
| `02_bayesian_networks/linear_gaussian_bn.py` | Linear-Gaussian BNs: an all-continuous chain, posterior draws via NUTS |
| `02_bayesian_networks/conditional_linear_gaussian_bn.py` | Conditional linear-Gaussian BNs — placeholder |
| `03_influence_diagrams/medical_treatment.ipynb` | `solve()`: optimal policy and expected utility for a treatment decision |
| `03_influence_diagrams/counterfactuals_with_policy.ipynb` | `infer(..., policy=)`: interventional queries, unbound-decision errors, solve + evaluate loop |
| `03_influence_diagrams/solver_comparison.py` | The NumPyro intervention scan: optimal policy, Monte-Carlo noise across seeds, mixed/continuous outcomes, and pyAgrum's exact solution as the reference |

## Running

```bash
uv sync                          # installs jupyterlab + pyagrum as dev dependencies
uv run jupyter lab examples/
```

Two examples (`solver_comparison.py`, `categorical_bn.py`) build the same
models in pyAgrum to show the exact answer next to the Monte-Carlo
estimate; pyagrum is in the `dev` dependency group, so `uv sync` installs it.

Each notebook is paired with a `.py` twin (Jupyter percent format, kept in sync
with jupytext) so the examples are greppable and reviewable. Regenerate a
twin after editing either side with:

```bash
uv run jupytext --sync examples/01_model_creation/workspace_and_validation.ipynb
```

To re-execute a notebook (regenerating its outputs):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/01_model_creation/workspace_and_validation.ipynb
```

Mermaid diagrams render natively in JupyterLab ≥ 4.1 and Notebook ≥ 7.
