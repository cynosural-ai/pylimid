# decisionpy examples

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
| `03_influence_diagrams/solver_comparison.py` | The batched NumPyro intervention scan: optimal policy, Monte-Carlo noise across seeds, mixed/continuous outcomes, pyAgrum's exact solution as the reference, and the Oil Wildcatter ID compared against pyAgrum's exact LIMID solver |

## Running

```bash
uv sync                          # installs jupyterlab + pyagrum as dev dependencies
uv run jupyter lab examples/
```

Three examples (`solver_comparison.py`, `categorical_bn.py`, and the
`solver_comparison.py` Oil Wildcatter section) build the same models in
pyAgrum to show the exact answer next to the Monte-Carlo estimate; pyagrum
is in the `dev` dependency group, so `uv sync` installs it. The Oil
Wildcatter shares one spec between both libraries
(`03_influence_diagrams/_oil_wildcatter.py`), so the comparison is exact
engine vs Monte-Carlo scan on identical tables.

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
