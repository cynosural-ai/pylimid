# Oil policy comparison

The [runnable example](../../examples/oil_policy_comparison.py) compares pyAgrum, PyLIMID backward induction, and PyLIMID policy gradients with `TabularSoftmax` on the same categorical oil model. The model matches the existing oil tutorial: quality probabilities `(0.35, 0.45, 0.20)`, purchase rewards `(1250, 630, 0)`, an outside option of `350`, and a test cost of `30`. All utilities are in millions of dollars.

The buy decision observes the result and remembers whether the test ran. Skipping the test produces `no_result` with probability one. No solver receives extra information about the hidden quality.

## Run it

From the repository root, with the default project dependencies installed:

```bash
uv run python examples/oil_policy_comparison.py --csv design/continuous_information/oil_policy_comparison.csv
```

The defaults compare 50, 200 and 1,000 training steps with seeds 0, 1 and 2, using 2,000 scenarios per training update. Backward induction uses 20,000 scenarios per action per decision. The script evaluates the first seed at the largest training budget on another 100,000 independent draws. The selected seed and budget are predetermined, rather than selected by their achieved utility. Use `--steps`, `--seeds`, `--batch-size` and `--evaluation-samples` to change these budgets.

The three calls at the center of the example are:

```python
import jax

from examples.oil_policy_comparison import (
    build_pylimid,
    exact_policy_value,
    solve_pyagrum,
)
from pylimid import solve

reference, _ = solve_pyagrum()
backward = solve(build_pylimid(), method="backward_induction", num_samples=20000)
gradient = solve(
    build_pylimid(trainable=True),
    method="policy_gradient",
    num_steps=1000,
    num_samples=2000,
    rng_key=jax.random.PRNGKey(0),
)
print("Exact optimum:", reference)
print("Backward-induction policy:", exact_policy_value(backward.policy))
print("Trained stochastic policy:", exact_policy_value(gradient.policy))
```

The example's other functions build the identical pyAgrum model, evaluate policies exactly, and print the comparison. It uses pyAgrum's [`ShaferShenoyLIMIDInference`](https://pyagrum.gitlab.io/reference/4_other_graphical_models/1_infdiag/3_infdiaginference/) for the reference solution.

## Results

The exact optimum is **721.000**: skip the test and buy the field. Testing and buying only after a pass gives **696.950**. pyAgrum agrees with an independent enumeration of all 128 deterministic joint policies. Analytic checks also verify the policy evaluator on those two strategies and on uniform randomized policies, whose exact value is 520.500.

The table below shows seed 0; [the CSV](oil_policy_comparison.csv) includes all three seeds.

| Solver | Training steps | Solver's reported utility | Exact value of returned policy | Exact loss from optimum |
| --- | ---: | ---: | ---: | ---: |
| pyAgrum | — | 721.000 | 721.000 | 0.000 |
| PyLIMID backward induction | — | 715.958 | 721.000 | 0.000 |
| PyLIMID tabular policy gradient | 50 | 724.840 | 712.326 | 8.674 |
| PyLIMID tabular policy gradient | 200 | 734.685 | 719.698 | 1.302 |
| PyLIMID tabular policy gradient | 1,000 | 735.310 | 720.884 | 0.116 |

The backward-induction policy is optimal in this run, even though its reported Monte Carlo estimate is below 721. The gradient solver's reported estimate can exceed the true optimum because its final evaluation uses only 2,000 scenarios. This does not mean it has found a better strategy.

Across the three training seeds:

| Training steps | Mean exact loss | Smallest exact loss | Largest exact loss |
| --- | ---: | ---: | ---: |
| 50 | 8.914 | 8.674 | 9.075 |
| 200 | 1.266 | 1.217 | 1.302 |
| 1,000 | 0.110 | 0.106 | 0.116 |

These runs show improvement with the larger budgets. They do not establish monotone improvement at every step or a guarantee for other models and seeds.

## CPU timings

The [timing script](../../examples/time_oil_policies.py) measures the current implementations on an **Apple M4 Pro with 12 CPU cores and 24 GiB memory**, using Python 3.12.13, JAX/jaxlib 0.11.0 and pyAgrum 3.0.0. JAX is explicitly restricted to CPU. All runs use seed 0 and the same budgets as the accuracy comparison.

```bash
uv run python examples/time_oil_policies.py --csv design/continuous_information/oil_cpu_timings.csv
```

These are wall-clock durations of the solver calls on prebuilt diagrams. Imports, initial JAX device setup, random-key creation, model construction and the subsequent exact accuracy checks are outside the solve timer. Constructing the inference engine and extracting the policy are included for pyAgrum. The PyLIMID calls include validation, policy initialization, compilation, all solving or training, and their final sampled evaluation. Trained arrays are synchronized before stopping the timer.

Each configuration has one initial solve followed by five measured repetitions. Every repetition performs a complete fresh solve, using the same input diagram; policy-gradient training starts from uniform probabilities each time. The table summarizes the five repetitions; the [raw timings](oil_cpu_timings.csv) also preserve initial-call and construction durations. First calls occur sequentially in one process, so they are not independent fresh-process cold-start measurements. The repeated-call medians still include compilation performed inside each current `solve` call.

| Solver | Median solve time, seconds | Repeated-call range, seconds | Exact utility loss, millions |
| --- | ---: | ---: | ---: |
| pyAgrum | 0.000370 | 0.000322–0.000447 | 0.000 |
| PyLIMID backward induction | 0.442 | 0.415–0.474 | 0.000 |
| Tabular policy gradient, 50 steps | 0.412 | 0.405–0.447 | 8.674 |
| Tabular policy gradient, 200 steps | 0.504 | 0.482–0.522 | 1.302 |
| Tabular policy gradient, 1,000 steps | 0.925 | 0.900–1.034 | 0.116 |

On this small categorical example, 1,000-step policy-gradient training takes about **2.1 times** as long as backward induction, and both finish in about a second or less. The shorter gradient budgets are quicker but produce larger utility losses. pyAgrum is much faster here, around **0.37 milliseconds**, while also finding the exact optimum.

Backward induction uses approximately 100,000 complete simulated trajectories: 20,000 per action for each of two binary decisions, plus 20,000 for its final evaluation. The 1,000-step gradient configuration uses 2,000,000 training trajectories, plus 2,000 for final evaluation and initialization/pilot draws. It also computes score gradients and Adam updates. Wall time includes fixed setup and compilation costs as well as that work, so trajectory counts alone do not predict the timing ratio.

These measurements describe this implementation, hardware, model and accuracy budget. They do not establish how the methods scale on larger diagrams or predict NVIDIA GPU timings. For this fully categorical oil model, pyAgrum is the strongest choice on both accuracy and CPU time. Policy-gradient training provides a route to other information sets and policy families; this example does not demonstrate a speed advantage for it.

## Measuring approximation

For this small categorical model, there is no need to estimate a fixed policy's value by simulation. The evaluator sums all 36 combinations of quality, test action, result and buy action, including the zero-probability combinations:

```text
J(policy) = sum over q,t,r,b of
    P(q) * policy_test(t) * P(r | q,t) * policy_buy(b | t,r) * U(q,t,b)

regret = exact optimum - J(policy)
```

This calculation preserves the trained policy's randomized probabilities and the dependence of the result on the test action. Its regret therefore measures actual utility loss within this specified model, up to floating-point roundoff, with no evaluation sampling noise. The full tabular family can approach every deterministic policy; these tables do not impose the linear-family restriction on categorical interactions.

For the 1,000-step, seed-0 stochastic policy, the exact loss is **0.115658 million**, about **$116,000** or **0.016%** of the optimum. Whether this is too large depends on the accepted loss. For example, it meets a $1 million tolerance but does not meet a $100,000 tolerance. Larger training budgets and additional seeds can be evaluated against the same explicit threshold.

The three impossible buy-table rows have zero trajectory probability, so their action probabilities do not affect this value. Rows that are possible but rarely visited can matter; their importance is measured through their probability and utility consequences, rather than by requiring every table entry to match another solver.

## Sampling uncertainty and deterministic conversion

The independent 100,000-draw evaluation of the 1,000-step, seed-0 policy gives **721.461**, with an approximate 95% Monte Carlo interval of **[718.649, 724.273]**. The script computes this as the sample mean plus or minus `1.96 * sample_standard_deviation / sqrt(num_samples)`. The exact value, **720.884**, lies inside that interval. The interval describes evaluation uncertainty; it does not describe the distance from the optimum. Its width is much larger than the actual 0.116 loss, which is why comparing noisy solver estimates alone would be misleading here.

The script also separately evaluates the policy obtained by taking each row's largest probability. In every reported run, that deterministic conversion has exact value **721.000**. This is a useful finding for this example, not a reason to silently replace a stochastic result: the conversion is a different policy, and independent argmax conversions are not guaranteed to improve a joint policy in general.

## When exact evaluation is too expensive

Use a fresh, sufficiently large evaluation sample to compare fixed candidate policies, and report uncertainty for their difference. Paired simulations with shared external randomness can reduce comparison noise when the simulator supports them correctly. Evaluate sensitivity to training budgets and seeds, and compare with a trusted reference solver on smaller versions of the model.

Agreement between several approximate runs, or a narrow interval around one policy's value, does not certify proximity to the global optimum. A numerical global-loss bound requires an exact optimum or a valid upper bound to compare against. Increasing evaluation samples reduces uncertainty about a fixed candidate's value; increasing training effort addresses a separate optimization problem.
