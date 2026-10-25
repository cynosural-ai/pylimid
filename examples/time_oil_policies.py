"""
Measure CPU wall time for the three solvers on the categorical oil problem.

Run with ``uv run python examples/time_oil_policies.py``. Imports are excluded;
construction, first solve, and repeated solve calls are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import pyagrum as gum

from examples.oil_policy_comparison import (
    build_pyagrum,
    build_pylimid,
    exact_policy_value,
    solve_pyagrum,
)
from pylimid import TabularPolicy, solve


def main() -> None:
    """Measure current public solver calls sequentially and optionally save timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    print(f"CPU devices: {jax.devices()}", flush=True)
    print(f"Machine: {platform.machine()}, {os.cpu_count()} logical CPUs", flush=True)
    print(
        f"Python {platform.python_version()}, JAX {jax.__version__}, "
        f"pyAgrum {gum.__version__}",
        flush=True,
    )
    key = jax.random.PRNGKey(0)
    key.block_until_ready()
    configurations = [
        ("pyAgrum", 0, 0, build_pyagrum),
        ("Backward induction", 0, 20000, build_pylimid),
        ("Policy gradient", 50, 2000, lambda: build_pylimid(trainable=True)),
        ("Policy gradient", 200, 2000, lambda: build_pylimid(trainable=True)),
        ("Policy gradient", 1000, 2000, lambda: build_pylimid(trainable=True)),
    ]
    rows = []
    for method, steps, samples, build in configurations:
        start = perf_counter()
        diagram = build()
        jax.effects_barrier()
        build_seconds = perf_counter() - start
        print(f"\n{method}, {steps} steps, {samples} samples", flush=True)
        for repeat in range(args.repeats + 1):
            start = perf_counter()
            if method == "pyAgrum":
                _, policy = solve_pyagrum(diagram)
            else:
                options = {"num_steps": steps} if method == "Policy gradient" else {}
                result = solve(
                    diagram,
                    method="policy_gradient" if steps else "backward_induction",
                    num_samples=samples,
                    rng_key=key,
                    **options,
                )
                policy = result.policy
                for rule in policy.values():
                    if isinstance(rule, TabularPolicy):
                        rule.logits.block_until_ready()
            elapsed = perf_counter() - start
            value = exact_policy_value(policy)
            rows.append(
                {
                    "method": method,
                    "steps": steps,
                    "num_samples": samples,
                    "phase": "first" if repeat == 0 else "repeat",
                    "repeat": repeat,
                    "build_seconds": build_seconds,
                    "solve_seconds": elapsed,
                    "exact_policy_utility": value,
                }
            )
            print(
                f"  call {repeat}: {elapsed:.6f} s; exact utility {value:.6f}",
                flush=True,
            )
        timings = [
            row["solve_seconds"]
            for row in rows
            if row["method"] == method
            and row["steps"] == steps
            and row["phase"] == "repeat"
        ]
        print(f"  repeated-call median: {median(timings):.6f} s", flush=True)
    if args.csv is not None:
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
