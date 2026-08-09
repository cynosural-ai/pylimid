"""
Influence-diagram solvers — exact discrete policies via bucket elimination.

Public API::

    from decisionpy.inference.id import solve

    result = solve(diagram.snapshot())
    # result.policy   -> {"treat": {(0,): 0, (1,): 1}, ...}
    # result.expected_utility -> 78.0
"""

from decisionpy.inference.id.bucket_elim import Policy, Solution, solve  # noqa: F401
