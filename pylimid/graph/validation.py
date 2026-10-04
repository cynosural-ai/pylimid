"""
Validation — the problem model shared by the diagram's checks.

The graph layer's checking surface is a list of `DiagramProblem` instances:
`InfluenceDiagram.validate` collects structural issues (dangling parents,
cycles, unconfigured or stale nodes, utility sinks), and
`InfluenceDiagram.probe_discrete_parents` adds the runtime
parent-sensitivity warning. The model lives here, separate from the
container, because it is the API a UI or LLM renders and acts on — and
because the checking surface is the part of the layer most likely to grow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["DiagramProblem", "ProblemKind"]


class ProblemKind(Enum):
    """
    Category of a `DiagramProblem`.

    Attributes:
        DANGLING_PARENT: A node lists a parent that is not in the diagram.
        CYCLE: The graph contains a cycle.
        UNCONFIGURED: A node's configurable field (``dist`` / ``values`` /
            action ``states``) is unset; inference cannot run on it.
        STALE: A node's configurable field signature does not match its
            parents.
        UTILITY_NOT_SINK: A utility node has acquired a child — a payoff must
            be terminal.
        DIST_IGNORES_PARENT: A node's callable returns the same output for
            every value of a discrete parent — found by
            `InfluenceDiagram.probe_discrete_parents`, a warning rather than
            an error.
    """

    DANGLING_PARENT = "dangling_parent"
    CYCLE = "cycle"
    UNCONFIGURED = "unconfigured"
    STALE = "stale"
    UTILITY_NOT_SINK = "utility_not_sink"
    DIST_IGNORES_PARENT = "dist_ignores_parent"


@dataclass(frozen=True)
class DiagramProblem:
    """
    A single issue found by a diagram check.

    Carries enough structure for a UI or LLM to render or act on the problem
    without parsing prose.
    """

    kind: ProblemKind
    node: str
    message: str
