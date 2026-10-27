"""Named keyword argument compatibility for node functions and decision rules."""

import inspect
from collections.abc import Callable
from typing import Any


def signature_matches(
    callable_: Callable[..., Any], parent_names: tuple[str, ...]
) -> bool:
    """
    Whether ``callable_`` names every entry of ``parent_names`` as a parameter.

    The guarantee the consistency gate can make is narrow: a ``CONSISTENT``
    callable *could* have used its parents. Coverage is by named parameters
    only — a parent that exists nowhere in the signature except a variadic
    ``*args`` / ``**kwargs`` does not count, because the gate cannot verify
    that a generic callable actually reads it. A ``**kwargs``-only callable is
    therefore ``CONSISTENT`` only for an empty parent set and stale for any
    wired parents; callables that resolve parents dynamically must generate a
    named-parameter signature (the gate then works for them like anyone
    else).

    A callable that names all parents but cannot be bound (a missing required
    extra parameter) is reported as not matching. A callable that *ignores*
    its parents but could have used them (an independent node) is a
    legitimate model; signature inspection cannot tell it apart from a wiring
    mistake, and the gate does not try.

    A callable that cannot be introspected (some builtins, C extensions) is
    conservatively reported as not matching.

    Args:
        callable_: Function whose signature will be inspected without executing it.
        parent_names: Declared parents passed to the function as keyword arguments.

    Returns:
        Whether every parent is explicitly named and the signature can be bound with
        exactly those keyword arguments.
    """
    try:
        sig = inspect.signature(callable_)
    except (TypeError, ValueError):
        return False
    named = {
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if any(parent not in named for parent in parent_names):
        return False
    try:
        sig.bind(**dict.fromkeys(parent_names))
    except TypeError:
        return False
    return True
