# Copyright 2026 HiperMaximus
"""Canonical selected-runtime row_id composition (single source).

The benchmark generator EMITS a selected row_id that encodes the winning
``(model x hardware)`` runtime shape, and the plan parser VALIDATES that id
structurally by recomposing it from the plan's own fields (Spec 0011 S17b). Both
sides must use the exact same formula or the structural check is worthless, so it
lives here in one stdlib-only leaf that either side imports, mirroring
``schedule.py`` / ``roots.py``.
"""

from __future__ import annotations

DEFAULT_RUNTIME_POLICY_ID = "fp32_eager_default"
# Policy ids that carry no ``__policy_`` suffix: the empty id and the default
# eager policy both denote "no distinguishing runtime policy", so the composed
# row_id is the bare base (matching the reference/candidate rows).
_SUFFIXLESS_RUNTIME_POLICY_IDS = frozenset({"", DEFAULT_RUNTIME_POLICY_ID})


def compose_row_id_base(
    *,
    accelerator_mode: str,
    batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
) -> str:
    """Return the base row_id shared by candidate and selected rows.

    Returns:
        The ``{accel}__bs{N}__{precision}__compile_{scope}__{corruption}`` base id.

    """
    return (
        f"{accelerator_mode}__bs{batch_size}__{precision_policy}"
        f"__compile_{compile_scope}__{corruption_strategy}"
    )


def compose_selected_row_id(  # noqa: PLR0913
    *,
    accelerator_mode: str,
    batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
    runtime_policy_id: str,
) -> str:
    """Return the full selected row_id, appending the runtime-policy suffix.

    The empty id and the default eager policy carry no suffix, matching the
    reference/candidate rows the base composer produces.

    Returns:
        The base id, plus ``__policy_{runtime_policy_id}`` for a distinguishing policy.

    """
    base = compose_row_id_base(
        accelerator_mode=accelerator_mode,
        batch_size=batch_size,
        precision_policy=precision_policy,
        compile_scope=compile_scope,
        corruption_strategy=corruption_strategy,
    )
    if runtime_policy_id in _SUFFIXLESS_RUNTIME_POLICY_IDS:
        return base
    return f"{base}__policy_{runtime_policy_id}"


__all__ = [
    "DEFAULT_RUNTIME_POLICY_ID",
    "compose_row_id_base",
    "compose_selected_row_id",
]
