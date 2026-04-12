from __future__ import annotations

from typing import Any, Callable, Optional


LogFn = Optional[Callable[[str], None]]


def enforce_channel_template_invariant(
    target: Any,
    *,
    enabled_attr: str,
    requested_attr: str = "channel_template_requested",
    forced_attr: str = "channel_template_forced_enabled",
    reason_attr: str = "channel_template_forced_reason",
    log_info: LogFn = None,
    context_label: str = "channel template",
    reason: str,
    force_enabled: bool = False,
) -> bool:
    """Normalize channel-template enablement on a config/args object.

    This helper is intentionally shared so orchestrator, CLI entry points, and
    future programmatic call paths can normalize the same invariant in one place.

    It records three audit fields on ``target``: the originally requested
    state, whether this helper had to override that request, and (only when
    forced) the reason for the override.

    Returns the originally requested enabled state.
    """
    requested = bool(getattr(target, enabled_attr, False))
    forced = bool(force_enabled and (not requested))
    if forced and log_info is not None:
        log_info(
            f"{context_label} invariant: overriding request to disable template; forcing enabled. Reason: {reason}"
        )
    setattr(target, enabled_attr, True if forced else requested)
    setattr(target, requested_attr, requested)
    setattr(target, forced_attr, forced)
    setattr(target, reason_attr, reason if forced else None)
    return requested
