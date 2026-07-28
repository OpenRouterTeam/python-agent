"""Normalize a `hooks` option into a `HooksManager` instance. Mirrors
upstream `hooks-resolve.ts`."""

from __future__ import annotations

from typing import Optional, Union

from .hooks_manager import HooksManager
from .hooks_schemas import BUILT_IN_HOOK_NAMES
from .hooks_types import HookEntry, InlineHookConfig


def resolve_hooks(hooks: Optional[Union[InlineHookConfig, HooksManager]]) -> Optional[HooksManager]:
    """
    - `None` -> `None` (no hooks)
    - `HooksManager` -> passthrough
    - plain dict (`InlineHookConfig`) -> construct a `HooksManager` and
      register every entry

    Any non-built-in key in the inline config is ignored with a warning:
    inline config only supports built-in hooks, custom hooks must be
    registered through a `HooksManager` instance via `on()`.
    """
    if not hooks:
        return None

    if isinstance(hooks, HooksManager):
        return hooks

    manager = HooksManager()
    for hook_name, entries in hooks.items():
        if not entries:
            continue
        if hook_name not in BUILT_IN_HOOK_NAMES:
            import warnings

            warnings.warn(
                f'[resolve_hooks] Ignoring inline hook entry for unknown hook name "{hook_name}". '
                "Inline config only supports built-in hooks; register custom hooks via a HooksManager instance.",
                stacklevel=2,
            )
            continue
        for entry in entries:
            manager.on(hook_name, entry if isinstance(entry, HookEntry) else HookEntry(**entry))

    return manager
