"""Guards the public re-exports that the generated SDK has renamed under us.

The package constraint is ``openrouter>=1.1,<2``, so a plain install resolves
whatever 1.1.x is newest. A symbol this package re-exports by name can vanish
inside that range -- openrouter 1.1.40 renamed ``OutputImage`` -- and the
failure mode is an ImportError at ``import openrouter_agent``, i.e. the whole
package, not just the renamed type.
"""

from __future__ import annotations

import openrouter_agent


def test_output_image_is_exported_under_both_names() -> None:
    assert openrouter_agent.OutputImage is openrouter_agent.OutputInputImage


def test_output_image_has_the_code_interpreter_image_shape() -> None:
    fields = openrouter_agent.OutputImage.model_fields

    assert set(fields) == {"type", "url"}


def test_declared_public_api_is_fully_importable() -> None:
    missing = [name for name in openrouter_agent.__all__ if not hasattr(openrouter_agent, name)]

    assert missing == []
