"""Apple Music Lossless music provider: Apple Music with ALAC streams fetched by amdl."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import AppleMusicLosslessProvider

__all__ = ["AppleMusicLosslessProvider", "setup"]

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return AppleMusicLosslessProvider(mass, manifest, config)
