"""Tests for the Apple Music Lossless provider's stream resolution."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import MediaNotFoundError

from music_assistant.providers.apple_music.provider import AppleMusicProvider
from music_assistant.providers.apple_music_lossless.provider import (
    PREFETCH,
    PREFETCH_TEMPLATE,
    AppleMusicLosslessProvider,
)
from music_assistant.providers.apple_music_lossless.stream import clear

from .test_stream import MASTER, MEDIA


@pytest.fixture
def provider() -> AppleMusicLosslessProvider:
    """Return a provider with wrapper-lite and the playlists stubbed."""
    prov = object.__new__(AppleMusicLosslessProvider)
    prov.mass = MagicMock()
    prov.config = MagicMock(instance_id="apple_music_lossless--test")
    prov.logger = MagicMock()
    config = {"lite_server": "http://lite:12340", "alac_max": 48000}
    prov.get_config_value = lambda key, default=None: config.get(key, default)  # type: ignore[method-assign,assignment]
    lite = {"/m3u8": {"m3u8": "https://aod.example/master.m3u8"}, "/key": PREFETCH_TEMPLATE}
    prov._lite = AsyncMock(side_effect=lambda endpoint, **_: lite[endpoint])  # type: ignore[method-assign]
    pages = {"https://aod.example/master.m3u8": MASTER, "https://aod.example/alac48.m3u8": MEDIA}
    prov._get_text = AsyncMock(side_effect=pages.__getitem__)  # type: ignore[method-assign]
    return prov


async def test_catalog_track_streams_lossless(provider: AppleMusicLosslessProvider) -> None:
    """A catalog track resolves to a custom ALAC stream of the allowed variant."""
    details = await provider.get_stream_details("1440857781", MediaType.TRACK)
    assert details.stream_type == StreamType.CUSTOM
    assert details.audio_format.codec_type == ContentType.ALAC
    assert (details.audio_format.sample_rate, details.audio_format.bit_depth) == (48000, 24)
    playlist = details.data
    assert playlist.file_url == "https://aod.example/song.mp4"
    # segments: bundled first-segment key, the song key (fetched up front), then clear
    assert playlist.decryptors[0] is PREFETCH
    assert playlist.decryptors[1] not in (PREFETCH, clear)
    assert playlist.decryptors[2] is clear
    cast("AsyncMock", provider._lite).assert_any_await(
        "/key", adamId="1440857781", uri="skd://itunes.apple.com/song/key2"
    )
    assert cast("AsyncMock", provider._lite).await_count == 2


async def test_wrapper_lite_failure_falls_back_to_aac(
    provider: AppleMusicLosslessProvider,
) -> None:
    """When wrapper-lite cannot help, the built-in AAC stream is used."""
    cast("AsyncMock", provider._lite).side_effect = MediaNotFoundError("song not found")
    aac = MagicMock()
    with patch.object(AppleMusicProvider, "get_stream_details", AsyncMock(return_value=aac)):
        assert await provider.get_stream_details("1440857781", MediaType.TRACK) is aac


async def test_library_track_uses_aac_path(provider: AppleMusicLosslessProvider) -> None:
    """Library uploads never go to wrapper-lite."""
    aac = MagicMock()
    with patch.object(AppleMusicProvider, "get_stream_details", AsyncMock(return_value=aac)):
        assert await provider.get_stream_details("i.AbC123", MediaType.TRACK) is aac
    cast("AsyncMock", provider._lite).assert_not_awaited()
