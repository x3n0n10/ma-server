"""Apple Music provider that streams lossless ALAC instead of the 256kbps AAC stream."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from aiohttp import ClientError, ClientTimeout
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ContentType, MediaType, StreamType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.helpers.throttle_retry import Throttler
from music_assistant.providers.apple_music.helpers.utils import is_library_id
from music_assistant.providers.apple_music.provider import AppleMusicProvider

from .stream import Decryptor, clear, decrypt_stream, parse_media_playlist, pick_alac_variant
from .temari import Template

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

CONF_LITE_SERVER = "lite_server"
CONF_ALAC_MAX = "alac_max"
DEFAULT_ALAC_MAX = 48000
# the first segments of every song share one well-known key; amdl ships its template
PREFETCH_TEMPLATE = json.loads(Path(__file__).with_name("prefetch_template.json").read_text())
PREFETCH = Template(PREFETCH_TEMPLATE).decrypt
LITE_TIMEOUT = ClientTimeout(total=30)


@dataclass
class LosslessPlaylist:
    """Where to fetch a song's lossless file, and how to decrypt each of its segments."""

    file_url: str
    decryptors: list[Decryptor]


class AppleMusicLosslessProvider(AppleMusicProvider):
    """Apple Music, with catalog tracks streamed in ALAC via a wrapper-lite instance."""

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the Apple Music entries plus the wrapper-lite settings."""
        return (
            *await super().get_config_entries(),
            ConfigEntry(
                key=CONF_LITE_SERVER,
                type=ConfigEntryType.STRING,
                default_value="http://127.0.0.1:12340",
                required=True,
            ),
            ConfigEntry(
                key=CONF_ALAC_MAX,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_ALAC_MAX,
                options=[
                    ConfigValueOption(r, title=f"{r} Hz") for r in (44100, 48000, 96000, 192000)
                ],
                required=True,
            ),
        )

    async def handle_async_init(self) -> None:
        """Set up the Apple Music session and the wrapper-lite throttle."""
        await super().handle_async_init()
        self._lite_throttler = Throttler(rate_limit=2, period=1)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return a lossless stream for catalog tracks, falling back to AAC."""
        # library ids are the user's own (unencrypted) uploads, which the AAC path serves
        if media_type != MediaType.TRACK or is_library_id(item_id):
            return await super().get_stream_details(item_id, media_type)
        try:
            playlist, rate, depth = await self._lossless_playlist(item_id)
        except (MediaNotFoundError, ClientError, TimeoutError, ValueError, KeyError) as err:
            self.logger.warning("No lossless stream for %s, using AAC: %s", item_id, err)
            return await super().get_stream_details(item_id, media_type)
        return StreamDetails(
            item_id=item_id,
            provider=self.instance_id,
            stream_type=StreamType.CUSTOM,
            audio_format=AudioFormat(
                content_type=ContentType.MP4,
                codec_type=ContentType.ALAC,
                sample_rate=rate,
                bit_depth=depth,
            ),
            data=playlist,
            # the core seeks by skipping through the stream; the file has no seek index here
            can_seek=False,
            allow_seek=True,
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """Stream the lossless file, decrypting it in memory as it arrives."""
        playlist: LosslessPlaylist = streamdetails.data
        async with self.mass.http_session.get(playlist.file_url) as response:
            response.raise_for_status()
            async for chunk in decrypt_stream(
                response.content.iter_chunked(1 << 16), playlist.decryptors
            ):
                yield chunk

    async def _lossless_playlist(self, song_id: str) -> tuple[LosslessPlaylist, int, int]:
        """Resolve the best allowed ALAC variant of a song via wrapper-lite."""
        master_url = (await self._lite("/m3u8", adamId=song_id))["m3u8"]
        if not master_url:
            raise MediaNotFoundError("wrapper-lite returned no playlist")
        max_rate = int(self.get_config_value(CONF_ALAC_MAX, DEFAULT_ALAC_MAX))
        variant = pick_alac_variant(await self._get_text(master_url), max_rate)
        if not variant:
            raise MediaNotFoundError(f"no ALAC variant up to {max_rate} Hz")
        media_url = urljoin(master_url, variant[0])
        file_url, keys = parse_media_playlist(await self._get_text(media_url), media_url)
        # fetched now, so a key wrapper-lite cannot provide still falls back to AAC
        decryptors: dict[str | None, Decryptor] = {
            None: clear,
            PREFETCH_TEMPLATE["keyUri"]: PREFETCH,
        }
        for uri in dict.fromkeys(k for k in keys if k and k not in decryptors):
            decryptors[uri] = Template(await self._lite("/key", adamId=song_id, uri=uri)).decrypt
        return LosslessPlaylist(file_url, [decryptors[k] for k in keys]), variant[1], variant[2]

    async def _lite(self, endpoint: str, **params: str) -> dict[str, Any]:
        """Call a wrapper-lite endpoint and return its data payload."""
        url = str(self.get_config_value(CONF_LITE_SERVER)).rstrip("/") + endpoint
        async with (
            self._lite_throttler,
            self.mass.http_session.get(url, params=params, timeout=LITE_TIMEOUT) as resp,
        ):
            resp.raise_for_status()
            body = await resp.json(content_type=None)
        if not isinstance(body, dict) or body.get("code") != 0:
            raise MediaNotFoundError(f"wrapper-lite {endpoint} failed: {body}")
        data: dict[str, Any] = body["data"]
        return data

    async def _get_text(self, url: str) -> str:
        """Fetch a playlist."""
        async with self.mass.http_session.get(url, timeout=LITE_TIMEOUT) as resp:
            resp.raise_for_status()
            return await resp.text()
