"""Tests for lossless playlist parsing and in-memory fragmented MP4 decryption."""

import shutil
import struct
import subprocess
from collections.abc import AsyncIterator

import pytest

from music_assistant.providers.apple_music_lossless import stream
from music_assistant.providers.apple_music_lossless.stream import (
    decrypt_stream,
    fragment_samples,
    parse_media_playlist,
    pick_alac_variant,
)

MASTER = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=300000,CODECS="mp4a.40.2",AUDIO="audio-stereo-256"
aac.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=100,CODECS="alac",AUDIO="alac"
odd.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,CODECS="alac",AUDIO="audio-alac-stereo-44100-16"
alac44.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2500000,CODECS="alac",AUDIO="audio-alac-stereo-48000-24"
alac48.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=9000000,CODECS="alac",AUDIO="audio-alac-stereo-192000-24"
alac192.m3u8
"""

MEDIA = """#EXTM3U
#EXT-X-MAP:URI="song.mp4",BYTERANGE="700@0"
#EXT-X-KEY:METHOD=SAMPLE-AES,URI="skd://itunes.apple.com/P000000000/s1/e1",KEYFORMAT="com.apple.streamingkeydelivery",KEYFORMATVERSIONS="1"
#EXT-X-KEY:METHOD=SAMPLE-AES,URI="data:text/plain;base64,AAAA",KEYFORMAT="urn:uuid:edef8ba9"
#EXTINF:6.0,
#EXT-X-BYTERANGE:1000@700
song.mp4
#EXT-X-KEY:METHOD=SAMPLE-AES,URI="skd://itunes.apple.com/song/key2",KEYFORMAT="com.apple.streamingkeydelivery",KEYFORMATVERSIONS="1"
#EXTINF:6.0,
#EXT-X-BYTERANGE:1000@1700
song.mp4
#EXT-X-KEY:METHOD=NONE
#EXTINF:6.0,
#EXT-X-BYTERANGE:1000@2700
song.mp4
"""


@pytest.mark.parametrize(
    ("max_rate", "expected"),
    [
        (192000, ("alac192.m3u8", 192000, 24)),
        (48000, ("alac48.m3u8", 48000, 24)),
        (44100, ("alac44.m3u8", 44100, 16)),
        (22050, None),
    ],
)
def test_pick_alac_variant(max_rate: int, expected: tuple[str, int, int] | None) -> None:
    """The best ALAC variant within the limit wins; AAC variants are ignored."""
    assert pick_alac_variant(MASTER, max_rate) == expected


def test_parse_media_playlist_follows_fairplay_keys() -> None:
    """Each segment gets its FairPlay key; other DRM systems' keys are skipped."""
    url, keys = parse_media_playlist(MEDIA, "https://aod.itunes.apple.com/a/b/alac48.m3u8")
    assert url == "https://aod.itunes.apple.com/a/b/song.mp4"
    assert keys == [
        "skd://itunes.apple.com/P000000000/s1/e1",
        "skd://itunes.apple.com/song/key2",
        None,
    ]


def _box(box_type: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body) + 8) + box_type + body


def _children(box: bytes, start: int) -> list[bytes]:
    out = []
    while start < len(box):
        size = struct.unpack_from(">I", box, start)[0]
        out.append(box[start : start + size])
        start += size
    return out


def _protect(box: bytes) -> bytes:
    """Rebuild a clear moov/moof the way Apple ships it: enca entries, senc, seig groups."""
    box_type = box[4:8]
    if box_type == b"stsd":
        entry = _children(box, 16)[0]
        sinf = _box(
            b"sinf",
            _box(b"frma", b"alac")
            + _box(b"schm", b"\0\0\0\0cbcs\0\1\0\0")
            + _box(b"schi", _box(b"tenc", bytes(24))),
        )
        enca = _box(b"enca", entry[8:] + sinf)
        return _box(b"stsd", box[8:12] + struct.pack(">I", 2) + enca + enca)
    if box_type == b"traf":
        count = len(_children(box, 8))  # any value; senc content is never read
        senc = _box(b"senc", struct.pack(">II", 0, count))
        sgpd = _box(b"sgpd", b"\1\0\0\0seig" + bytes(12))
        return _box(b"traf", b"".join(_protect(c) for c in _children(box, 8)) + senc + sgpd)
    if box_type in stream.CONTAINERS:
        return _box(box_type, b"".join(_protect(c) for c in _children(box, 8)))
    return box


def _xor(data: bytes | bytearray) -> bytes:
    """Stand-in cipher: like FairPlay cbcs, only the 16-byte-aligned prefix is scrambled."""
    head = len(data) // 16 * 16
    return bytes(b ^ 0x5A for b in data[:head]) + data[head:]


def _encrypt_file(clear: bytes) -> bytes:
    """Turn a clear fragmented ALAC MP4 into an Apple-style encrypted one."""
    out, pos = b"", 0
    while pos < len(clear):
        size = struct.unpack_from(">I", clear, pos)[0]
        box = clear[pos : pos + size]
        if box[4:8] == b"moov":
            box = _protect(box)
        elif box[4:8] == b"moof":
            mdat_size = struct.unpack_from(">I", clear, pos + size)[0]
            frag = bytearray(clear[pos : pos + size + mdat_size])
            for offset, length in fragment_samples(frag, 0):
                frag[offset : offset + length] = _xor(frag[offset : offset + length])
            moof = bytearray(_protect(box))
            # protecting grew the moof, so the samples moved by the same amount
            stream.shift_data_offsets(moof, len(moof) - size)
            box = bytes(moof) + bytes(frag[size:])
            size += mdat_size
        out += box
        pos += size
    return out


def _ffmpeg(args: list[str], data: bytes | None = None) -> bytes:
    return subprocess.run(  # noqa: S603
        ["ffmpeg", "-v", "error", *args],  # noqa: S607
        input=data,
        capture_output=True,
        check=True,
    ).stdout


async def _chunks(data: bytes, size: int = 1000) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
async def test_decrypted_stream_decodes_to_the_original_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An Apple-style encrypted ALAC file comes out as clear ALAC that ffmpeg decodes."""
    sine = ["-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100", "-ac", "2"]
    clear = _ffmpeg(
        [
            *sine,
            *("-c:a", "alac", "-f", "mp4", "-frag_duration", "500000"),
            *("-movflags", "+frag_keyframe+empty_moov+default_base_moof", "pipe:1"),
        ]
    )
    reference = _ffmpeg([*sine, "-f", "s16le", "pipe:1"])
    encrypted = _encrypt_file(clear)
    assert encrypted != clear
    frag_count = encrypted.count(b"moof")
    used: list[str] = []

    def decryptor(key: str) -> stream.Decryptor:
        def decrypt(samples: list[bytes]) -> list[bytes]:
            used.append(key)
            return [_xor(s) for s in samples]

        return decrypt

    # small batches so key switches and batch flushes are both exercised
    monkeypatch.setattr(stream, "MIN_BATCH_BYTES", 4000)
    key_b = decryptor("key-b")
    decryptors = [decryptor("key-a")] + [key_b] * (frag_count - 1)
    out = b"".join([c async for c in decrypt_stream(_chunks(encrypted), decryptors)])

    assert b"enca" not in out
    assert b"senc" not in out
    assert used[0] == "key-a"
    assert set(used[1:]) == {"key-b"}
    pcm = _ffmpeg(["-f", "mp4", "-i", "pipe:0", "-f", "s16le", "pipe:1"], out)
    assert pcm == reference


async def test_more_fragments_than_playlist_segments_fails() -> None:
    """A file that does not match its playlist is rejected instead of misdecrypted."""
    moof = _box(b"moof", _box(b"mfhd", bytes(8)))
    data = _box(b"ftyp", b"M4A ") + moof + _box(b"mdat", b"") + moof + _box(b"mdat", b"")
    with pytest.raises(ValueError, match="more fragments"):
        _ = [c async for c in decrypt_stream(_chunks(data), [stream.clear])]
