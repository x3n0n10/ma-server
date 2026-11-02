"""Lossless HLS playlist parsing and in-memory fragmented MP4 decryption."""

from __future__ import annotations

import asyncio
import re
import struct
from collections.abc import AsyncIterator, Callable, Sequence
from urllib.parse import urljoin

from music_assistant.helpers.playlists import parse_m3u

# fragments are decrypted in batches of at least this many bytes: decryption time grows
# with sample length but barely with the number of samples, so wide batches are cheap
MIN_BATCH_BYTES = 2 << 20

Decryptor = Callable[[list[bytes]], list[bytes]]


def clear(samples: list[bytes]) -> list[bytes]:
    """Pass the samples of an unencrypted segment through."""
    return samples


def pick_alac_variant(master_m3u8: str, max_rate: int) -> tuple[str, int, int] | None:
    """Return (uri, sample rate, bit depth) of the best ALAC variant within max_rate."""
    best: tuple[str, int, int] | None = None
    for item in parse_m3u(master_m3u8):
        info = item.stream_info or {}
        if info.get("CODECS", "").strip('"') != "alac":
            continue
        # group ids look like audio-alac-stereo-48000-24
        if match := re.search(r"-(\d+)-(\d+)\"?$", info.get("AUDIO", "")):
            rate, depth = map(int, match.groups())
            if rate <= max_rate and (not best or (rate, depth) > best[1:]):
                best = (item.path, rate, depth)
    return best


def parse_media_playlist(m3u8: str, base_url: str) -> tuple[str, list[str | None]]:
    """Return the single byte-range file URL and the FairPlay key URI of each segment."""
    file_uri: str | None = None
    key_uri: str | None = None
    keys: list[str | None] = []
    for line in m3u8.splitlines():
        line = line.strip()  # noqa: PLW2901
        # every segment carries one EXT-X-KEY per DRM system; only FairPlay is decryptable
        if line.startswith("#EXT-X-KEY:METHOD=NONE"):
            key_uri = None
        elif line.startswith("#EXT-X-KEY:") and "streamingkeydelivery" in line:
            if match := re.search(r'URI="([^"]+)"', line):
                key_uri = match[1]
        elif line and not line.startswith("#"):
            if file_uri not in (None, line):
                raise ValueError("lossless playlist spans several files")
            file_uri = line
            keys.append(key_uri)
    if not file_uri:
        raise ValueError("lossless playlist has no segments")
    return urljoin(base_url, file_uri), keys


async def iter_boxes(chunks: AsyncIterator[bytes]) -> AsyncIterator[tuple[bytes, bytes]]:
    """Yield (type, whole box) for each top-level MP4 box in a byte stream."""
    buf = bytearray()
    async for chunk in chunks:
        buf += chunk
        while len(buf) >= 8:
            size, box_type = struct.unpack_from(">I4s", buf)
            if size == 1:
                if len(buf) < 16:
                    break
                size = struct.unpack_from(">Q", buf, 8)[0]
            if size < 8:
                raise ValueError(f"unsupported MP4 box size {size}")
            if len(buf) < size:
                break
            yield box_type, bytes(buf[:size])
            del buf[:size]
    if buf:
        raise ValueError("stream ended inside an MP4 box")


def _children(data: bytes | bytearray, box: int, header: int = 8) -> dict[bytes, list[int]]:
    """Map the child box types of the box at offset box to their offsets in data."""
    start, end = box + header, box + struct.unpack_from(">I", data, box)[0]
    found: dict[bytes, list[int]] = {}
    while start + 8 <= end:
        size, box_type = struct.unpack_from(">I4s", data, start)
        if size < 8:
            raise ValueError("bad MP4 child box")
        found.setdefault(box_type, []).append(start)
        start += size
    return found


def trex_sample_size(moov: bytes) -> int:
    """Return the default sample size from moov/mvex/trex (0 when absent)."""
    mvex = _children(moov, 0).get(b"mvex")
    if not mvex:
        return 0
    trex = _children(moov, mvex[0])
    return struct.unpack_from(">I", moov, trex[b"trex"][0] + 24)[0] if b"trex" in trex else 0


def fragment_samples(frag: bytes | bytearray, default_size: int) -> list[tuple[int, int]]:
    """Return (offset, size) of each sample of a moof(+mdat) fragment, offsets into frag."""
    samples: list[tuple[int, int]] = []
    for traf in _children(frag, 0).get(b"traf", []):
        boxes = _children(frag, traf)
        tfhd = boxes[b"tfhd"][0]
        flags = struct.unpack_from(">I", frag, tfhd + 8)[0] & 0xFFFFFF
        if flags & 0x1:
            raise ValueError("absolute base data offsets are not supported")
        pos = tfhd + 16 + (4 if flags & 0x2 else 0) + (4 if flags & 0x8 else 0)
        size = struct.unpack_from(">I", frag, pos)[0] if flags & 0x10 else default_size
        offset = 0  # the traf data starts at the moof (default-base-is-moof)
        for trun in boxes.get(b"trun", []):
            tflags = struct.unpack_from(">I", frag, trun + 8)[0] & 0xFFFFFF
            count = struct.unpack_from(">I", frag, trun + 12)[0]
            pos = trun + 16
            if tflags & 0x1:
                offset = struct.unpack_from(">i", frag, pos)[0]
                pos += 4
            pos += 4 if tflags & 0x4 else 0
            fields = [f for f in (0x100, 0x200, 0x400, 0x800) if tflags & f]
            for _ in range(count):
                sample_size = size
                for field in fields:
                    if field == 0x200:
                        sample_size = struct.unpack_from(">I", frag, pos)[0]
                    pos += 4
                if not sample_size:
                    raise ValueError("fragment has no sample sizes")
                samples.append((offset, sample_size))
                offset += sample_size
    if samples and samples[-1][0] + samples[-1][1] > len(frag):
        raise ValueError("fragment samples run past the mdat")
    return samples


CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"moof", b"traf"}
ENCRYPTION_BOXES = {b"sinf", b"senc", b"saiz", b"saio", b"pssh"}
ENCRYPTION_GROUPS = {b"seig", b"seam"}


def strip_encryption(box: bytes) -> bytes:
    """Rebuild a moov or moof without its encryption boxes, as a clear ALAC track."""
    box_type = box[4:8]
    if box_type in ENCRYPTION_BOXES:
        return b""
    if box_type in (b"sbgp", b"sgpd") and box[12:16] in ENCRYPTION_GROUPS:
        return b""
    if box_type == b"stsd":
        # keep the first sample entry only; Apple repeats the same entry twice
        first = _children(box, 0, header=16)
        entry = min(o for offsets in first.values() for o in offsets)
        size = struct.unpack_from(">I", box, entry)[0]
        body = box[8:12] + struct.pack(">I", 1) + strip_encryption(box[entry : entry + size])
    elif box_type == b"enca":
        # 28 bytes of audio sample entry fields, then children (alac cookie, sinf)
        sinf = _children(box, 0, header=36)[b"sinf"][0]
        frma = _children(box, sinf)[b"frma"][0]
        box_type = box[frma + 8 : frma + 12]
        body = box[8:36] + _strip_children(box, 36)
    elif box_type in CONTAINERS:
        body = _strip_children(box, 8)
    else:
        return box
    return struct.pack(">I", len(body) + 8) + box_type + body


def _strip_children(box: bytes, start: int) -> bytes:
    """Return the children of box from start on, each stripped of encryption."""
    out = b""
    while start + 8 <= len(box):
        size = struct.unpack_from(">I", box, start)[0]
        out += strip_encryption(box[start : start + size])
        start += size
    return out


def decrypt_fragments(frags: list[bytes], default_size: int, decrypt: Decryptor) -> bytes:
    """Decrypt several moof+mdat fragments in one batch and return them as clear ALAC."""
    out = [bytearray(f) for f in frags]
    where = [(i, o, s) for i, f in enumerate(out) for o, s in fragment_samples(f, default_size)]
    plain = decrypt([bytes(out[i][o : o + s]) for i, o, s in where])
    for (i, o, s), sample in zip(where, plain, strict=True):
        out[i][o : o + s] = sample
    return b"".join(_strip_fragment(bytes(f)) for f in out)


def _strip_fragment(frag: bytes) -> bytes:
    """Strip a fragment's moof and move its sample data offsets by the bytes removed."""
    moof_size = struct.unpack_from(">I", frag)[0]
    moof = bytearray(strip_encryption(frag[:moof_size]))
    shift_data_offsets(moof, len(moof) - moof_size)
    return bytes(moof) + frag[moof_size:]


def shift_data_offsets(moof: bytearray, delta: int) -> None:
    """Move every trun data offset of a moof by delta bytes, after the moof changed size."""
    for traf in _children(moof, 0).get(b"traf", []):
        for trun in _children(moof, traf).get(b"trun", []):
            if struct.unpack_from(">I", moof, trun + 8)[0] & 0x1:
                offset = struct.unpack_from(">i", moof, trun + 16)[0]
                struct.pack_into(">i", moof, trun + 16, offset + delta)


async def decrypt_stream(
    chunks: AsyncIterator[bytes],
    decryptors: Sequence[Decryptor],
) -> AsyncIterator[bytes]:
    """
    Turn the encrypted lossless file into a clear fragmented ALAC MP4 stream.

    :param decryptors: The decryptor of each playlist segment, in order.
    """
    default_size = 0
    moof: bytes | None = None
    batch: list[bytes] = []
    batch_decrypt: Decryptor = clear
    batch_bytes = 0
    segment = 0

    async def flush() -> bytes:
        nonlocal batch, batch_bytes
        data = await asyncio.to_thread(decrypt_fragments, batch, default_size, batch_decrypt)
        batch, batch_bytes = [], 0
        return data

    async for box_type, box in iter_boxes(chunks):
        if box_type == b"ftyp":
            yield box
        elif box_type == b"moov":
            default_size = trex_sample_size(box)
            yield strip_encryption(box)
        elif box_type == b"moof":
            moof = box
        elif box_type == b"mdat" and moof:
            # fragment n of the file is segment n of the playlist, which names its key
            if segment >= len(decryptors):
                raise ValueError("lossless file has more fragments than the playlist")
            decrypt = decryptors[segment]
            segment += 1
            if batch and decrypt is not batch_decrypt:
                yield await flush()
            batch.append(moof + box)
            batch_decrypt, moof = decrypt, None
            batch_bytes += len(batch[-1])
            if batch_bytes >= MIN_BATCH_BYTES:
                yield await flush()
    if batch:
        yield await flush()
