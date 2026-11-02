"""The Python Temari port must decrypt exactly like the Go original in amdl."""

import json
from pathlib import Path

from music_assistant.providers.apple_music_lossless.provider import PREFETCH_TEMPLATE
from music_assistant.providers.apple_music_lossless.temari import Template

# produced by amdl's internal/temari Decrypt with its embedded prefetch template
VECTORS = json.loads(Path(__file__).with_name("temari_vectors.json").read_text())


def test_matches_go_temari() -> None:
    """Samples of mixed lengths decrypted as one batch match the Go output, tails untouched."""
    samples = [bytes.fromhex(v["In"]) for v in VECTORS]
    expected = [bytes.fromhex(v["Out"]) for v in VECTORS]
    assert Template(PREFETCH_TEMPLATE).decrypt(samples) == expected


def test_single_sample_matches_batch() -> None:
    """A sample decrypts the same alone as inside a batch (lanes are independent)."""
    sample = bytes.fromhex(VECTORS[-1]["In"])
    assert Template(PREFETCH_TEMPLATE).decrypt([sample]) == [bytes.fromhex(VECTORS[-1]["Out"])]
