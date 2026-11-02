"""
Apple Music FairPlay sample decryption driven by a wrapper-lite key template.

Python/NumPy port of Temari (https://github.com/WorldObservationLog/Temari, MIT) as
ported to Go in zhaarey/apple-music-downloader (internal/temari, MIT). The round chain
below is machine-transpiled from that Go code; do not edit it by hand.

Every value is a NumPy uint32 array holding one lane per sample, so all samples of a
fragment are decrypted together: each 16-byte block runs the chain once for every lane.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# ruff: noqa: ANN201, ARG001, N802, N806, N814, PLR0913
from __future__ import annotations

import base64
import re
from types import SimpleNamespace as NS
from typing import Any

import numpy as np

CTX_SIZE = 0x8000
ST_SIZE = 2048
ST_SLOTS = (
    0x30,
    0x40,
    0x48,
    0x50,
    0x58,
    0x5A,
    0x60,
    0x68,
    0x70,
    0x72,
    0x78,
    0x80,
    0x88,
    0x90,
    0x96,
    0x98,
    0xA0,
    0xB0,
    0xC0,
    0xC8,
    0xD0,
    0xD8,
    0xE0,
    0xE8,
    0xF0,
    0xF8,
    0x100,
    0x104,
    0x108,
    0x112,
    0x118,
    0x120,
    0x130,
    0x144,
    0x150,
    0x152,
    0x158,
    0x160,
    0x168,
    0x170,
    0x178,
    0x180,
    0x188,
    0x190,
    0x192,
    0x198,
    0x1A0,
    0x1A8,
    0x1B0,
    0x1B8,
    0x1C0,
    0x200,
    0x208,
    0x210,
    0x216,
    0x218,
    0x220,
    0x224,
    0x228,
    0x230,
    0x232,
    0x238,
    0x240,
    0x248,
    0x250,
    0x256,
    0x258,
    0x260,
    0x264,
    0x268,
    0x270,
    0x280,
    0x288,
    0x290,
    0x298,
    0x2A0,
    0x2A8,
    0x2B0,
    0x2B8,
    0x2C0,
    0x2C8,
    0x2D0,
    0x2D8,
    0x2E0,
    0x2E8,
    0x2F0,
    0x2F8,
    0x300,
    0x304,
    0x308,
    0x310,
    0x318,
    0x320,
    0x328,
    0x330,
    0x336,
    0x338,
    0x340,
    0x344,
    0x348,
    0x350,
    0x352,
    0x358,
    0x360,
    0x368,
    0x370,
    0x376,
    0x378,
    0x380,
    0x384,
    0x388,
    0x390,
    0x392,
    0x398,
    0x3B0,
    0x3B8,
    0x3C0,
    0x3C8,
    0x3D0,
    0x3E0,
    0x3E8,
    0x3F8,
    0x400,
    0x408,
    0x410,
    0x416,
    0x418,
    0x420,
    0x424,
    0x428,
    0x430,
    0x432,
    0x440,
    0x448,
    0x450,
    0x460,
    0x468,
    0x470,
    0x478,
    0x480,
    0x488,
    0x490,
    0x4D8,
    0x4E0,
    0x4E8,
    0x4F0,
    0x4F8,
    0x500,
    0x508,
    0x510,
    0x512,
    0x518,
    0x520,
    0x528,
    0x530,
    0x536,
    0x538,
    0x540,
    0x548,
    0x550,
    0x552,
    0x558,
    0x560,
    0x568,
    0x570,
    0x576,
    0x578,
    0x580,
    0x584,
    0x588,
    0x590,
    0x592,
    0x598,
    0x5A0,
    0x5A8,
    0x5B0,
    0x5B8,
    0x5C0,
    0x5C8,
    0x5D8,
    0x5E8,
    0x5F0,
    0x5F8,
    0x600,
    0x608,
    0x610,
    0x616,
    0x618,
    0x620,
    0x624,
    0x628,
    0x638,
    0x640,
    0x648,
    0x656,
    0x658,
    0x660,
    0x664,
    0x668,
    0x670,
    0x672,
    0x678,
    0x680,
    0x688,
    0x690,
    0x696,
    0x698,
    0x6A8,
    0x6B0,
    0x6B8,
    0x6C0,
    0x6C8,
    0x6D8,
    0x6E0,
    0x6E8,
    0x6F0,
    0x6F8,
    0x700,
    0x708,
    0x720,
    0x728,
    0x736,
    0x740,
    0x752,
    0x760,
    0x768,
    0x770,
    0x778,
    0x780,
    0x792,
    0x816,
    0x824,
    0x872,
    0x912,
    0x944,
    0x952,
    0x960,
    0x992,
    0x1040,
    0x1088,
    0x1328,
    0x1360,
    0x1368,
    0x1384,
    0x1408,
    0x1416,
    0x1424,
    0x1432,
    0x1440,
    0x1448,
    0x1456,
    0x1480,
    0x1512,
    0x1528,
    0x1536,
    0x1552,
    0x1560,
    0x1568,
    0x1576,
    0x1632,
    0x1704,
    0x1720,
    0x1760,
)
U32_TABLES = (12272, 13328, 14672, 15728, 17040, 18080, 19136, 20208)
BYTE_TABLES = (12000, 16768, 21248)


class Template:
    """A decryption template from wrapper-lite's /key endpoint (immutable)."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Build the lookup tables and initial state from the /key JSON payload."""
        ctx = _b64(data["ctx"])
        state = _b64(data["state"])
        if len(ctx) < CTX_SIZE or len(state) < 0x2000:
            raise ValueError("key template too short")
        tables = {f"t{o}": np.frombuffer(ctx, "<u4", 256, o).astype(np.uint32) for o in U32_TABLES}
        tables |= {
            f"s{o}": np.frombuffer(ctx, np.uint8, 256, o).astype(np.uint32) for o in BYTE_TABLES
        }
        self.ctx = NS(c14573=np.uint32(ctx[14573]), **tables)
        self.r1Entry = NS(
            rcx=np.uint32(int(data.get("rcx") or "0", 16)),
            rax=np.uint32(int(data.get("rax") or "0", 16)),
        )
        self.st_init = [np.uint32(0)] * ST_SIZE
        for off in ST_SLOTS:
            if off < ST_SIZE and (pos := 0x2000 - off) + 4 <= len(state):
                self.st_init[off] = np.uint32(int.from_bytes(state[pos : pos + 4], "little"))

    def decrypt(self, samples: list[bytes]) -> list[bytes]:
        """Decrypt the 16-byte-aligned prefix of each sample; tails pass through."""
        lengths = [len(s) // 16 * 16 for s in samples]
        width = max(lengths, default=0)
        if not width:
            return list(samples)
        # column-major: the chain reads and writes one byte column of every lane at a time
        ct = np.zeros((len(samples), width), np.uint32, order="F")
        for i, (s, n) in enumerate(zip(samples, lengths, strict=True)):
            ct[i, :n] = np.frombuffer(s, np.uint8, n)
        out = np.empty((len(samples), width), np.uint8, order="F")
        st = list(self.st_init)
        with np.errstate(over="ignore"):
            for bi in range(width // 16):
                out16: list[Any] = [0] * 16
                processBlock(self, st, ct, bi, out16)
                for k, v in enumerate(out16):
                    out[:, bi * 16 + k] = v
        return [
            out[i, :n].tobytes() + s[n:]
            for i, (s, n) in enumerate(zip(samples, lengths, strict=True))
        ]


def _b64(s: str) -> bytes:
    """Decode leniently like upstream: stop at the first '=', skip non-alphabet bytes."""
    s = re.sub(r"[^A-Za-z0-9+/]", "", s.split("=", 1)[0])
    return base64.b64decode(s[: len(s) // 4 * 4 + (len(s) % 4 if len(s) % 4 > 1 else 0)] + "==")


def t4(tab: Any, idx: Any) -> Any:
    """Look a value up by the low byte of idx."""
    return tab.take(idx & 255)


ctxb = t4


def uint32(x: Any) -> Any:
    """Convert like Go's uint32 cast."""
    return np.uint32(x) if isinstance(x, int) else x


def uint8(x: Any) -> Any:
    """Convert like Go's uint8 cast."""
    return x & 255


def round1Mid(ctx, st, ct, rdi, rsi, rcx, rax):  # noqa: D103, PLR0915
    rdi_2 = rdi
    rcx_4 = ctxb(ctx.s12000, rcx) ^ 77
    rax_3 = rax ^ 77
    rcx_8 = rcx_4 << 16
    rax_7 = (((rax_3 << 8) ^ 0x80004300) + 0x2EFF0000) ^ 0xAEFF4300
    r10_3 = uint32(ctx.c14573)
    rax_10 = ctxb(ctx.s16768, uint32(ct[:, int((rsi & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[304] = rax_10
    rax_11 = rax_10 & 255
    rsi_8 = (rdi_2 | 9) ^ 0x1EB2C6B4
    st[344] = rsi_8
    rbx_4 = ctxb(ctx.s16768, uint32(ct[:, int((rsi_8 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[360] = rbx_4
    rsi_10 = rbx_4 & 255
    rdx_17 = rax_7 | rcx_8
    rax_15 = rax_11 << 16
    rsi_14 = rsi_10 << 8
    rcx_24 = rsi_14 | rax_15
    rax_24 = (rdi_2 | 10) ^ 0x1EB2C6B4
    st[424] = rax_24
    rax_27 = ctxb(ctx.s16768, uint32(ct[:, int((rax_24 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[368] = rax_27
    rax_28 = rax_27 & 255
    rax_32 = rax_28 | rcx_24
    rcx_32 = ctxb(ctx.s12000, st[1280]) ^ 77
    rcx_36 = rcx_32 | rdx_17
    rax_36 = rax_32 << 8
    rcx_40 = rcx_36 << 8
    rdx_31 = (rdi_2 | 11) ^ 0x1EB2C6B4
    st[416] = rdx_31
    rdx_34 = ctxb(ctx.s16768, uint32(ct[:, int((rdx_31 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[384] = rdx_34
    rsi_22 = rdx_34 & 255
    st[144] = rsi_22
    rdx_41 = rsi_22 | rax_36
    rdx_43 = ctxb(ctx.s12000, st[1320])
    st[72] = rdx_43
    rdx_44 = rdx_43 ^ 77
    rdx_48 = rdx_44 | rcx_40
    rdx_51 = (rdx_48 ^ 0x49058B58) ^ (rdx_41 ^ 0x3AD4AE12)
    st[80] = rdx_51
    st[600] = (rdx_51 >> 8) & 255
    st[536] = rdx_51
    rax_49 = (rdi_2 | 6) ^ 0x1EB2C6B0
    rax_52 = (rdi_2 | 3) ^ 0x1EB2C6B4
    rax_55 = (rdi_2 | 14) ^ 0x1EB2C6B0
    rax_58 = (rdi_2 | 5) ^ 0x1EB2C6B0
    st[176] = ((rax_58 ^ 0x97A344B0) + 0x685CBB4D) + 0x17A344B2
    rax_63 = (rdi_2 | 13) ^ 0x1EB2C6B0
    rax_66 = (rdi_2 | 15) ^ 0x1EB2C6B0
    rax_69 = ctxb(ctx.s12000, st[776]) ^ 77
    rcx_70 = (rdi_2 | 12) ^ 0x1EB2C6B0
    st[448] = rcx_70
    rcx_73 = ctxb(ctx.s16768, uint32(ct[:, int((rcx_70 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[1088] = rcx_73
    rcx_74 = rcx_73 & 255
    rdx_58 = ctxb(ctx.s12000, st[1672]) ^ 77
    rdi_7 = (((rax_63 ^ 0xBDE122A2) + 0x421EDD5B) + 0x3DE122A4) ^ 0x3DE122A3
    st[440] = rdi_7
    rbx_6 = ctxb(ctx.s16768, uint32(ct[:, int((rdi_7 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[584] = rbx_6
    rsi_30 = rbx_6 & 255
    rcx_78 = rcx_74 << 16
    rsi_34 = rsi_30 << 8
    rax_73 = rax_69 << 16
    rdx_62 = rdx_58 << 8
    rdx_66 = rdx_62 | rax_73
    rsi_38 = rsi_34 | rcx_78
    rax_85 = ctxb(ctx.s12000, st[1312]) ^ 77
    rax_89 = rax_85 | rdx_66
    r14_5 = (rdi_2 | 4) ^ 0x1EB2C6B0
    st[408] = r14_5
    rdx_71 = rdi_2 ^ 0x1EB2C6B4
    st[120] = rdx_71
    rsi_46 = (rdi_2 | 7) ^ 0x1EB2C6B0
    st[608] = rsi_46
    r15_5 = (((rax_52 ^ 0xB8663664) + 0x4799C999) + 0x38663666) ^ 0x38663665
    st[528] = r15_5
    r11_5 = (rdi_2 | 1) ^ 0x1EB2C6B4
    st[560] = r11_5
    r13_5 = rax_49
    st[432] = r13_5
    st[152] = (rdi_2 | 2) ^ 0x1EB2C6B4
    r9_5 = (((rax_55 ^ 0x9CA0BC11) + 0x635F43DE) + 0x1CA0BC1A) ^ 0x1CA0BC19
    st[696] = r9_5
    st[96] = ctxb(ctx.s16768, uint32(ct[:, int((rsi_46 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    rdx_77 = ctxb(ctx.s16768, uint32(ct[:, int((r13_5 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[552] = rdx_77
    r8_5 = st[176] ^ 0x17A344B1
    st[176] = r8_5
    r12_5 = (((rax_66 ^ 0xF3042570) + 0xCFBDA83) + 0x73042577) ^ 0x73042576
    st[736] = r12_5
    st[88] = ctxb(ctx.s16768, uint32(ct[:, int((rdx_71 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[336] = ctxb(ctx.s16768, uint32(ct[:, int((r15_5 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    r15_7 = ctxb(ctx.s16768, uint32(ct[:, int((r14_5 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[568] = (st[568] & 0xFFFFFF00) | (r15_7 & 255)
    st[104] = ctxb(ctx.s16768, uint32(ct[:, int((r11_5 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    rbx_12 = ctxb(ctx.s16768, uint32(ct[:, int((r9_5 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[576] = rbx_12
    rdx_91 = ctxb(ctx.s16768, uint32(ct[:, int((r8_5 & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    st[512] = rdx_91
    st[112] = ctxb(ctx.s16768, uint32(ct[:, int((st[152] & 4095) + (rsi & 0xFFFFF000))])) ^ r10_3
    r10_4 = r10_3 ^ ctxb(ctx.s16768, uint32(ct[:, int((r12_5 & 4095) + (rsi & 0xFFFFF000))]))
    st[672] = (st[672] & 0xFFFFFF00) | (r10_4 & 255)
    rdx_99 = rbx_12 & 255
    rdx_103 = rdx_99 | rsi_38
    rax_93 = rax_89 << 8
    rcx_91 = ctxb(ctx.s12000, st[1096])
    st[592] = rcx_91
    rcx_92 = rcx_91 ^ 77
    rdx_107 = rdx_103 << 8
    r14_6 = r10_4 & 255
    rdi_31 = r14_6 | rdx_107
    rcx_96 = rcx_92 | rax_93
    return NS(
        rax=((rcx_96 ^ 0x122014A5) ^ (rdi_31 ^ 0x20D6500F)),
        r11=r11_5,
        r12=r12_5,
        r13=rdx_77,
        r15=r15_7,
        r8=rdx_91,
        r14=r14_6,
        rbx=rbx_12,
        r10=r10_4,
        rcx=rcx_96,
    )


def round1Tail(ctx, st, eax, r13, r15, r8, r14):  # noqa: D103
    st[616] = (eax >> 8) & 255
    rax_5 = ctxb(ctx.s12000, st[760]) ^ 77
    rcx_9 = ctxb(ctx.s12000, st[1712]) ^ 77
    rdx_6 = r15 & 255
    rax_9 = rax_5 << 16
    rcx_13 = rcx_9 << 8
    rcx_17 = rcx_13 | rax_9
    rax_13 = r8 & 255
    rdx_10 = rdx_6 << 16
    rax_17 = rax_13 << 8
    rax_21 = rax_17 | rdx_10
    rdx_17 = r13 & 255
    rdx_21 = rdx_17 | rax_21
    rax_27 = ctxb(ctx.s12000, st[1128]) ^ 77
    rax_31 = rax_27 | rcx_17
    rax_35 = rax_31 << 8
    r8_1 = ctxb(ctx.s12000, st[1288])
    rbx_4 = r8_1 ^ 77
    rdx_25 = rdx_21 << 8
    rbx_8 = st[96] & 255
    rsi_11 = rbx_8 | rdx_25
    rdi_37 = rbx_4 | rax_35
    rax_44 = (rdi_37 ^ 0x4DBBDEDE) ^ (rsi_11 ^ 0x6B7E3FC1)
    rax_45 = st[88] & 255
    rsi_15 = ctxb(ctx.s12000, st[888]) ^ 77
    rdi_41 = ctxb(ctx.s12000, st[1736]) ^ 77
    rdx_38 = st[104] & 255
    rsi_19 = rsi_15 << 16
    rdi_45 = rdi_41 << 8
    rsi_26 = rdi_45 | rsi_19
    rsi_29 = ctxb(ctx.s12000, st[1304]) ^ 77
    rsi_33 = rsi_29 | rsi_26
    rax_49 = rax_45 << 16
    rdx_42 = rdx_38 << 8
    rcx_58 = rdx_42 | rax_49
    rcx_59 = st[112] & 255
    rdx_49 = rcx_59 | rcx_58
    rsi_37 = rsi_33 << 8
    rcx_60 = st[1680]
    r10_2 = ctxb(ctx.s12000, rcx_60)
    rax_66 = r10_2 ^ 77
    rax_70 = rax_66 | rsi_37
    rdx_53 = rdx_49 << 8
    r8_2 = st[336] & 255
    rsi_51 = r8_2 | rdx_53
    r12_5 = ((eax >> 16) & 255) ^ 246
    rax_73 = (rax_70 ^ 0x371DC5A0) ^ (rsi_51 ^ 0x53A29860)
    return NS(
        rdi=st[64],
        rsi=((rax_73 >> 8) & 255),
        rdx=rax_73,
        rcx=rcx_60,
        r8=r8_2,
        r9=rax_44,
        rax=((rax_73 >> 24) ^ 100),
        rbx=(t4(ctx.t14672, (((rbx_8 ^ r8_1) ^ 77) & 255)) ^ t4(ctx.t12272, (r12_5 & 255))),
        r10=r10_2,
        r11=eax,
        r12=r12_5,
        r13=rax_44,
        r14=r14,
        r15=((rax_44 >> 8) & 255),
    )


def round2Sub6400(ctx, st, a1, a2, a3, a4, a5, a6, v6, v7, v9, v10, v11, v12, v13, v14):  # noqa: D103, PLR0915
    cp12 = [0, 0, 0]
    v14 = t4(ctx.t19136, v6) ^ v7
    v15 = (
        (t4(ctx.t18080, (a2 ^ 93)) ^ t4(ctx.t12272, (((v11 >> 16) & 255) ^ 197))) ^ st[1568]
    ) ^ t4(ctx.t19136, (((st[536] >> 24) & 255) ^ 115))
    v16 = (
        (
            (t4(ctx.t18080, (v13 ^ 225)) ^ t4(ctx.t14672, (((v9 ^ a5) & 255) ^ 77)))
            ^ t4(ctx.t19136, (((v10 >> 24) & 255) ^ 50))
        )
        ^ st[1704]
    ) ^ t4(ctx.t12272, (((st[80] >> 16) & 255) ^ 209))
    st[656] = (v16 >> 8) & 255
    st[80] = v16
    v17 = (t4(ctx.t12272, (((a3 >> 16) & 255) ^ 191)) ^ st[1552]) ^ t4(
        ctx.t19136, (((a6 >> 24) & 255) ^ 38)
    )
    v18 = t4(ctx.t14672, (((st[592] ^ v12) & 255) ^ 77)) ^ v15
    st[536] = v18 & 255
    st[592] = (v18 >> 8) & 255
    v19 = ctxb(ctx.s12000, st[768]) ^ 77
    v20 = v19 << 16
    v21a = ctxb(ctx.s12000, st[1648]) ^ 77
    v21 = v21a << 8
    v22 = ctxb(ctx.s12000, st[1160]) ^ 77
    v23 = v20 | v21
    v24 = v23 | v22
    v25 = v24 << 8
    v26 = ctxb(ctx.s12000, st[1152]) ^ 77
    v27 = v25 | v26
    v28 = ((v27 ^ 0x3F9C9A) ^ t4(ctx.t18080, (st[600] ^ 37))) ^ v14
    v29 = ((((v27 ^ 0x8000001A) + 0x7FFFFF80) ^ (t4(ctx.t18080, (st[600] ^ 37)) & 255)) ^ v14) & 255
    v30 = (
        ((((v27 ^ 0x80001C9A) + 0x7FFF8000) ^ (t4(ctx.t18080, (st[600] ^ 37)) & 65535)) ^ v14) >> 8
    ) & 255
    v31 = (t4(ctx.t14672, (((st[72] ^ st[144]) & 255) ^ 77)) ^ t4(ctx.t18080, (st[616] ^ 68))) ^ v17
    st[72] = v31 & 255
    v32 = (
        (
            (t4(ctx.t18080, (v30 ^ 227)) ^ t4(ctx.t12272, (((v31 >> 16) & 255) ^ 223)))
            ^ t4(ctx.t14672, ((v16 & 255) ^ 14))
        )
        ^ st[1760]
    ) ^ t4(ctx.t19136, (((v18 >> 24) & 255) ^ 84))
    v33 = (
        (t4(ctx.t18080, (((v31 >> 8) & 255) ^ 89)) ^ t4(ctx.t19136, (((v16 >> 24) & 255) ^ 58)))
        ^ st[1560]
    ) ^ t4(ctx.t12272, (((v18 >> 16) & 255) ^ 85))
    v34 = v32
    st[144] = v32 & 255
    v35 = (v32 >> 8) & 255
    st[648] = v32
    v36 = t4(ctx.t14672, (v29 ^ 40)) ^ v33
    st[616] = v36 & 255
    v37 = (
        (t4(ctx.t18080, (st[656] ^ 143)) ^ t4(ctx.t12272, (((v28 >> 16) & 255) ^ 41))) ^ st[1720]
    ) ^ t4(ctx.t19136, (((v31 >> 24) & 255) ^ 23))
    v38 = ctxb(ctx.s12000, st[704]) ^ 77
    v39 = ctxb(ctx.s12000, st[1688]) ^ 77
    v40 = v38 << 16
    v41 = v39 << 8
    v42 = v40 | v41
    v43a = ctxb(ctx.s12000, st[1120]) ^ 77
    v43 = v42 | ((((0xFFFFFFFE * (v43a & 76)) + v43a) + 0x29C4B64C) ^ 0x29C4B64C)
    Q44 = (v43 << 8) ^ 0xB880D00
    v44 = (Q44 ^ 0xB880D00) - (2 * (Q44 & 0x4017F000))
    v45 = ctxb(ctx.s12000, st[1248]) ^ 77
    v46 = ((v44 + 0x4017F083) ^ 0x4017F083) | v45
    v47 = t4(ctx.t14672, (st[536] ^ 101)) ^ v37
    v48 = (
        (
            (
                t4(ctx.t12272, (((st[80] >> 16) & 255) ^ 188))
                ^ t4(ctx.t19136, (((v28 >> 24) & 255) ^ 107))
            )
            ^ t4(ctx.t18080, (st[592] ^ 203))
        )
        ^ t4(ctx.t14672, (st[72] ^ 245))
    ) ^ (v46 ^ 0x4B35704F)
    v49 = v48 & 255
    v50 = (v48 >> 8) & 255
    v51 = (v48 >> 16) & 255
    st[72] = (t4(ctx.t18080, (((v36 >> 8) & 255) ^ 132)) ^ st[1632]) ^ t4(
        ctx.t19136, (((v47 >> 24) & 255) ^ 91)
    )
    v52 = ctxb(ctx.s12000, st[880]) ^ 77
    v53 = ctxb(ctx.s12000, st[1752]) ^ 77
    v54 = v52 << 16
    v55 = v53 << 8
    v56 = v54 | v55
    v57a = ctxb(ctx.s12000, st[1136]) ^ 77
    v57 = v56 | v57a
    v58 = v57 << 8
    v59a = ctxb(ctx.s12000, st[1256]) ^ 77
    v59 = v58 | v59a
    v60 = v59 ^ 0xE982105
    v61 = ((((v47 & 255) ^ 86) - (2 * (v47 & 41))) + 0x4A3D1A29) ^ 0x8E581820
    v62 = (
        (t4(ctx.t12272, (((v36 >> 16) & 255) ^ 4)) ^ v60)
        ^ t4(ctx.t19136, (((v48 >> 24) & 255) ^ 32))
    ) ^ t4(ctx.t14672, (((v61 ^ 0x4210285) + 0x3024400) ^ 0xC34644A7))
    v63 = (v47 >> 8) & 255
    v64 = (t4(ctx.t19136, (((v34 >> 24) & 255) ^ 84)) ^ st[1416]) ^ t4(
        ctx.t12272, (((v47 >> 16) & 255) ^ 209)
    )
    v65 = t4(ctx.t18080, (v35 ^ 13)) ^ v62
    v66 = v65 & 255
    st[600] = (v65 >> 8) & 255
    v35 = (v35 & 0xFFFF00FF) | (((v65 >> 16) & 255) << 16)
    v67 = (
        (
            (t4(ctx.t14672, (v49 ^ 253)) ^ t4(ctx.t12272, (((st[648] >> 16) & 255) ^ 239)))
            ^ t4(ctx.t19136, (((v36 >> 24) & 255) ^ 60))
        )
        ^ st[1384]
    ) ^ t4(ctx.t18080, (v63 ^ 216))
    v68 = ctxb(ctx.s12000, st[1728]) ^ 77
    v69a = ctxb(ctx.s12000, st[688]) ^ 77
    v69 = v69a << 16
    v70 = v68 << 8
    v71 = v69 | v70
    v72 = v71 ^ 0xD752362
    v73 = ctxb(ctx.s12000, st[1104]) ^ 77
    v74 = (v72 ^ 0xD752362) | v73
    v75 = v74 << 8
    v76a = ctxb(ctx.s12000, st[1144]) ^ 77
    v76 = v75 | v76a
    v77 = (t4(ctx.t14672, (st[616] ^ 50)) ^ t4(ctx.t18080, (v50 ^ 15))) ^ v64
    v78 = (t4(ctx.t14672, (st[144] ^ 178)) ^ t4(ctx.t12272, (v51 ^ 35))) ^ st[72]
    v79 = (
        (
            (t4(ctx.t18080, (((v77 >> 8) & 255) ^ 147)) ^ (v76 ^ 0x767BE3DA))
            ^ t4(ctx.t14672, ((v78 & 255) ^ 80))
        )
        ^ t4(ctx.t19136, (((v65 >> 24) & 255) ^ 101))
    ) ^ t4(ctx.t12272, (((v67 >> 16) & 255) ^ 42))
    v80 = v77 & 255
    v81 = v77
    v82 = (
        (t4(ctx.t18080, (((v78 >> 8) & 255) ^ 139)) ^ t4(ctx.t12272, (((v77 >> 16) & 255) ^ 136)))
        ^ st[1512]
    ) ^ t4(ctx.t19136, (((v67 >> 24) & 255) ^ 73))
    v83 = (t4(ctx.t19136, (((v78 >> 24) & 255) ^ 110)) ^ st[1536]) ^ t4(
        ctx.t12272, (((v35 >> 16) & 255) ^ 142)
    )
    st[680] = v79
    st[536] = v79 & 255
    st[72] = (v79 >> 8) & 255
    st[80] = v79
    v84 = t4(ctx.t18080, (((v67 >> 8) & 255) ^ 237)) ^ v83
    v85 = t4(ctx.t14672, (v66 ^ 183)) ^ v82
    v86 = (v78 >> 16) & 255
    v87 = (
        (t4(ctx.t14672, ((v67 & 255) ^ 111)) ^ t4(ctx.t19136, (((v81 >> 24) & 255) ^ 83)))
        ^ st[1480]
    ) ^ t4(ctx.t18080, (st[600] ^ 94))
    v88 = (v85 >> 16) & 255
    st[720] = v85 & 255
    st[624] = (v85 >> 8) & 255
    st[144] = v85
    v89 = t4(ctx.t14672, (v80 ^ 13)) ^ v84
    v90 = t4(ctx.t12272, (v86 ^ 250)) ^ v87
    st[616] = v90 & 255
    st[648] = (v90 >> 8) & 255
    v91 = v89
    st[640] = v89 & 255
    st[656] = (v89 >> 8) & 255
    v92 = ctxb(ctx.s12000, st[1664]) ^ 77
    v93a = ctxb(ctx.s12000, st[848]) ^ 77
    v93 = v93a << 16
    v94 = v92 << 8
    v85lo = ctxb(ctx.s12000, st[976]) ^ 77
    v85lo = (v85lo ^ 107) - 107
    v95 = v93 | v94
    v96 = v95 ^ 0x71554D0C
    v97 = ctxb(ctx.s12000, st[832]) ^ 77
    v98 = (v96 ^ 0x71554D0C) | v97
    v99 = v98 << 8
    v100a = ctxb(ctx.s12000, st[744]) ^ 77
    v100 = v99 | v100a
    v101 = t4(ctx.t12272, (((v89 >> 16) & 255) ^ 117))
    v102 = v101 & 1
    v103 = (2 * v102) ^ 2
    v104 = 0x2C2A63E0 - (v103 + v102)
    X = ctxb(ctx.s12000, st[968])
    st[600] = ((((X >> 7) | (2 * (X ^ 77))) >> 1) | ((((X >> 7) | (2 * (X ^ 77))) & 1) << 7)) & 255
    st[592] = v85lo << 16
    v85 = (st[80] >> 24) & 255
    v105 = (st[680] >> 16) & 65535
    st[144] = ((st[144] >> 24) & 255) ^ 84
    st[80] = v85 ^ 29
    st[72] = st[72] ^ 156
    st[680] = ((v90 >> 16) & 255) ^ 19
    v106 = ((v91 >> 24) & 255) ^ 30
    v107 = (st[1592] ^ t4(ctx.t19136, (((v90 >> 24) & 255) ^ 44))) ^ 0x4CAEE99A
    v108 = (v101 & 0xFFFFFFFE) ^ 0xCFDCCC9D
    v109 = v107 & ((v103 + v102) - 1)
    v110 = (v103 + v102) + 0x2C2A63DE
    v110 = np.where((v109 != (2 * (v109 & 0x700AC381))), v104, v110)
    v111 = ((t4(ctx.t18080, (st[624] ^ 93)) ^ t4(ctx.t19136, v106)) ^ st[1544]) ^ t4(
        ctx.t12272, ((v105 & 255) ^ 109)
    )
    v112 = ((t4(ctx.t14672, (st[640] ^ 178)) ^ (v100 ^ 0x5E27D07A)) ^ t4(ctx.t19136, st[80])) ^ t4(
        ctx.t12272, (v88 ^ 141)
    )
    v113 = (
        (
            (((v110 + v107) + 0x3C672974) - (((2 * (v107 + v110)) - 0x5854C7BE) & 0xD1231AA6))
            ^ (v108 ^ 0xC11E0ED5)
        )
        ^ t4(ctx.t18080, st[72])
    ) ^ t4(ctx.t14672, (st[720] ^ 172))
    st[80] = (v113 >> 8) & 255
    v114 = v113
    v115 = t4(ctx.t18080, (st[648] ^ 176)) ^ v112
    v116 = t4(ctx.t14672, (st[616] ^ 25)) ^ v111
    v117 = (t4(ctx.t12272, st[680]) ^ st[1368]) ^ t4(ctx.t19136, st[144])
    v118 = v116 & 255
    st[72] = v116
    st[648] = (v116 >> 8) & 255
    v119 = (t4(ctx.t18080, (st[656] ^ 123)) ^ t4(ctx.t14672, (st[536] ^ 104))) ^ v117
    st[656] = (v115 >> 8) & 255
    v120 = v115
    v121 = (
        (t4(ctx.t14672, ((v113 & 255) ^ 206)) ^ t4(ctx.t12272, (((v115 >> 16) & 255) ^ 49)))
        ^ t4(ctx.t18080, (((v119 >> 8) & 255) ^ 162))
    ) ^ (st[1432] ^ t4(ctx.t19136, (((v116 >> 24) & 255) ^ 91)))
    v122 = (t4(ctx.t19136, (((v119 >> 24) & 255) ^ 15)) ^ st[1392]) ^ t4(
        ctx.t12272, (((v113 >> 16) & 255) ^ 14)
    )
    st[144] = v121
    v123 = v121 & 255
    v124 = (v121 >> 8) & 255
    st[616] = v121
    v125 = (t4(ctx.t14672, ((v115 & 255) ^ 200)) ^ t4(ctx.t18080, (st[648] ^ 147))) ^ v122
    st[536] = v125 & 255
    v126 = (v125 >> 16) & 255
    st[680] = (v125 >> 8) & 255
    v127 = v125
    v121 = ctxb(ctx.s12000, st[1640]) ^ 77
    v128 = (st[592] + 0x1F6B0000) ^ 0x1F6B0000
    v121 = (((v121 << 8) ^ 0xFFFFC400) - 0x78930000) ^ 0x876CC400
    v129 = ctxb(ctx.s12000, st[784]) ^ 77
    v121 = (((v128 ^ 0xCC30000) + 41497) ^ 0xCC3A219) | (((v121 ^ 0x773AB600) + 137) ^ 0x773AB689)
    v130 = v121 | v129
    v131 = v130 << 8
    v132a = ctxb(ctx.s12000, st[1048]) ^ 77
    v132 = v131 | v132a
    v133 = (
        (
            ((v132 ^ 0x73A6AD9D) ^ t4(ctx.t12272, (((v119 >> 16) & 255) ^ 68)))
            ^ t4(ctx.t19136, (((v120 >> 24) & 255) ^ 53))
        )
        ^ t4(ctx.t14672, (v118 ^ 53))
    ) ^ t4(ctx.t18080, (st[80] ^ 195))
    st[640] = (v133 >> 8) & 255
    v134 = (
        (
            (t4(ctx.t19136, (((v114 >> 24) & 255) ^ 107)) ^ t4(ctx.t18080, (st[656] ^ 175)))
            ^ t4(ctx.t12272, (((st[72] >> 16) & 255) ^ 23))
        )
        ^ st[1624]
    ) ^ t4(ctx.t14672, ((v119 & 255) ^ 157))
    v135 = (t4(ctx.t14672, ((v133 & 255) ^ 47)) ^ t4(ctx.t19136, (((v127 >> 24) & 255) ^ 58))) ^ t4(
        ctx.t18080, (v124 ^ 22)
    )
    st[656] = v135
    v121 = t4(ctx.t12272, (((v134 >> 16) & 255) ^ 223))
    st[592] = v121
    v136 = (v121 ^ st[1528]) ^ v135
    st[72] = v136
    st[648] = v136 & 255
    st[728] = (v136 >> 8) & 255
    st[720] = v136
    v121 = ctxb(ctx.s12000, st[856]) ^ 77
    v137 = ctxb(ctx.s12000, st[1656]) ^ 77
    v121 = (((v121 << 16) ^ 0x80B20000) + 0x52000000) ^ 0xD2B20000
    v138 = v137 << 8
    v139 = (((v121 ^ 0x18F10000) + 23783) ^ 0x18F15CE7) | v138
    v121 = v139 ^ 0x3DB103D
    v140 = ctxb(ctx.s12000, st[864]) ^ 77
    v141 = (v121 ^ 0x3DB103D) | v140
    v142 = v141 << 8
    v121 = v142 ^ 0x57F73C23
    v143 = ctxb(ctx.s12000, st[808]) ^ 77
    v144 = (v121 ^ 0x57F73C23) | v143
    v145 = (
        (t4(ctx.t12272, (v126 ^ 118)) ^ t4(ctx.t18080, (((v134 >> 8) & 255) ^ 109)))
        ^ t4(ctx.t19136, (((v133 >> 24) & 255) ^ 24))
    ) ^ t4(ctx.t14672, (v123 ^ 108))
    st[752] = v145
    v121 = v145 ^ (v144 ^ 0x6F83A9D4)
    st[816] = v121
    st[624] = (v145 ^ ((v144 ^ 0x80000054) + 0x7FFFFF80)) & 255
    st[792] = ((v145 ^ ((v144 ^ 0x800029D4) + 0x7FFF8000)) >> 8) & 255
    st[80] = v121
    v146 = t4(ctx.t18080, (st[680] ^ 155))
    v147 = t4(ctx.t18080, (st[640] ^ 210))
    v121 = ((((st[600] & 255) - (2 * (st[600] & 249))) << 16) + 0x70F90000) ^ 0x70F90000
    v148a = ctxb(ctx.s12000, st[1768]) ^ 77
    v148 = v148a << 8
    v149 = (((v121 ^ 0x3990000) + 9254) ^ 0x3992426) | v148
    v121 = v149 ^ 0x6F03DB72
    v150 = ctxb(ctx.s12000, st[800]) ^ 77
    v151 = (v121 ^ 0x6F03DB72) | v150
    v152 = ((ctxb(ctx.s12000, st[920]) ^ 77) ^ 94) - 94
    v153 = ctxb(ctx.s12000, st[1056]) ^ 77
    v154 = (v153 ^ 136) - 136
    v155 = t4(ctx.t19136, (((v134 >> 24) & 255) ^ 24)) ^ t4(
        ctx.t12272, (((st[144] >> 16) & 255) ^ 191)
    )
    v156 = (
        t4(ctx.t12272, (((v133 >> 16) & 255) ^ 176)) ^ t4(ctx.t19136, (((st[616] >> 24) & 255) ^ 7))
    ) ^ t4(ctx.t14672, ((v134 & 255) ^ 128))
    v157 = (v147 & 0xFBFFFFFF) ^ 0x436721C0
    v158 = (v146 ^ st[1360]) ^ v156
    st[144] = v158
    v159 = (v146 ^ st[1456]) ^ v156
    v121 = (v147 ^ st[1440]) ^ v155
    v160 = v147 & 0x4000000
    v161 = st[1408] ^ v155
    v162 = (((v160 ^ 0xDF7FF7FF) + (2 * v160)) + 0x20800801) & (v161 ^ 0x40000000)
    st[600] = v152 << 16
    v163 = st[536] ^ 254
    st[872] = v154 << 16
    v164 = v151 << 8
    v165 = t4(ctx.t14672, v163)
    st[616] = v165
    v166 = (st[720] >> 16) & 65535
    st[72] = ((st[72] >> 24) & 255) ^ 17
    v167 = v165 ^ v121
    st[824] = ((v158 >> 24) & 255) ^ 30
    v168 = v157 ^ 0x12A7AD0E
    st[80] = st[80] >> 16
    st[944] = (st[144] >> 16) & 65535
    v169 = v162 == (2 * (v162 & 0x24000000))
    v170 = ((v160 ^ 0xDF7FF7FF) + (2 * v160)) + 0x27C9EFA8
    v170 = np.where(np.logical_not(v169), (0xE6C9DFA6 - ((v160 ^ 0xDF7FF7FF) + (2 * v160))), v170)
    st[640] = v159
    st[536] = v159 & 255
    st[912] = (v159 >> 8) & 255
    v171 = (
        (
            (((v161 ^ 0x439AFBDB) + v170) + 0x42028BA7)
            - (((2 * ((v161 ^ 0x439AFBDB) + v170)) - 0xE93CF4E) & 0x9298E69C)
        )
        ^ st[616]
    ) ^ (v168 ^ 0xAE437DE)
    st[720] = (
        (
            ((((v161 ^ 219) + v170) - 89) - (((2 * ((v161 ^ 219) + v170)) - 78) & 156))
            ^ (st[616] & 255)
        )
        ^ ((v168 ^ 0x8400005E) + 0x7BFFFF80)
    ) & 255
    st[664] = v167
    st[616] = (v167 >> 8) & 255
    v172 = st[144]
    v173 = (
        (
            t4(ctx.t18080, (((v171 >> 8) & 255) ^ 138))
            ^ t4(ctx.t19136, (((st[816] >> 24) & 255) ^ 4))
        )
        ^ t4(ctx.t12272, ((v166 & 255) ^ 245))
    ) ^ t4(ctx.t14672, ((v172 & 255) ^ 58))
    st[816] = (v172 >> 8) & 255
    v174 = ctxb(ctx.s12000, st[904]) ^ 77
    v175 = v164 | v174
    v176 = st[752] ^ (v175 ^ 0x520EDFF7)
    st[752] = v176
    st[960] = v176 & 255
    st[680] = v176
    st[952] = (v176 >> 8) & 255
    st[144] = (t4(ctx.t18080, (st[728] ^ 23)) ^ st[1576]) ^ t4(ctx.t19136, st[824])
    cp12 = [st[72], st[592], st[656]]
    v177 = ctxb(ctx.s12000, st[1784]) ^ 77
    v178 = (st[872] + 0x3880000) ^ 0x3880000
    v179 = v177 << 8
    v180 = (((v178 ^ 0x2F6F0000) + 5653) ^ 0x2F6F1615) | v179
    v181 = ctxb(ctx.s12000, st[1168]) ^ 77
    v182 = v180 | v181
    v183 = v182 << 8
    v184 = v183 ^ 0x4146DE9D
    v185 = ctxb(ctx.s12000, st[1064]) ^ 77
    v186 = (v184 ^ 0x4146DE9D) | v185
    v187 = (v186 ^ 0x176239CE) ^ v173
    v188 = (
        (t4(ctx.t19136, (((v171 >> 24) & 255) ^ 48)) ^ t4(ctx.t12272, ((st[944] & 255) ^ 221)))
        ^ st[1328]
    ) ^ t4(ctx.t18080, (st[792] ^ 214))
    st[592] = v187
    st[72] = (v187 >> 8) & 255
    st[656] = v187
    v189 = t4(ctx.t14672, (st[648] ^ 220)) ^ v188
    st[648] = v189 & 255
    st[728] = (v189 >> 8) & 255
    return NS(cp12=cp12, v171=v171, v187=v187, v189=v189)


def round3Sub8000(ctx, st, a2, a3, a4, a5, a6, v6, v8, v9, v10, v11, v12, out16):  # noqa: D103, PLR0915
    v13 = t4(ctx.t14672, (a2 ^ 102)) ^ v9
    v14 = t4(ctx.t14672, (st[960] ^ 69)) ^ v6
    st[624] = v14
    st[960] = v14 & 255
    v15 = v14
    st[912] = (v14 >> 8) & 255
    v16 = t4(ctx.t18080, (st[816] ^ 193)) ^ v13
    v17 = (
        (
            (t4(ctx.t18080, (st[952] ^ 160)) ^ t4(ctx.t12272, (((a6 >> 16) & 255) ^ 57)))
            ^ t4(ctx.t19136, (((st[664] >> 24) & 255) ^ 94))
        )
        ^ st[1448]
    ) ^ t4(ctx.t14672, (a4 ^ 147))
    v18 = (t4(ctx.t12272, ((st[80] & 255) ^ 149)) ^ t4(ctx.t14672, (st[720] ^ 223))) ^ st[144]
    st[144] = v18
    v19 = (
        (t4(ctx.t18080, (st[72] ^ 70)) ^ t4(ctx.t12272, (((v18 >> 16) & 255) ^ 117))) ^ st[216]
    ) ^ t4(ctx.t19136, (((a3 >> 24) & 255) ^ 84))
    v20 = (
        (
            (t4(ctx.t18080, (((st[144] >> 8) & 255) ^ 173)) ^ t4(ctx.t14672, (v10 ^ 124)))
            ^ t4(ctx.t19136, (((v16 >> 24) & 255) ^ 21))
        )
        ^ st[200]
    ) ^ t4(ctx.t12272, (((v12 >> 16) & 255) ^ 196))
    v21 = (
        t4(ctx.t12272, (((st[752] >> 16) & 255) ^ 24))
        ^ t4(ctx.t19136, (((st[640] >> 24) & 255) ^ 47))
    ) ^ v11
    st[80] = v21
    st[640] = v21 & 255
    v22 = (v21 >> 8) & 255
    v23 = v21
    st[664] = v16
    v24 = v16 & 255
    st[720] = (v16 >> 8) & 255
    v25 = ctxb(ctx.s12000, st[1000]) ^ 77
    v26 = ctxb(ctx.s12000, st[1776]) ^ 77
    v27 = v25 << 16
    v28 = v26 << 8
    v29 = v27 | v28
    v30e = ctxb(ctx.s12000, st[1264]) ^ 77
    v30 = v29 | ((((0xFFFFFFFE * (v30e & 99)) + v30e) + 0x11604A63) ^ 0x11604A63)
    v31 = v30 << 8
    v32 = ctxb(ctx.s12000, st[1272]) ^ 77
    v33 = v31 | v32
    v34 = (
        (
            ((v33 ^ 0x60E5E5A6) ^ t4(ctx.t12272, (v8 ^ 132)))
            ^ t4(ctx.t19136, (((st[680] >> 24) & 255) ^ 57))
        )
        ^ t4(ctx.t14672, (st[536] ^ 79))
    ) ^ t4(ctx.t18080, (st[616] ^ 164))
    inner = (
        (
            ((((v33 ^ 0x80000026) + 0x7FFFFF80) & 255) ^ (t4(ctx.t12272, (v8 ^ 132)) & 255))
            ^ (t4(ctx.t19136, (((st[680] >> 24) & 255) ^ 57)) & 255)
        )
        ^ (t4(ctx.t14672, (st[536] ^ 79)) & 255)
    ) ^ (t4(ctx.t18080, (st[616] ^ 164)) & 255)
    st[72] = (
        (
            (t4(ctx.t18080, ((v22 & 255) ^ 181)) ^ t4(ctx.t12272, (((v17 >> 16) & 255) ^ 215)))
            ^ t4(ctx.t14672, (inner ^ 20))
        )
        ^ st[200]
    ) ^ t4(ctx.t19136, (((v15 >> 24) & 255) ^ 93))
    v35 = st[1424]
    v36 = st[144] & 255
    v37 = st[144]
    st[144] = (
        (t4(ctx.t18080, (st[912] ^ 39)) ^ t4(ctx.t19136, (((v23 >> 24) & 255) ^ 13))) ^ v35
    ) ^ t4(ctx.t12272, (((v34 >> 16) & 255) ^ 243))
    v38 = t4(ctx.t18080, (((v34 >> 8) & 255) ^ 154)) ^ t4(ctx.t14672, (st[960] ^ 200))
    v39 = (t4(ctx.t19136, (((v37 >> 24) & 255) ^ 93)) ^ v35) ^ t4(
        ctx.t12272, (((st[656] >> 16) & 255) ^ 116)
    )
    v40 = v17 & 255
    v41 = (v17 >> 8) & 255
    v42 = v17
    v43 = (st[600] - 0x26A20000) ^ 0xD95E0000
    v44_ = ctxb(ctx.s12000, st[1800]) ^ 77
    v44 = v44_ << 8
    v45 = (((v43 ^ 0x134D0000) + 35148) ^ 0x134D894C) | v44
    v46 = ctxb(ctx.s12000, st[1024]) ^ 77
    v47 = v45 | v46
    v48 = v47 << 8
    v49 = v48 ^ 0x6750E07D
    v50 = ctxb(ctx.s12000, st[896]) ^ 77
    v51 = (v49 ^ 0x6750E07D) | v50
    v52 = v51 ^ 0x66DF5819
    v53 = ((t4(ctx.t18080, (st[728] ^ 124)) ^ v52) ^ t4(ctx.t14672, (v36 ^ 246))) ^ t4(
        ctx.t19136, (((st[592] >> 24) & 255) ^ 124)
    )
    v54 = (
        (t4(ctx.t18080, (v41 ^ 251)) ^ t4(ctx.t12272, (((st[624] >> 16) & 255) ^ 239)))
        ^ t4(ctx.t19136, (((v34 >> 24) & 255) ^ 11))
    ) ^ v52
    v55 = (st[216] ^ t4(ctx.t19136, (((v42 >> 24) & 255) ^ 15))) ^ 0x46AEA1
    v56 = t4(ctx.t12272, (((st[80] >> 16) & 255) ^ 112)) ^ 0x2F5CCE55
    v57 = ((v56 & v55) ^ 0x129E25A6) - 0x129E25A6
    v58 = t4(ctx.t14672, (v40 ^ 56)) ^ st[144]
    P59 = (((2 * v57) + 0x253C4B4C) ^ 0xDAC3B4B2) + 2
    v59 = ((((P59 ^ 0xFDFBDE6F) + v55) + v56) + 0x3F823735) + (2 * (P59 & 0xFDFBDE6E))
    v60 = t4(ctx.t14672, (v24 ^ 191)) ^ v19
    v61 = (t4(ctx.t14672, (st[648] ^ 158)) ^ t4(ctx.t18080, (st[720] ^ 151))) ^ v39
    v62 = ((v59 - (((2 * v59) + 0x503D4B8) & 0x2492A88C)) - 0x2B34C15E) ^ v38
    v63 = t4(ctx.t12272, (((st[664] >> 16) & 255) ^ 163)) ^ v53
    v64 = t4(ctx.t14672, (st[640] ^ 40)) ^ v54
    v65 = (st[72] ^ 0x4AB8AF80) + ((st[72] ^ v20) ^ 0x31D443C4)
    v66 = (((v65 ^ 0xFDB576FF) + (v20 ^ 0xB547507F)) + (2 * (v65 & 0xFDB576FF))) - 0x1E3DA80F
    v67 = v62
    A68 = (v64 ^ 0xDC927AB) + ((v64 ^ v63) ^ 0x482D8CA3)
    v68 = (((A68 ^ 0x75B3FF42) + (v63 ^ 0xF236D854)) + (2 * (A68 & 0xF5B3FF42))) - 0x75B3FF41
    B69 = ((v61 ^ v58) ^ 0x686D45AE) + (v58 ^ 0x38F593ED)
    v69 = (
        ((((B69 ^ 0xF7D6FDCF) + (v61 ^ 0xC70A6C12)) + (2 * (B69 & 0xF7D6FDCF))) + 0x8290232)
        ^ 0x5EFDFEBA
    ) + (
        2
        * (
            ((((B69 ^ 0xF7D6FDCF) + (v61 ^ 0xC70A6C12)) + (2 * (B69 & 0xF7D6FDCF))) + 0x8290232)
            & 0xDEFDFEBA
        )
    )
    v70 = ((0xAF45A39F * v66) - (((0x5E8B473E * v66) + 0x8D2991E) & 0x54603596)) - 0x116698A6
    C71 = ((v62 ^ v60) ^ 0xC81D1646) + (v62 ^ 0x304816C7)
    v71 = 0xDDDB4679 * (
        (((C71 ^ 0x1FDE8EEB) + (v60 ^ 0xF2E4DD8A)) + (2 * (C71 & 0x1FDE8EEB))) - 0xD3EAF70
    )
    v72 = (v71 + (v70 ^ 0xD5CFE534)) - 0x240C5DBA
    v73 = ((0xA371574D * v69) - (((0x46E2AE9A * v69) + 0x7BDE581C) & 0xDE38E6A0)) - 0x52F460A2
    E74 = v68 + 0x3AAE5BDF
    v74 = v73 ^ (
        ((0xC6DA4F7B * E74) - (((0x8DB49EF6 * E74) + 0x13F415B6) & 0x3498EBE4)) + 0x644680CD
    )
    v75 = v74 ^ v70
    Q76 = ((v74 ^ 0x755006A2) - v72) - 0x20026111
    v76 = (Q76 ^ 0x7F97FBB7) + (2 * (Q76 & 0xFF97FBB7))
    R77 = v73 ^ 0x6F1C7350
    v77 = (
        v75 ^ ((R77 + 0x7FEE5ACB) - (v71 + (((2 * (R77 - v71)) + 0x813F954) & 0xF7C8BC42)))
    ) ^ 0x24844248
    S78 = v77 + (v75 ^ 0x5F601C69)
    v78 = (S78 ^ 0x5EAF75E7) + (2 * (S78 & 0xDEAF75E7))
    T79 = v76 - v78
    v79 = (T79 + 0x59ECC8E0) - (((2 * T79) - 0x41D10BA0) & 0xF5AA9D60)
    v80 = v64 ^ v79
    U = v76 - v78
    st[144] = (v64 ^ ((U - 32) - (((2 * U) + 96) & 96))) & 255
    st[536] = ((v64 ^ ((U - 14112) - (((2 * U) - 2976) & 40288))) >> 8) & 255
    st[80] = v80
    V81 = v77 + (v79 ^ 0x852AB14F)
    v81 = (V81 + 0x42B5E6A3) - (((2 * V81) + 2) & 0x856BCD44)
    W82 = v77 + (v79 ^ 79)
    v82 = (v58 ^ ((W82 - 93) - (((2 * W82) + 2) & 68))) & 255
    st[592] = ((v58 ^ v81) >> 8) & 255
    v83 = v58 ^ v81
    X84 = ((v72 - v76) + (v81 ^ 0x42B5E6A2)) - 0x6065A338
    v84 = ((v72 + (v81 ^ 0x42B5E6A2)) - 0x445B9DE) - (v76 + (2 * (X84 & 0x5C1FE95A)))
    v85 = st[72] ^ (
        (
            (v78 + (v84 ^ 0xA3E016A5))
            - (((2 * (v78 + (v84 ^ 0xA3E016A5))) + 0x42A11434) & 0xC3F87424)
        )
        + 0x34CC42C
    )
    st[72] = v85 & 255
    st[600] = (v85 >> 8) & 255
    v86 = v84 ^ v67
    v87 = (v84 ^ v67) & 255
    v88 = t4(ctx.t20208, ((((v84 ^ v67) >> 8) & 255) ^ 116)) ^ t4(
        ctx.t13328, (((v80 >> 24) & 255) ^ 52)
    )
    v89 = ctxb(ctx.s12000, st[1016]) ^ 77
    v90 = v89 << 16
    v91 = v90 ^ 0x32B761C1
    v92_ = ctxb(ctx.s12000, st[1888]) ^ 77
    v92 = v92_ << 8
    v93 = ctxb(ctx.s12000, st[1496]) ^ 77
    v94 = (v91 ^ 0x32B761C1) | v92
    v95 = v94 | v93
    v96 = v95 << 8
    v97 = ctxb(ctx.s12000, st[1352]) ^ 77
    v98 = v96 | v97
    v82_orig = v82
    _ = v82_orig
    t_v99_1 = t4(ctx.t15728, (((v85 >> 16) & 255) ^ 253))
    t_v99_2 = t4(ctx.t17040, (v82 ^ 241))
    t_v99_3 = v98 ^ 0x66D9AF56
    v99 = ((t_v99_1 ^ t_v99_2) ^ t_v99_3) ^ v88
    v82 = (st[400] ^ ((v99 >> 16) & 255)) & 255
    v100 = ctxb(ctx.s12000, st[1072]) ^ 77
    v101 = ctxb(ctx.s12000, st[1792]) ^ 77
    v102 = v100 << 16
    v103 = v101 << 8
    v104 = v102 | v103
    v105a = ctxb(ctx.s12000, st[1400]) ^ 77
    v105 = v104 | ((((0xFFFFFFFE * (v105a & 125)) + v105a) + 0x5E77297D) ^ 0x5E77297D)
    v106 = v105 << 8
    v107 = v106 ^ 0x416E1575
    v108 = ctxb(ctx.s12000, st[1296]) ^ 77
    v109 = (v107 ^ 0x416E1575) | v108
    v110 = (
        (
            (t4(ctx.t20208, (st[600] ^ 26)) ^ (v109 ^ 0x14D1BB74))
            ^ t4(ctx.t15728, (((st[80] >> 16) & 255) ^ 160))
        )
        ^ t4(ctx.t17040, (v87 ^ 122))
    ) ^ t4(ctx.t13328, (((v83 >> 24) & 255) ^ 228))
    v111 = (
        (t4(ctx.t17040, (st[144] ^ 224)) ^ t4(ctx.t15728, (((v86 >> 16) & 255) ^ 58)))
        ^ t4(ctx.t13328, (((v85 >> 24) & 255) ^ 131))
    ) ^ t4(ctx.t20208, (st[592] ^ 235))
    v112 = (
        (t4(ctx.t17040, (st[72] ^ 31)) ^ t4(ctx.t20208, (st[536] ^ 56)))
        ^ t4(ctx.t15728, (((v83 >> 16) & 255) ^ 167))
    ) ^ t4(ctx.t13328, (((v86 >> 24) & 255) ^ 117))
    pt0 = ctxb(ctx.s21248, (((st[376] & 255) ^ ((v99 >> 24) & 255)) ^ 14)) ^ 118
    v113 = ctxb(ctx.s12000, st[1240])
    st[144] = ctxb(ctx.s12000, st[1472])
    v114 = ctxb(ctx.s12000, st[1600])
    v115 = ctxb(ctx.s12000, st[1904])
    v116 = ctxb(ctx.s12000, st[1920])
    st[72] = ctxb(ctx.s12000, st[1376])
    v117 = ctxb(ctx.s12000, st[1856])
    v118 = ctxb(ctx.s12000, st[840])
    v119 = st[1896]
    _ = v119
    v120 = (((((v118 ^ 77) & 255) - (2 * ((v118 ^ 77) & 96))) << 16) - 0x20A00000) ^ 0xDF600000
    v121 = (((((v115 ^ 77) & 255) - (2 * ((v115 ^ 77) & 109))) << 8) - 0x7F949300) ^ 0x806B6D00
    v122 = v120 | v121
    v123 = v122 | (((((v116 ^ 77) & 255) - (2 * ((v116 ^ 77) & 11))) + 0x1D66910B) ^ 0x1D66910B)
    v124 = v123 << 8
    v125 = v124 | (((((v117 ^ 77) & 255) - (2 * ((v117 ^ 77) & 62))) + 0x6BD28F3E) ^ 0x6BD28F3E)
    v126 = (v125 ^ 0x50E547DF) ^ v112
    v127 = (((((v113 ^ 77) & 255) - (2 * ((v113 ^ 77) & 156))) << 16) - 0x33640000) ^ 0xCC9C0000
    v128 = v127 ^ 0x7B9D1559
    v129 = (v114 ^ 77) << 8
    v132 = (v128 ^ 0x7B9D1559) | v129
    st48x = st[72] ^ 77
    v133 = v132 | (((st48x ^ 175) + 0x2E7F0B00) ^ 0x2E7F0BAF)
    v134 = v133 << 8
    st90x = st[144] ^ 77
    v135 = v134 | (((st90x ^ 74) + 0x58DE3E00) ^ 0x58DE3E4A)
    v136 = (v135 ^ 0x59F80B3) ^ v111
    o0, b0 = st[120], uint8(pt0 & 255)
    o1, b1 = st[560], uint8((ctxb(ctx.s21248, ((v82 & 255) ^ 147)) ^ 118) & 255)
    o2, b2 = (
        st[152],
        uint8((ctxb(ctx.s21248, (((st[280] & 255) ^ ((v99 >> 8) & 255)) ^ 35)) ^ 118) & 255),
    )
    o3, b3 = st[528], uint8((ctxb(ctx.s21248, (((st[992] & 255) ^ (v99 & 255)) ^ 44)) ^ 118) & 255)
    o4, b4 = (
        st[408],
        uint8((ctxb(ctx.s21248, (((st[352] & 255) ^ ((v110 >> 24) & 255)) ^ 124)) ^ 118) & 255),
    )
    o5, b5 = (
        st[176],
        uint8((ctxb(ctx.s21248, (((st[392] & 255) ^ ((v110 >> 16) & 255)) ^ 155)) ^ 118) & 255),
    )
    o6, b6 = (
        st[432],
        uint8((ctxb(ctx.s21248, (((st[224] & 255) ^ ((v110 >> 8) & 255)) ^ 55)) ^ 118) & 255),
    )
    o7, b7 = st[608], uint8((ctxb(ctx.s21248, (((st[288] & 255) ^ (v110 & 255)) ^ 14)) ^ 118) & 255)
    o8, b8 = (
        st[544],
        uint8((ctxb(ctx.s21248, (((st[256] & 255) ^ ((v126 >> 24) & 255)) ^ 56)) ^ 118) & 255),
    )
    o9, b9 = (
        st[344],
        uint8((ctxb(ctx.s21248, (((st[208] & 255) ^ ((v126 >> 16) & 255)) ^ 175)) ^ 118) & 255),
    )
    o10, b10 = (
        st[424],
        uint8((ctxb(ctx.s21248, (((st[248] & 255) ^ ((v126 >> 8) & 255)) ^ 203)) ^ 118) & 255),
    )
    o11, b11 = (
        st[416],
        uint8((ctxb(ctx.s21248, (((st[264] & 255) ^ (v126 & 255)) ^ 165)) ^ 118) & 255),
    )
    o12, b12 = (
        st[448],
        uint8((ctxb(ctx.s21248, (((st[240] & 255) ^ ((v136 >> 24) & 255)) ^ 109)) ^ 118) & 255),
    )
    o13, b13 = (
        st[440],
        uint8((ctxb(ctx.s21248, (((st[232] & 255) ^ ((v136 >> 16) & 255)) ^ 213)) ^ 118) & 255),
    )
    o14, b14 = (
        st[696],
        uint8((ctxb(ctx.s21248, (((st[1040] & 255) ^ ((v136 >> 8) & 255)) ^ 12)) ^ 118) & 255),
    )
    o15, b15 = (
        st[736],
        uint8((ctxb(ctx.s21248, (((st[192] & 255) ^ (v136 & 255)) ^ 201)) ^ 118) & 255),
    )
    base = min(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)
    d = o0 - base
    if d < 16:
        out16[int(d)] = b0
    d = o1 - base
    if d < 16:
        out16[int(d)] = b1
    d = o2 - base
    if d < 16:
        out16[int(d)] = b2
    d = o3 - base
    if d < 16:
        out16[int(d)] = b3
    d = o4 - base
    if d < 16:
        out16[int(d)] = b4
    d = o5 - base
    if d < 16:
        out16[int(d)] = b5
    d = o6 - base
    if d < 16:
        out16[int(d)] = b6
    d = o7 - base
    if d < 16:
        out16[int(d)] = b7
    d = o8 - base
    if d < 16:
        out16[int(d)] = b8
    d = o9 - base
    if d < 16:
        out16[int(d)] = b9
    d = o10 - base
    if d < 16:
        out16[int(d)] = b10
    d = o11 - base
    if d < 16:
        out16[int(d)] = b11
    d = o12 - base
    if d < 16:
        out16[int(d)] = b12
    d = o13 - base
    if d < 16:
        out16[int(d)] = b13
    d = o14 - base
    if d < 16:
        out16[int(d)] = b14
    d = o15 - base
    if d < 16:
        out16[int(d)] = b15
    st[288] = (st[288] & 0xFFFFFF00) | (st[96] & 255)
    st[208] = (st[208] & 0xFFFFFF00) | (st[360] & 255)
    st[256] = (st[256] & 0xFFFFFF00) | (st[304] & 255)
    st[352] = (st[352] & 0xFFFFFF00) | (st[568] & 255)
    st[400] = (st[400] & 0xFFFFFF00) | (st[104] & 255)
    st[280] = (st[280] & 0xFFFFFF00) | (st[112] & 255)
    st[248] = (st[248] & 0xFFFFFF00) | (st[368] & 255)
    st[376] = (st[376] & 0xFFFFFF00) | (st[88] & 255)
    st[224] = (st[224] & 0xFFFFFF00) | (st[552] & 255)
    st[192] = (st[192] & 0xFFFFFF00) | (st[672] & 255)
    st[232] = (st[232] & 0xFFFFFF00) | (st[584] & 255)
    st[240] = (st[240] & 0xFFFFFF00) | (st[1088] & 255)
    st[392] = (st[392] & 0xFFFFFF00) | (st[512] & 255)
    st[1040] = (st[1040] & 0xFFFFFF00) | (st[576] & 255)
    st[992] = (st[992] & 0xFFFFFF00) | (st[336] & 255)


def processBlock(tmpl, st, ct, bi, out16):  # noqa: D103
    b = uint32(bi)
    ctx = tmpl.ctx
    mid = round1Mid(
        ctx, st, ct, (0x1EB2C6B4 ^ (b << 4)), (0x8 + (b << 4)), tmpl.r1Entry.rcx, tmpl.r1Entry.rax
    )
    r2 = round1Tail(
        ctx, st, mid.rax, (mid.r13 & 0xFF), (mid.r15 & 0xFF), (mid.r8 & 0xFF), (mid.r14 & 0xFF)
    )
    r2v = round2Sub6400(
        ctx,
        st,
        r2.rdi,
        r2.rsi,
        r2.rdx,
        r2.rcx,
        r2.r8,
        r2.r9,
        r2.rax,
        r2.rbx,
        r2.r10,
        r2.r11,
        r2.r13,
        r2.r14,
        r2.r15,
        0,
    )
    cp12 = r2v.cp12
    r8p = (cp12[2] ^ st[0x5B8]) ^ cp12[1]
    v6 = (
        (t4(ctx.t18080, (st[0x390] ^ 0x2B)) ^ st[0x5F0])
        ^ t4(ctx.t19136, (((r8p >> 24) & 0xFF) ^ 0x29))
    ) ^ t4(ctx.t12272, (((st[0x298] >> 16) & 0xFF) ^ 0xD6))
    v9 = (t4(ctx.t19136, cp12[0]) ^ st[0x540]) ^ t4(ctx.t12272, (((r2v.v171 >> 16) & 0xFF) ^ 0x69))
    v11 = (t4(ctx.t14672, ((st[0x298] & 0xFF) ^ 0x57)) ^ st[0x538]) ^ t4(
        ctx.t18080, (((r8p >> 8) & 0xFF) ^ 0x2F)
    )
    a2 = st[0x270]
    a6 = st[0x280]
    round3Sub8000(
        ctx,
        st,
        a2,
        r2v.v189,
        (r8p & 0xFF),
        ((r8p >> 16) & 0xFFFF),
        a6,
        v6,
        ((r8p >> 16) & 0xFF),
        v9,
        (r2v.v187 & 0xFF),
        v11,
        r2v.v189,
        out16,
    )
    st[0x108] = st[0x180]
    st[0x220] += 0x10
