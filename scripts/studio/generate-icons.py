#!/usr/bin/env python3
"""Generate SciRust Studio's application icons.

The icons are generated rather than checked in as opaque binaries so they
can be reviewed, reproduced and changed by editing geometry instead of by
trusting a blob. Run it from anywhere:

    python3 scripts/studio/generate-icons.py

It writes into ``apps/scirust-studio/src-tauri/icons``:

    icon.ico          16, 32, 48, 64, 128, 256 — required by tauri-build on
                      Windows, which embeds it as the executable's resource
    icon.png          512, the source-of-truth raster
    32x32.png         Linux
    128x128.png       Linux
    128x128@2x.png    Linux HiDPI (256)

``icon.ico`` is the one whose absence is not visible on Linux: ``tauri-build``
only demands it when building for Windows, so a missing one fails on a
Windows runner and nowhere else. ``src-tauri/tests/security_audit.rs``
asserts every icon the configuration names actually exists, so that gap now
fails on every platform.

The mark is the one the interface's title bar uses — a diamond outline
around a filled centre — on the accent blue, drawn with 4x supersampling for
antialiasing. No third-party imaging library is involved: PNG is zlib plus
four chunks, and ICO is a directory of BMP images, both written here
directly.
"""

from __future__ import annotations

import pathlib
import struct
import zlib

# The accent blue from the interface's stylesheet, as a vertical gradient,
# with the mark in white. A single flat colour reads as a placeholder at
# small sizes; two stops give the shape somewhere to sit.
TOP = (0x3F, 0x7B, 0xFF)
BOTTOM = (0x25, 0x57, 0xC9)
MARK = (0xFF, 0xFF, 0xFF)

SUPERSAMPLE = 4


def rounded_square_covers(x: float, y: float, radius: float) -> bool:
    """Whether (x, y) is inside a rounded square spanning [-1, 1]."""
    inset = 1.0 - radius
    ax, ay = abs(x), abs(y)
    if ax <= inset or ay <= inset:
        return ax <= 1.0 and ay <= 1.0
    dx, dy = ax - inset, ay - inset
    return dx * dx + dy * dy <= radius * radius


def mark_covers(x: float, y: float) -> bool:
    """The ◈ mark: a diamond ring around a filled diamond."""
    d = abs(x) + abs(y)
    ring = 0.50 <= d <= 0.68
    centre = d <= 0.24
    return ring or centre


def render(size: int) -> bytes:
    """Render one RGBA image, row-major, top-down."""
    pixels = bytearray()
    step = 1.0 / (SUPERSAMPLE + 1)
    for py in range(size):
        for px in range(size):
            background = 0
            mark = 0
            for sy in range(1, SUPERSAMPLE + 1):
                for sx in range(1, SUPERSAMPLE + 1):
                    # Map to [-1, 1] with a small margin so the rounded
                    # square does not touch the edge of the canvas.
                    u = ((px + sx * step) / size * 2.0 - 1.0) / 0.94
                    v = ((py + sy * step) / size * 2.0 - 1.0) / 0.94
                    if rounded_square_covers(u, v, 0.42):
                        background += 1
                        if mark_covers(u, v):
                            mark += 1
            samples = SUPERSAMPLE * SUPERSAMPLE
            if background == 0:
                pixels.extend((0, 0, 0, 0))
                continue

            blend = py / max(size - 1, 1)
            base = tuple(
                round(TOP[i] + (BOTTOM[i] - TOP[i]) * blend) for i in range(3)
            )
            weight = mark / samples
            colour = tuple(
                round(base[i] + (MARK[i] - base[i]) * weight) for i in range(3)
            )
            alpha = round(255 * background / samples)
            pixels.extend((colour[0], colour[1], colour[2], alpha))
    return bytes(pixels)


def png(size: int, rgba: bytes) -> bytes:
    """Encode RGBA as a PNG."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    stride = size * 4
    raw = bytearray()
    for row in range(size):
        raw.append(0)  # filter type 0: none
        raw.extend(rgba[row * stride : (row + 1) * stride])

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def bmp_entry(size: int, rgba: bytes) -> bytes:
    """One ICO image, in the BMP form every Windows version reads.

    PNG-compressed ICO entries are legal since Vista but are not universally
    handled by resource tooling; the BMP form has no such question hanging
    over it, and these images are small.
    """
    # BITMAPINFOHEADER, with the doubled height that signals XOR + AND masks.
    header = struct.pack(
        "<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, 0, 0, 0, 0, 0
    )
    xor = bytearray()
    for row in reversed(range(size)):  # bottom-up
        for col in range(size):
            r, g, b, a = rgba[(row * size + col) * 4 : (row * size + col) * 4 + 4]
            xor.extend((b, g, r, a))
    # The AND mask is unused because the alpha channel carries transparency,
    # but the format requires it. Rows are padded to four bytes.
    mask_stride = ((size + 31) // 32) * 4
    and_mask = bytes(mask_stride * size)
    return header + bytes(xor) + and_mask


def ico(images: list[tuple[int, bytes]]) -> bytes:
    """Assemble an ICO from rendered images."""
    entries = [bmp_entry(size, rgba) for size, rgba in images]
    offset = 6 + 16 * len(entries)
    directory = struct.pack("<HHH", 0, 1, len(entries))
    for (size, _), payload in zip(images, entries):
        directory += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,
            size if size < 256 else 0,
            0,
            0,
            1,
            32,
            len(payload),
            offset,
        )
        offset += len(payload)
    return directory + b"".join(entries)


def icns(images: dict[str, bytes]) -> bytes:
    """Assemble an ICNS from PNG payloads.

    ICNS is a magic word, a total length, then typed chunks. The ``ic0*``
    and ``ic1*`` types take a PNG directly, which is why this needs no
    second encoder.
    """
    body = b"".join(
        kind.encode("ascii") + struct.pack(">I", len(payload) + 8) + payload
        for kind, payload in images.items()
    )
    return b"icns" + struct.pack(">I", len(body) + 8) + body


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    icons = root / "apps" / "scirust-studio" / "src-tauri" / "icons"
    icons.mkdir(parents=True, exist_ok=True)

    rendered: dict[int, bytes] = {}

    def image(size: int) -> bytes:
        if size not in rendered:
            rendered[size] = render(size)
        return rendered[size]

    for size, name in [
        (32, "32x32.png"),
        (128, "128x128.png"),
        (256, "128x128@2x.png"),
        (512, "icon.png"),
    ]:
        path = icons / name
        path.write_bytes(png(size, image(size)))
        print(f"{path.relative_to(root)}  {size}x{size}  {path.stat().st_size} bytes")

    ico_sizes = [16, 32, 48, 64, 128, 256]
    path = icons / "icon.ico"
    path.write_bytes(ico([(s, image(s)) for s in ico_sizes]))
    print(
        f"{path.relative_to(root)}  {'/'.join(str(s) for s in ico_sizes)}  "
        f"{path.stat().st_size} bytes"
    )

    # macOS. The chunk types map to the sizes Finder and the Dock ask for.
    path = icons / "icon.icns"
    path.write_bytes(
        icns(
            {
                "ic11": png(32, image(32)),
                "ic12": png(64, image(64)),
                "ic07": png(128, image(128)),
                "ic13": png(256, image(256)),
                "ic09": png(512, image(512)),
            }
        )
    )
    print(f"{path.relative_to(root)}  32/64/128/256/512  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
