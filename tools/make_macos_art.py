"""Draw the macOS app icon (.icns) and the DMG window background (1.3.1).

The Windows icon is a 256 px bitmap; macOS shows icons up to 1024 px on Retina
displays, so the same design (dark tile, sage rounded frame, ring and "L") is
redrawn as vectors on the macOS icon grid (824 px body on a 1024 canvas with a
soft shadow).  Usage: python tools/make_macos_art.py <output folder>
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND, FRAME, FILL = (24, 31, 26), (166, 187, 142), (41, 59, 42)
RING, LETTER = (198, 215, 180), (238, 243, 227)
# PNG-based icns entries: (type, pixel size).
ICNS_TYPES = [(b'icp4', 16), (b'icp5', 32), (b'ic11', 32), (b'ic12', 64), (b'ic07', 128),
              (b'ic13', 256), (b'ic08', 256), (b'ic14', 512), (b'ic09', 512), (b'ic10', 1024)]


def icon(size=1024, supersample=4):
    """macOS icon grid: 824 px body, 100 px margin; the 256 px Windows design scaled onto the body."""
    s = size * supersample
    canvas = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    body = 824 / 1024 * s
    origin = (s - body) / 2
    unit = body / 256                        # one pixel of the original 256 px design

    def box(x0, y0, x1, y1):
        return [origin + x0 * unit, origin + y0 * unit, origin + x1 * unit, origin + y1 * unit]

    shadow = Image.new('L', (s, s), 0)
    ImageDraw.Draw(shadow).rounded_rectangle(box(0, 4, 256, 260), radius=57.6 * unit, fill=110)
    shadow = shadow.filter(ImageFilter.GaussianBlur(14 * supersample))
    canvas.paste(Image.new('RGBA', (s, s), (0, 0, 0, 255)), (0, 0), shadow)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box(0, 0, 256, 256), radius=57.6 * unit, fill=BACKGROUND + (255,))
    draw.rounded_rectangle(box(16, 16, 241, 241), radius=55 * unit, fill=FRAME + (255,))
    draw.rounded_rectangle(box(21, 21, 236, 236), radius=50 * unit, fill=FILL + (255,))
    cx, cy = 128.5, 118.5
    draw.ellipse(box(cx - 70, cy - 70, cx + 70, cy + 70), fill=RING + (255,))
    draw.ellipse(box(cx - 57, cy - 57, cx + 57, cy + 57), fill=FILL + (255,))
    draw.rectangle(box(78, 73, 106, 184), fill=LETTER + (255,))
    draw.rectangle(box(78, 157, 180, 184), fill=LETTER + (255,))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def icns(master):
    chunks = b''
    for kind, size in ICNS_TYPES:
        stream = io.BytesIO()
        master.resize((size, size), Image.Resampling.LANCZOS).save(stream, 'PNG', optimize=True)
        data = stream.getvalue()
        chunks += kind + struct.pack('>I', len(data) + 8) + data
    return b'icns' + struct.pack('>I', len(chunks) + 8) + chunks


def background(scale=1):
    """660 x 400 pt Finder window: app on the left, Applications on the right."""
    w, h = 660 * scale, 400 * scale
    image = Image.new('RGB', (w, h), (25, 29, 27))
    draw = ImageDraw.Draw(image)
    for y in range(h):                       # quiet vertical gradient
        t = y / h
        draw.line([(0, y), (w, y)], fill=(int(29 - 8 * t), int(34 - 9 * t), int(31 - 8 * t)))
    font_path = ROOT / 'assets' / 'NotoSansSC.ttf'
    def font(px):
        try:
            return ImageFont.truetype(str(font_path), px * scale)
        except OSError:
            return ImageFont.load_default()
    title = font(22)
    draw.text((w / 2, 46 * scale), 'LUMEN RAW', font=title, fill=(229, 234, 220), anchor='mm')
    draw.text((w / 2, 76 * scale), '风光与旅行 RAW 工作室 · Apple 芯片版', font=font(13), fill=(135, 147, 137), anchor='mm')
    # Arrow between the two icon positions (180, 200) and (480, 200).
    y = 205 * scale
    draw.line([(262 * scale, y), (390 * scale, y)], fill=(160, 177, 143), width=3 * scale)
    draw.polygon([(398 * scale, y), (384 * scale, y - 9 * scale), (384 * scale, y + 9 * scale)], fill=(160, 177, 143))
    draw.text((w / 2, 330 * scale), '将 LUMEN RAW 拖到「应用程序」文件夹完成安装', font=font(13),
              fill=(201, 213, 190), anchor='mm')
    draw.text((w / 2, 356 * scale), '首次打开若被系统拦截：系统设置 → 隐私与安全性 → 仍要打开', font=font(11),
              fill=(122, 135, 118), anchor='mm')
    return image


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    master = icon()
    master.save(folder / 'LumenRAW-1024.png')
    (folder / 'LumenRAW.icns').write_bytes(icns(master))
    background(1).save(folder / 'dmg-background.png')
    background(2).save(folder / 'dmg-background@2x.png')
    print(f'Icon and DMG background written to {folder}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'build' / 'macos')
