"""Float32 linear-light RAW pipeline, masks and optional CUDA / ONNX processing."""
from __future__ import annotations
import os
import colorsys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from .model import COLORS
from . import white_balance, retouch, develop, selection, dng, photo_metadata
from . import curves as tone_curves
from . import large_image, performance

RAW_EXTENSIONS = {'.arw', '.sr2', '.srf', '.dng', '.nef', '.nrw', '.crw', '.cr2', '.cr3', '.raf', '.rw2', '.raw', '.orf'}
RAW_FILTER = 'Sony (*.arw *.sr2 *.srf);;Canon (*.crw *.cr2 *.cr3);;Nikon (*.nef *.nrw);;Fujifilm (*.raf);;Panasonic (*.rw2 *.raw);;DNG (*.dng)'
PHOTO_FILTER = '图片与工程 (' + ' '.join('*'+x for x in sorted(RAW_EXTENSIONS)) + ' *.jpg *.jpeg *.png *.tif *.tiff *.lumen *.lumenalbum);;' + RAW_FILTER + ';;所有文件 (*)'
MAX_EXPORT_PIXELS = large_image.MAX_PIXELS
Image.MAX_IMAGE_PIXELS = MAX_EXPORT_PIXELS


def to_linear(rgb):
    if rgb.nbytes>large_image.MAP_BYTES and rgb.shape[0]>large_image.STRIP_ROWS:
        out=large_image.allocate(rgb.shape)
        for y,block in large_image.strips(rgb):out[y:y+len(block)]=to_linear(block)
        return out
    return np.where(rgb <= .04045, rgb / 12.92, np.maximum((rgb + .055) / 1.055, 0) ** 2.4).astype(np.float32)


def to_srgb(rgb):
    if rgb.nbytes>large_image.MAP_BYTES and rgb.shape[0]>large_image.STRIP_ROWS:
        out=large_image.allocate(rgb.shape)
        for y,block in large_image.strips(rgb):out[y:y+len(block)]=to_srgb(block)
        return out
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055).astype(np.float32)


def resize_limit(image, limit):
    h, w = image.shape[:2]
    if limit and max(h, w) > limit:
        ratio = limit / max(h, w)
        return cv2.resize(image, (max(1, round(w * ratio)), max(1, round(h * ratio))), interpolation=cv2.INTER_AREA)
    return image


def load_image(path, preview_limit=1600):
    """RAW remains linear. Export decodes again at full resolution."""
    path = Path(path)
    info = dict(name=path.name, format=path.suffix[1:].upper(), path=str(path.resolve()))
    if path.suffix.lower() == '.dng' and dng.is_rendered(path):
        width,height=dng.dimensions(path)
        rgb = dng.read(path,preview_limit)
        info.update(width=width,height=height,raw=False,note='Lumen 16-bit 线性 DNG · 已应用编辑')
        info['photo'] = photo_metadata.embedded(path)
    elif path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            info['width'], info['height'] = ((raw.sizes.height,raw.sizes.width) if raw.sizes.flip in (5,6) else (raw.sizes.width,raw.sizes.height))
            info['raw'] = True
            tags, warning = white_balance.metadata(path)
            info['white_balance'] = white_balance.from_metadata(tags, white_balance.estimate(raw))
            info['camera_model'] = tags.get('Model', '')
            info['photo'] = photo_metadata.from_tags(tags)
            info['wb_warning'] = warning
            info['camera_wb_gains'] = list(raw.camera_whitebalance)
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                  gamma=(1, 1), output_color=rawpy.ColorSpace.sRGB,
                                  half_size=bool(preview_limit), user_flip=None,
                                  highlight_mode=rawpy.HighlightMode.Blend).astype(np.float32) / 65535
            if preview_limit:
                curve,label = develop.camera_curve(raw, rgb)
                info['develop'] = dict(mode='camera',curve=curve,source=label)
        info['note'] = 'LibRaw · 相机白平衡 · 线性 sRGB · 16-bit 解码'
    elif path.suffix.lower() in {'.tif', '.tiff', '.png'}:
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError('无法解码这张图片。')
        # Pillow handles EXIF orientation and ICC for 8-bit sources; preserve 16-bit input.
        if arr.dtype == np.uint16:
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            rgb = to_linear(arr[..., :3][..., ::-1].astype(np.float32) / 65535)
            info.update(width=rgb.shape[1], height=rgb.shape[0], raw=False,
                        note='16-bit 输入按 sRGB 解读（建议先转换非 sRGB 文件）')
            info['photo'] = photo_metadata.embedded(path) or photo_metadata.from_tags(white_balance.metadata(path)[0])
        else:
            return load_pillow(path, preview_limit)
    else:
        return load_pillow(path, preview_limit)
    large_image.validate_size((info['height'],info['width']))
    return np.ascontiguousarray(resize_limit(rgb, preview_limit)), info


def load_pillow(path, limit):
    with Image.open(path) as src:
        large_image.validate_size((src.height,src.width))
        img = ImageOps.exif_transpose(src)
        profile = src.info.get('icc_profile')
        if profile:
            import io
            img = ImageCms.profileToProfile(img.convert('RGB'), ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                           ImageCms.createProfile('sRGB'), outputMode='RGB')
        else:
            img = img.convert('RGB')
        info = dict(name=path.name, width=img.width, height=img.height, raw=False,
                    format=path.suffix[1:].upper(), path=str(path.resolve()), note='sRGB · EXIF 方向已应用')
        info['photo'] = photo_metadata.from_tags(white_balance.metadata(path)[0])
        if limit:
            img.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        return to_linear(np.asarray(img).astype(np.float32) / 255), info


class Backend:
    def __init__(self, mode='auto'):
        self.xp = np
        self.name = 'CPU · NumPy / OpenCV'
        self.warning = ''
        if mode != 'cpu':
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount():
                    test = cp.array([1.], dtype=cp.float32)
                    float(cp.asnumpy(test)[0])
                    self.xp = cp
                    name = cp.cuda.runtime.getDeviceProperties(0)['name']
                    self.name = 'CUDA · ' + (name.decode() if isinstance(name, bytes) else str(name))
            except Exception:
                self.warning = '未检测到可用 CuPy / CUDA，已使用 CPU。'

    def tonal(self, image, a):
        try:
            result = self._tonal(image, a, self.xp)
            if self.xp is not np:
                result = self.xp.asnumpy(result)
            return result
        except Exception as exc:
            if self.xp is np:
                raise
            self.xp = np
            self.name = 'CPU · CUDA 回退'
            self.warning = 'CUDA 运行失败，自动回退 CPU：' + str(exc)[:100]
            return self._tonal(image, a, np)

    @staticmethod
    def _tonal(image, a, xp):
        x = xp.asarray(image, dtype=xp.float32).copy()
        temp, tint = a['temperature'] / 100, a['tint'] / 100
        gains = xp.asarray([2 ** (.4 * temp + .15 * tint), 2 ** (-.15 * tint),
                            2 ** (-.4 * temp + .15 * tint)], dtype=xp.float32)
        x *= 2 ** a['exposure'] * gains
        lum = x[..., 0] * .2126 + x[..., 1] * .7152 + x[..., 2] * .0722
        p = xp.clip(lum, 0, 1) ** .45
        sw = (1 - p) ** 2
        hw = p ** 3
        stops = (a['shadows'] * sw + a['highlights'] * hw) / 65
        stops += (a['blacks'] * (1 - p) ** 6 + a['whites'] * p ** 6) / 85
        x *= (2 ** stops)[..., None]
        x = xp.maximum(x, 0)
        x = xp.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)
        if a['contrast']:
            x = (x - .5) * (1 + a['contrast'] / 125) + .5
        return xp.clip(x, 0, 1)


def dehaze(rgb, amount):
    # Dark channel prior with a softened transmission map; CPU OpenCV.
    small = resize_limit(rgb, 900)
    dark = cv2.erode(np.min(small, axis=2), np.ones((15, 15), np.uint8))
    ids = np.argpartition(dark.ravel(), max(0, dark.size - max(1, dark.size // 500)))[-max(1, dark.size // 500):]
    air = np.maximum(np.max(small.reshape(-1, 3)[ids], axis=0), .35)
    norm = np.min(small / air, axis=2)
    transmission = 1 - min(.9, amount / 110) * cv2.erode(norm, np.ones((15, 15), np.uint8))
    transmission = cv2.GaussianBlur(transmission, (0, 0), 7)
    transmission = cv2.resize(transmission, (rgb.shape[1], rgb.shape[0]))
    return np.clip((rgb - air) / np.maximum(transmission[..., None], .22) + air, 0, 1)


def details(rgb, a, detail_scale=1.):
    x = np.ascontiguousarray(rgb, dtype=np.float32)
    if a['denoise'] > 0 or a['color_noise'] > 0:
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)
        if a['denoise'] > 0:
            smooth = cv2.bilateralFilter(lab[..., 0], 7, max(1., a['denoise'] / 4), 3)
            mix = a['denoise'] / 100
            lab[..., 0] = lab[..., 0] * (1 - mix) + smooth * mix
        if a['color_noise'] > 0:
            mix = a['color_noise'] / 100
            for c in (1, 2):
                smooth = cv2.bilateralFilter(lab[..., c], 9, max(1., a['color_noise'] / 3), 4)
                lab[..., c] = lab[..., c] * (1 - mix) + smooth * mix
        x = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    if a['dehaze'] > 0:
        x = dehaze(x, a['dehaze'])
    elif a['dehaze'] < 0:
        x = x * (1 + a['dehaze'] / 220) - a['dehaze'] / 220
    if a['clarity']:
        lum = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(lum, (0, 0), max(1., min(x.shape[:2]) / 90))
        x = np.clip(x + ((lum - blur) * a['clarity'] / 65)[..., None], 0, 1)
    if a.get('texture', 0):
        # Band-pass detail: keep large structures and the finest sensor noise out.
        lum = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        radius = max(.65, min(x.shape[:2]) / 800)
        fine = cv2.GaussianBlur(lum, (0, 0), radius)
        coarse = cv2.GaussianBlur(lum, (0, 0), radius * 3.5)
        band = np.clip(fine - coarse, -.08, .08)
        x = np.clip(x + (band * a['texture'] / 65)[..., None], 0, 1)
    if a['sharpness'] > 0:
        blur = cv2.GaussianBlur(x, (0, 0), max(.45, detail_scale))
        delta = x - blur
        delta *= np.minimum(1, np.abs(delta) / .008)
        x = np.clip(x + delta * a['sharpness'] / 40, 0, 1)
    lum = x[..., 0:1] * .2126 + x[..., 1:2] * .7152 + x[..., 2:3] * .0722
    saturation = 1 + a['saturation'] / 100
    if a['vibrance']:
        spread = np.max(x, axis=2, keepdims=True) - np.min(x, axis=2, keepdims=True)
        saturation = saturation + a['vibrance'] / 100 * (1 - spread)
    return np.clip(lum + (x - lum) * saturation, 0, 1)


def apply_hsl(rgb, values):
    if not np.any(values):
        return rgb
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].copy()
    centers = [c[1] for c in COLORS] + [360]
    controls = np.asarray(list(values) + [values[0]], dtype=np.float32)
    # Circular interpolation: exact control at each center, smooth adjacent transitions.
    dh = np.interp(hue, centers, controls[:, 0]) * .45
    ds = np.interp(hue, centers, controls[:, 1]) / 100
    dv = np.interp(hue, centers, controls[:, 2]) / 100
    hsv[..., 0] = (hue + dh) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + ds), 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * (2 ** dv), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_curves(rgb, curves, mode='linear'):
    x = rgb
    for channel, points in curves.items():
        if points == [[0., 0.], [1., 1.]]:
            continue
        axis = np.linspace(0, 1, 4097) if mode == 'smooth' else np.asarray(points)[:, 0]
        values = tone_curves.evaluate(points, axis, mode)
        if channel == 'RGB':
            x = np.interp(x, axis, values).astype(np.float32)
        else:
            x = x.copy()
            c = 'RGB'.index(channel)
            x[..., c] = np.interp(x[..., c], axis, values)
    return x


def mask_alpha(mask, shape, reference=None):
    h, w = shape[:2]
    feather = mask['feather'] / 100
    if mask['kind'] in ('sky','person','background','color','subject','foreground'):
        alpha = selection.raster_alpha(mask,shape)
        for stroke in mask.get('strokes',[]):
            brush = dict(mask,kind='brush',opacity=100,invert=False,strokes=[dict(stroke,erase=False)])
            paint = mask_alpha(brush,shape)
            alpha = alpha*(1-paint) if stroke.get('erase') else np.maximum(alpha,paint)
        if mask['invert']:
            alpha = 1-alpha
        return alpha * (mask['opacity']/100)
    if mask['kind'] == 'luminance':
        if reference is None or reference.shape[:2] != (h, w):
            raise ValueError('亮度范围蒙版需要原片亮度作为参考。')
        lum = reference[..., 0] * .2126 + reference[..., 1] * .7152 + reference[..., 2] * .0722
        lo, hi = np.array(mask.get('luminance_range', [50., 100.])) / 100
        falloff = max(.0001, mask.get('range_falloff', 20.) / 100)
        left = np.clip((lum - lo + falloff) / falloff, 0, 1)
        right = np.clip((hi + falloff - lum) / falloff, 0, 1)
        alpha = left * left * (3 - 2 * left) * right * right * (3 - 2 * right)
    elif mask['kind'] == 'brush':
        alpha = np.zeros((h, w), np.float32)
        for stroke in mask['strokes']:
            radius = max(1., stroke['radius'] * min(w, h))
            points = [(p[0] * (w - 1), p[1] * (h - 1)) for p in stroke['points']]
            stamps = []
            for i, p in enumerate(points):
                if i == 0:
                    stamps.append(p)
                else:
                    prev = points[i - 1]
                    count = max(1, int(np.hypot(p[0] - prev[0], p[1] - prev[1]) / max(1, radius * .25)))
                    stamps.extend((prev[0] + (p[0] - prev[0]) * j / count,
                                   prev[1] + (p[1] - prev[1]) * j / count) for j in range(1, count + 1))
            for cx, cy in stamps:
                x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius + 2))
                y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius + 2))
                yy, xx = np.ogrid[y0:y1, x0:x1]
                distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
                stamp = np.clip((1 - distance) / max(feather, .01), 0, 1)
                region = alpha[y0:y1, x0:x1]
                if stroke['erase']:
                    region *= (1 - stamp)
                else:
                    np.maximum(region, stamp, out=region)
    else:
        yy, xx = np.ogrid[0:h, 0:w]
        xx, yy = xx / max(w - 1, 1), yy / max(h - 1, 1)
        sx, sy = mask['start']
        ex, ey = mask['end']
        if mask['kind'] == 'linear':
            dx, dy = ex - sx, ey - sy
            alpha = np.clip(((xx - sx) * dx + (yy - sy) * dy) / max(dx * dx + dy * dy, 1e-6), 0, 1)
            alpha = alpha * alpha * (3 - 2 * alpha)
        else:
            cx, cy = (sx + ex) / 2, (sy + ey) / 2
            rx, ry = max(abs(ex - sx) / 2, .001), max(abs(ey - sy) / 2, .001)
            distance = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
            alpha = np.clip((1 - distance) / max(feather, .01), 0, 1)
    alpha = alpha.astype(np.float32)
    if mask['invert']:
        alpha = 1 - alpha
    return alpha * (mask['opacity'] / 100)


def process(source, edits, backend=None, apply_crop=True, detail_scale=1.,_stream=True):
    backend = backend or Backend('cpu')
    large_image.validate_size(source.shape)
    if _stream and source.nbytes>large_image.MAP_BYTES:
        return large_image.process(source,edits,backend,apply_crop,detail_scale)
    # Rotate at output only: mask / crop coordinates always refer to the original image.
    gain = np.asarray(edits.get('wb_gain', [1., 1., 1.]), np.float32) * white_balance.gains(edits.get('white_balance', {}))
    repaired = develop.apply(retouch.apply(source, edits.get('retouch', [])), edits.get('develop', {}))
    balanced = repaired if np.all(gain == 1) else repaired * gain
    x = backend.tonal(balanced, edits['adjustments'])
    x = details(x, edits['adjustments'], detail_scale)
    x = apply_hsl(x, edits['hsl'])
    x = apply_curves(x, edits['curves'], edits.get('curve_mode', 'linear'))
    if edits.get('monochrome', False):
        lum = x[..., 0] * .2126 + x[..., 1] * .7152 + x[..., 2] * .0722
        x = np.repeat(lum[..., None], 3, axis=2)
    x = color_grade(x, edits.get('grading', {}))
    mask_reference = np.clip(to_srgb(develop.apply(source,edits.get('develop',{}))), 0, 1) if any(m['kind'] == 'luminance' for m in edits['masks']) else None
    for mask in edits['masks']:
        if not mask['enabled'] or not any(mask['adjustments'].values()):
            continue
        alpha = mask_alpha(mask, x.shape, mask_reference)[..., None]
        local = details(backend.tonal(to_linear(x), mask['adjustments']), mask['adjustments'], detail_scale)
        x = x * (1 - alpha) + local * alpha
    x = finishing(x, edits.get('effects', {}), edits.get('crop'))
    if apply_crop:
        x = crop_rotate(x, edits)
    return np.ascontiguousarray(np.clip(x, 0, 1), dtype=np.float32)


def color_grade(rgb, grading):
    """Three tonal wheels with luminance-neutral tint vectors and soft weights."""
    if not any(grading.get(z, [0, 0])[1] for z in ('shadows', 'midtones', 'highlights')):
        return rgb
    import colorsys
    lum = rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722
    tone = np.clip(lum + grading.get('balance', 0) / 300, 0, 1)
    weights = [(1 - tone) ** 2, 2 * tone * (1 - tone), tone ** 2]
    out = rgb.copy()
    protection = .2 + .8 * np.sin(np.pi * np.clip(lum, 0, 1))
    for zone, weight in zip(('shadows', 'midtones', 'highlights'), weights):
        hue, strength = grading.get(zone, [0, 0])
        color = np.asarray(colorsys.hsv_to_rgb((hue % 360) / 360, 1, 1), np.float32)
        color -= np.dot(color, [.2126, .7152, .0722])
        out += (weight * protection * strength / 350)[..., None] * color
    return np.clip(out, 0, 1)


def finishing(rgb, settings, crop=None):
    vignette, grain = settings.get('vignette', 0), settings.get('grain', 0)
    if not vignette and not grain:
        return rgb
    h, w = rgb.shape[:2]
    x = rgb.copy()
    if vignette:
        a, b, c, d = crop or [0, 0, 1, 1]
        yy, xx = np.ogrid[0:h, 0:w]
        nx = (xx / max(w - 1, 1) - (a + c) / 2) / max((c - a) / 2, .001)
        ny = (yy / max(h - 1, 1) - (b + d) / 2) / max((d - b) / 2, .001)
        radius = np.sqrt(nx * nx + ny * ny) / 1.41421356
        midpoint = .05 + settings.get('midpoint', 50) / 100 * .75
        softness = .08 + settings.get('feather', 70) / 100 * .72
        edge = np.clip((radius - midpoint) / softness, 0, 1)
        edge = edge * edge * (3 - 2 * edge)
        if vignette < 0:
            x *= (2 ** (edge * vignette / 55))[..., None]
        else:
            x += (1 - x) * (edge * vignette / 170)[..., None]
    if grain:
        # A deterministic image-space grain field; slider updates never flicker.
        cells = int(1600 / (1 + settings.get('grain_size', 30) / 20))
        gh, gw = max(4, round(cells * h / max(h, w))), max(4, round(cells * w / max(h, w)))
        noise = np.random.default_rng(17031).normal(0, 1, (gh, gw)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        lum = np.mean(x, axis=2)
        response = .35 + .65 * (4 * lum * (1 - lum))
        x += (noise * response * grain / 2200)[..., None]
    return np.clip(x, 0, 1)


def auto_tone(source, gains=None):
    sample = resize_limit(source, 500)
    if gains is not None:
        sample = sample * np.asarray(gains, dtype=np.float32)
    lum = sample[..., 0] * .2126 + sample[..., 1] * .7152 + sample[..., 2] * .0722
    dark, mid, light = np.percentile(lum, [5, 50, 98])
    if light < 1e-5:
        return dict(exposure=0., shadows=0., highlights=0., blacks=0., whites=0., contrast=0.)
    exposure = float(np.clip(np.log2(.18 / max(mid, .003)), -2.5, 2.5))
    exposure = min(exposure, float(np.log2(1.6 / max(light, .01))))
    high_after, low_after = light * 2 ** exposure, dark * 2 ** exposure
    return dict(exposure=round(exposure, 2), shadows=round(float(np.clip((.05 - low_after) * 550, 0, 40))),
                highlights=round(float(np.clip((.7 - high_after) * 65, -65, 0))),
                blacks=-5., whites=0., contrast=5.)


def sample_white_balance(source, position):
    h, w = source.shape[:2]
    x, y = round(position[0] * (w - 1)), round(position[1] * (h - 1))
    radius = max(2, round(min(h, w) / 150))
    patch = source[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    sample = np.median(patch, axis=(0, 1))
    if np.min(sample) < .002 or np.max(sample) >= .985:
        raise ValueError('这个区域太暗或已过曝，请选择有细节的中性灰／白色区域。')
    target = float(sample @ np.array([.2126, .7152, .0722]))
    return np.clip(target / sample, .125, 8).astype(float).tolist()


def crop_rotate(image, edits):
    x = image
    if edits['crop']:
        h, w = x.shape[:2]
        a, b, c, d = edits['crop']
        x0, y0 = min(w - 1, int(a * w)), min(h - 1, int(b * h))
        x1, y1 = max(x0 + 1, min(w, round(c * w))), max(y0 + 1, min(h, round(d * h)))
        x = x[y0:y1, x0:x1]
    angle = edits.get('straighten', 0)
    if abs(angle) > .001:
        h, w = x.shape[:2]
        theta = np.deg2rad(abs(angle))
        # Enlarge the rotated image to cover all output corners, preserving crop ratio.
        scale = max(np.cos(theta) + (h / w) * np.sin(theta), np.cos(theta) + (w / h) * np.sin(theta))
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
        x = cv2.warpAffine(x, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(np.rot90(x, -edits['rotation']))


def super_resolve(rgb, scale=2, model_path=None, use_cuda=True, progress=None, cancel=None):
    h, w = rgb.shape[:2]
    if scale not in (1, 2, 4):
        raise ValueError('仅支持 1×、2× 或 4×。')
    if h * w * scale * scale > MAX_EXPORT_PIXELS:
        raise ValueError('放大后超过 4 亿像素，请先裁切或降低倍数。')
    if scale == 1:
        return rgb, '原始尺寸'
    if cancel is not None and cancel.is_set():
        raise InterruptedError('已取消增强')
    if model_path == ':quality:':
        from .restoration import super_resolution
        return super_resolution(rgb,scale,use_cuda,progress,cancel)
    if model_path == ':builtin:':
        path = Path(__file__).resolve().parents[1] / 'assets' / 'models' / 'realesr-general-x4v3.onnx'
        if not path.exists():
            raise FileNotFoundError('内置增强模型缺失，请重新解压完整软件包。')
        return onnx_super_resolve(rgb, scale, path, use_cuda, progress, cancel, native_scale=4)
    if model_path:
        return onnx_super_resolve(rgb, scale, model_path, use_cuda, progress, cancel)
    # Classical single-image iterative back projection (not a neural model).
    out = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    for _ in range(3):
        error = rgb - cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        out += .65 * cv2.resize(error, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0, 1), 'Lanczos + 迭代反投影 · CPU'


def onnx_super_resolve(rgb, scale, path, use_cuda, progress=None, cancel=None, native_scale=None):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('神经网络超分需要安装 onnxruntime-gpu 或 onnxruntime。') from exc
    providers = ['CPUExecutionProvider']
    if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    options = {}
    if hasattr(ort, 'SessionOptions'):
        opts = performance.session_options()
        options['sess_options'] = opts
    try:
        sess = ort.InferenceSession(str(path), providers=providers, **options)
    except Exception:
        if len(providers) == 1:
            raise
        sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'], **options)
    inp = sess.get_inputs()
    if len(inp) != 1 or inp[0].type != 'tensor(float)' or len(inp[0].shape) != 4:
        raise ValueError('模型需为单输入 float32 NCHW RGB，范围 0–1。')
    if isinstance(inp[0].shape[1], int) and inp[0].shape[1] != 3:
        raise ValueError('模型输入需为 3 通道 RGB。')
    if any(isinstance(n, int) for n in inp[0].shape[2:]):
        raise ValueError('分块超分要求模型支持动态宽高，当前模型为固定尺寸。')
    h, w = rgb.shape[:2]
    out = large_image.allocate((h * scale, w * scale, 3))
    native = native_scale or scale
    tile, pad = 192, 40
    total = ((h + tile - 1) // tile) * ((w + tile - 1) // tile)
    done = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if cancel is not None and cancel.is_set():
                raise InterruptedError('已取消增强')
            ey, ex = min(h, y + tile), min(w, x + tile)
            sy, sx = max(0, y - pad), max(0, x - pad)
            ty, tx = min(h, ey + pad), min(w, ex + pad)
            block = rgb[sy:ty, sx:tx].transpose(2, 0, 1)[None].copy()
            predicted = sess.run(None, {inp[0].name: block})[0]
            expected = (1, 3, (ty - sy) * native, (tx - sx) * native)
            if predicted.shape != expected:
                raise ValueError(f'模型输出尺寸 {predicted.shape} 与所选 {scale}× 不匹配。')
            predicted = predicted[0].transpose(1, 2, 0)
            if not np.isfinite(predicted).all():
                raise ValueError('超分模型返回无效像素。')
            if native != scale:
                predicted = cv2.resize(predicted, ((tx - sx) * scale, (ty - sy) * scale), interpolation=cv2.INTER_AREA)
            out[y * scale:ey * scale, x * scale:ex * scale] = predicted[(y - sy) * scale:(ey - sy) * scale, (x - sx) * scale:(ex - sx) * scale]
            done += 1
            if progress:
                progress(done, total)
    label = 'Real-ESRGAN · ' if native_scale else 'ONNX · '
    provider = sess.get_providers()[0]
    if use_cuda and provider == 'CPUExecutionProvider':
        label += 'CUDA 不可用，'
    return np.clip(out, 0, 1, out=out), label + provider


def export_image(path, rgb, quality=95, photo=None, provenance=None):
    """Atomic export. All outputs are encoded sRGB; TIFF preserves 16-bit precision."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng'}:
        raise ValueError('请选择 JPEG、PNG、TIFF 或 DNG 格式。')
    large_image.validate_size(rgb.shape)
    if not large_image.finite(rgb):
        raise ValueError('图像包含无效像素。')
    temp = path.with_name(path.stem + '.lumen-tmp' + ext)
    try:
        if ext == '.dng':
            dng.write(temp,rgb,photo,provenance)
        elif ext in {'.tif', '.tiff'}:
            import tifffile
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
            tifffile.imwrite(str(temp), large_image.encode_strips(rgb),shape=rgb.shape,dtype=np.uint16,
                             rowsperstrip=large_image.STRIP_ROWS,photometric='rgb', metadata=None,
                             description=photo_metadata.description(photo,provenance) if photo is not None else None,
                             extratags=[(34675, 'B', len(profile), profile, False)])
        else:
            data=large_image.allocate(rgb.shape,np.uint8)
            for y,block in large_image.strips(rgb):data[y:y+len(block)]=np.round(np.clip(block,0,1)*255).astype(np.uint8)
            image = Image.fromarray(data)
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
            options = dict(icc_profile=profile)
            if ext in {'.jpg', '.jpeg'}:
                options.update(quality=quality, subsampling=0)
            image.save(temp, **options)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
