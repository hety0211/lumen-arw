"""Ordered, non-destructive inpainting and aligned cloning in source coordinates."""
import cv2
import numpy as np


def apply(source, operations):
    if not any(o['enabled'] and o['opacity'] for o in operations):
        return source
    out = source.copy()
    h, w = source.shape[:2]
    for op in operations:
        if not op['enabled'] or not op['opacity']:
            continue
        pts = np.asarray(op['points']) * [w - 1, h - 1]
        radius = max(1., op['radius'] * min(h, w))
        margin = int(np.ceil(radius * 2 + 8))
        x0, y0 = np.maximum(0, np.floor(pts.min(axis=0)).astype(int) - margin)
        x1, y1 = np.minimum([w, h], np.ceil(pts.max(axis=0)).astype(int) + margin + 1)
        rh, rw = y1 - y0, x1 - x0
        mask = np.zeros((rh, rw), np.uint8)
        local = np.round(pts - [x0, y0]).astype(np.int32)
        r = max(1, round(radius))
        cv2.polylines(mask, [local], False, 255, r * 2, cv2.LINE_8)
        for p in (local[0], local[-1]):
            cv2.circle(mask, tuple(p), r, 255, -1)
        # Feather inward: pixels outside the painted stroke remain exactly intact.
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.minimum(1, distance / max(1, radius * op['feather'] / 100)) * op['opacity'] / 100
        current = out[y0:y1, x0:x1]
        if op['kind'] == 'clone':
            dx, dy = np.asarray(op['offset']) * [w - 1, h - 1]
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            mx, my = xx + dx, yy + dy
            alpha *= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
            replacement = cv2.remap(out, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        else:
            # NS supports single-channel float32, preserving high precision and HDR
            # values. Small sensor-dust regions work best; this is not generative AI.
            replacement = np.stack([cv2.inpaint(np.ascontiguousarray(current[..., c]), mask,
                max(2, min(12, radius / 3)), cv2.INPAINT_NS) for c in range(3)], axis=2)
        out[y0:y1, x0:x1] = current * (1 - alpha[..., None]) + replacement * alpha[..., None]
    return out
