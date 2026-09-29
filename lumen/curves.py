"""Shared tone-curve evaluation for the canvas and full-resolution renderer."""
import numpy as np


def evaluate(points, x, mode='smooth'):
    xp, yp = np.asarray(points, dtype=np.float64).T
    x = np.asarray(x)
    if mode == 'linear' or len(points) == 2:
        return np.interp(x, xp, yp)
    # Shape-preserving cubic Hermite interpolation: no overshoot between anchors.
    h = np.diff(xp)
    d = np.diff(yp) / h
    slopes = np.zeros_like(xp)
    slopes[0], slopes[-1] = d[0], d[-1]
    for i in range(1, len(xp) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    index = np.clip(np.searchsorted(xp, x, side='right') - 1, 0, len(xp) - 2)
    t = np.clip((x - xp[index]) / h[index], 0, 1)
    y = ((2*t**3 - 3*t**2 + 1)*yp[index] + (t**3 - 2*t**2 + t)*h[index]*slopes[index]
         + (-2*t**3 + 3*t**2)*yp[index+1] + (t**3 - t**2)*h[index]*slopes[index+1])
    return np.clip(y, 0, 1)
