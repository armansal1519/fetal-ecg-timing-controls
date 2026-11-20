"""Four-lead QRS timing information in milliseconds under Gaussian noise."""

import numpy as np


def qrs(time_ms, width=1.0):
    """Symmetric QRS with its positive R apex at zero and analytic time derivative."""
    t = np.asarray(time_ms) / width
    value = np.zeros_like(t, dtype=float)
    slope = np.zeros_like(value)
    for center, spread, amplitude in [(0, 8, 1.0), (-17, 5.5, -0.30), (17, 5.5, -0.30)]:
        z = (t - center) / spread
        wave = amplitude * np.exp(-0.5 * z**2)
        value += wave
        slope -= wave * z / (width * spread)
    return value, slope


def field(wave, direction):
    return (wave[..., None] * direction).reshape(*wave.shape[:-1], -1)


def maternal_basis(t, center, width, direction):
    wave, derivative = qrs(t - center, width)
    # Rescale the translation coordinate by 10 ms to comparable numerical units.
    shapes = [wave, -10 * derivative, -(t - center) * derivative]
    return np.column_stack([field(shape, direction) for shape in shapes])


def basis(matrix):
    u, sing, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = sing > (sing.max() * max(matrix.shape) * np.finfo(float).eps)
    return u[:, keep]


def without(x, nuisance):
    return x - (x @ nuisance) @ nuisance.T


def information(pulse, derivative, maternal, amplitude, sigma):
    """SD bound in ms, information retention and maternal-known SD bound."""
    velocity = amplitude * derivative
    base = without(velocity, basis(pulse[:, None]))
    full = without(velocity, basis(np.column_stack((pulse, maternal))))
    total = base @ base
    retained = np.clip((full @ full) / total, 0, 1)
    bound = sigma / np.linalg.norm(full) if retained > 1e-14 else np.inf
    return bound, retained, sigma / np.sqrt(total)
