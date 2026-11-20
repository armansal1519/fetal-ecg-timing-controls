"""Four-lead, count-preserving, cross-fitted fetal R-peak timing refiners."""

from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.special import softmax
from detector import bandpass, maternal_residual, nearest_distance
import baseline_classic

RATE = 500
GRID = np.arange(-14, 15)
NGRID = len(GRID)
LAGS = np.array([-8, -4, 0, 4, 8])
VIEW_NAMES = ("abdominal", "ica")
VIEW_FEATURES = (
    "cosine",
    "subspace",
    "norm",
    "project_m16",
    "project_m8",
    "project",
    "project_p8",
    "project_p16",
    "template_m8",
    "template",
    "template_p8",
)
NFEATURES = 2 * len(VIEW_FEATURES) + 3
STATE = Path(__file__).resolve().parent / "state/model_weights.json"


def resample(x, fs):
    ratio = Fraction(RATE / float(fs)).limit_denominator(10000)
    if ratio == 1:
        return np.asarray(x, dtype=float)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def whiten(x):
    # Winsorization protects scale estimation while retaining original beat amplitudes.
    sub = x[:: max(1, len(x) // 30000)]
    lo, hi = np.quantile(sub, [0.025, 0.975], axis=0)
    cov = np.cov(np.clip(sub, lo, hi).T)
    cov = 0.88 * cov + 0.12 * np.diag(np.diag(cov))
    ev, vec = np.linalg.eigh(cov)
    return x @ (vec / np.sqrt(np.maximum(ev, max(float(ev[-1]) * 1e-6, 1e-16))))


def per_view(view, support, centers):
    # Template and all timing candidates on the same 500-Hz grid.
    template_span = np.arange(-46, 47)
    support_wave = view[support[:, None] + template_span]
    template = np.median(support_wave, axis=0)
    magnitude = np.linalg.norm(template, axis=1)
    center = int(np.argmax(magnitude[46 - 16 : 46 + 17]) + 46 - 16)
    direction = template[center] / max(magnitude[center], 1e-8)
    core = np.arange(-19, 20)
    pulse = template[center + core].copy()
    taper = signal.windows.tukey(len(core), 0.30)
    pulse *= taper[:, None]
    idx = centers[:, None] + GRID
    patch = view[idx[:, :, None] + core]
    dot = np.einsum("bsdc,dc->bsc", patch, pulse)
    energy = np.einsum("bsdc,bsdc,d->bsc", patch, patch, taper * taper)
    pulse_energy = np.sum(pulse * pulse, axis=0)
    cosine = dot.sum(-1) / np.sqrt(
        np.maximum(energy.sum(-1) * pulse_energy.sum(), 1e-15)
    )
    subspace = np.sum(dot**2 / np.maximum(energy, 1e-10), axis=-1) / max(
        pulse_energy.sum(), 1e-10
    )
    amp = np.linalg.norm(view[idx], axis=-1)
    amp /= np.maximum(amp.max(axis=1, keepdims=True), max(np.median(magnitude), 0.1))
    # Resolve signed per-beat amplitudes on the polar axis of the coherent deflection.
    projection = np.einsum("bskc,c->bsk", view[idx[:, :, None] + LAGS], direction)
    projection /= np.maximum(
        np.max(np.abs(projection), axis=(1, 2), keepdims=True), 0.1
    )
    # Beat-level fit determines a translation; relative anatomy of the median QRS
    # contributes a separate, denoised fiducial likelihood.
    best = np.argmax(cosine, axis=1)
    relative = center + GRID[None, :] - GRID[best, None]
    tp = (
        template[
            np.clip(relative[:, :, None] + np.array([-4, 0, 4]), 0, len(template) - 1)
        ]
        @ direction
    )
    tp /= max(magnitude[center], 0.1)
    strength = magnitude[center] / max(
        np.median(np.linalg.norm(support_wave - template, axis=-1)), 0.1
    )
    return np.concatenate(
        [cosine[:, :, None], subspace[:, :, None], amp[:, :, None], projection, tp],
        axis=-1,
    ), strength


def extract(x, fs, anchors):
    x = np.asarray(x, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError("four finite abdominal leads required")
    if (
        np.any(np.diff(anchors) <= 0)
        or np.any(anchors < 0)
        or np.any(anchors >= len(x) / fs)
    ):
        raise ValueError("invalid anchors")
    out = np.zeros((len(anchors), NGRID, NFEATURES), np.float32)
    # Default at boundaries and for records without external template beats is the anchor.
    out[:, :, -3] = -((GRID[None, :] / 14) ** 2)
    if not len(anchors):
        return out
    y = resample(x, fs)
    res, _, mother = maternal_residual(bandpass(y, RATE, 3, 75), RATE)
    components, rate = baseline_classic.fetal_components(x, fs)
    views = [
        whiten(bandpass(res, RATE, 7, 60)),
        whiten(bandpass(resample(components, rate), RATE, 7, 60)),
    ]
    n = min(len(v) for v in views)
    t = np.rint(anchors * RATE).astype(int)
    lag = nearest_distance(t, mother)
    interior = (t >= 48) & (t < n - 48)
    for group in np.unique(t // (RATE * 10)):
        target = np.flatnonzero((t // (RATE * 10) == group) & interior)
        if not len(target):
            continue
        clean = interior & ((t < t[target[0]] - 100) | (t > t[target[-1]] + 100))
        outside = np.flatnonzero(clean & (lag >= 32))
        if len(outside) < 6:
            outside = np.flatnonzero(clean)
        if len(outside) < 4:
            continue
        neighbors = outside[np.argsort(np.abs(t[outside] - np.median(t[target])))[:80]]
        f0, s0 = per_view(views[0], t[neighbors], t[target])
        f1, s1 = per_view(views[1], t[neighbors], t[target])
        # Both refiners see precisely the same two whitened views, external
        # fiducial templates, candidate shifts, and signal-quality weights.
        weights = np.clip(np.array([s0, s1]), 0.05, 20)
        weights /= weights.sum()
        nf = len(VIEW_FEATURES)
        out[target, :, :nf] = f0 * weights[0]
        out[target, :, nf : 2 * nf] = f1 * weights[1]
        out[target, :, -2] = np.maximum(0, 1 - lag[target, None] / 28) * (
            f0[:, :, 0] * weights[0] + f1[:, :, 0] * weights[1]
        )
        out[target, :, -1] = np.minimum(f0[:, :, 0], f1[:, :, 0])
    return out


def condition_geometry(features):
    """Covariate-dependent exponential family on delays.

    Interactions let local geometry respond to maternal overlap and to the
    reliability of the template without changing the number of initial calls.
    """
    mf = features[:, :, 0] + features[:, :, len(VIEW_FEATURES)]
    strength = np.clip((mf.max(1) - 0.35) / 0.50, 0, 1)[:, None, None]
    ix = np.argmax(abs(mf), axis=1)
    ii = np.arange(len(mf))
    proximity = np.clip(
        features[ii, ix, -2] / np.where(abs(mf[ii, ix]) > 1e-9, mf[ii, ix], 1), 0, 1
    )[:, None, None]
    return np.concatenate([features, features * strength, features * proximity], axis=2)


def solutions(features, anchors, duration, state):
    anchors = np.asarray(anchors)
    if not len(anchors):
        return {
            k: anchors.copy()
            for k in ("v2", "matched_filter", "power_mf", "raw_matched_filter")
        }
    anchor_prior = features[:, :, -3]
    raw_mf = features[:, :, 0] + features[:, :, len(VIEW_FEATURES)]
    strength = np.ptp(raw_mf, axis=1)

    def decision(coef, window, gate):
        design = (
            condition_geometry(features) if len(coef) > features.shape[-1] else features
        )
        logits = design @ np.asarray(coef, dtype=np.float32)
        posterior = softmax(logits, axis=1)
        utility = (np.abs(GRID[:, None] - GRID[None, :]) <= window).astype(np.float32)
        scores = posterior @ utility
        result = (scores + 1e-7 * anchor_prior).argmax(1)
        return np.where((strength > 0) & (strength >= gate), result, len(GRID) // 2)

    best = decision(
        state["coef"], state["half_window_samples"], state["min_profile_range"]
    )
    if state.get("mf_coef") is not None:
        mf_idx = decision(
            state["mf_coef"],
            state["mf_half_window_samples"],
            state["mf_min_profile_range"],
        )
    else:
        mf_idx = np.argmax(raw_mf + 1e-6 * anchor_prior, axis=1)
    # High two-view template coherence reduces the allowable discrepancy
    # between the geometric and conventional normalized-correlation modes.
    agreement = state.get("agreement_blend", 0.0)
    if state.get("record_reliability", False):
        eligible = strength > 1e-7
        quality = np.median(raw_mf.max(1)[eligible]) if np.any(eligible) else 0.0
        agreement *= np.clip((quality - 0.45) / 0.35, 0.0, 1.0)
    raw_index = np.argmax(raw_mf + 1e-6 * anchor_prior, axis=1)
    best = np.rint((1 - agreement) * best + agreement * raw_index).astype(int)
    if state.get("control_shared_fusion", False):
        mf_idx = np.rint((1 - agreement) * mf_idx + agreement * raw_index).astype(int)
    candidates = np.rint(anchors[:, None] * RATE) / RATE + GRID[None, :] / RATE
    # Keep original sub-sample time when the safety gate selects no shift.
    candidates[:, len(GRID) // 2] = anchors
    allrows = np.arange(len(anchors))
    bounded = lambda time: np.clip(time, 0, np.nextafter(duration, 0))
    return dict(
        v2=bounded(candidates[allrows, best]),
        matched_filter=bounded(candidates[allrows, mf_idx]),
        power_mf=anchors,
        raw_matched_filter=bounded(candidates[allrows, raw_index]),
    )


def predict(x, fs, anchors=None, state=None):
    if anchors is None:
        anchors = baseline_classic.predict("power_mf", x, fs)
    if state is None:
        state = json.loads(STATE.read_text())
    return solutions(extract(x, fs, anchors), anchors, len(x) / fs, state)
