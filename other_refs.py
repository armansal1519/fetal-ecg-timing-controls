"""Offline fetal timing comparators, four abdominal channels in, seconds out.

See other_methods/METHODS.md for sources, implementation details and departures.
"""

import warnings

import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


def _band(x, fs, limits):
    return signal.sosfiltfilt(
        signal.butter(3, limits, btype="bandpass", fs=fs, output="sos"), x, axis=0
    )


def _ica(x):
    # Fit to evenly spaced observations; transform every native-rate sample.
    step = max(1, len(x) // 30000)
    sv, basis = np.linalg.eigh(np.cov(x[::step], rowvar=False))
    keep = sv > max(1e-20, np.max(sv) * 1e-12)
    if not np.any(keep):
        return np.empty((len(x), 0))
    if not np.all(keep):
        x = x @ basis[:, keep]
    if x.shape[1] == 1:
        return (x - np.mean(x)) / max(float(np.std(x)), 1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        alg = FastICA(
            n_components=x.shape[1],
            whiten="unit-variance",
            random_state=617,
            max_iter=300,
            tol=2e-4,
        )
        alg.fit(x[::step])
    return alg.transform(x)


def _peaks(w, fs, minimum):
    """Magnitude peak selector with a 5-s local amplitude threshold."""
    mag = np.abs(w - np.median(w))
    block = max(1, round(fs * 5))
    bounds = np.arange(0, len(w), block)
    floors = [0.48 * np.quantile(mag[i : i + block], 0.97) for i in bounds]
    threshold = np.interp(
        np.arange(len(w)), np.minimum(bounds + block // 2, len(w) - 1), floors
    )
    p, _ = signal.find_peaks(
        mag, height=np.maximum(threshold, 1e-12), distance=max(1, round(minimum * fs))
    )
    return p


def select_fetal_lead(wave, fs):
    """Choose a single fetal lead by rhythm stability and repeatable QRS morphology."""
    best = (-np.inf, np.array([], dtype=int), 0)
    for ch, w in enumerate(wave.T):
        p = _peaks(w, fs, 0.26)
        if len(p) < 3:
            continue
        rr = np.diff(p) / fs
        period = np.median(rr)
        # Rhythm outside likely fetal range, including residual maternal beats,
        # reduces lead quality. Detections in every lead still use one fixed rule.
        rhythm = (
            np.mean(np.abs(rr - period) < 0.10 * period)
            - 2 * max(0.0, period - 0.62)
            - 2 * max(0.0, 0.31 - period)
        )
        half = round(0.025 * fs)
        valid = p[(p >= half) & (p < len(w) - half)]
        if not len(valid):
            continue
        beats = w[valid[:, None] + np.arange(-half, half + 1)]
        template = np.median(beats, axis=0)
        strength = np.mean(template**2) / max(np.mean(beats**2), 1e-12)
        qual = rhythm + 0.6 * strength
        if qual > best[0]:
            best = (qual, p, ch)
    return best[2], best[1]


def _synthetic_template_frontend(x, fs):
    filtered = _band(x, fs, [3, 75])
    xica = _ica(filtered)
    best = (-np.inf, np.array([], dtype=int))
    for s in np.column_stack((filtered, xica)).T:
        p = _peaks(s, fs, 0.37)
        if len(p) < 4:
            continue
        rr = np.diff(p) / fs
        period = np.median(rr)
        q = np.mean(np.abs(rr - period) < 0.1 * period) + 0.3 * np.log1p(
            np.median(np.abs(s[p])) / max(np.median(np.abs(s)), 1e-8)
        )
        q -= 4 * max(0.0, 0.50 - period) + 2 * max(0.0, period - 1.25)
        if q > best[0]:
            best = (q, p)
    m = best[1]
    left, right = round(0.20 * fs), round(0.33 * fs)
    off = np.arange(-left, right + 1)
    m = m[(m >= left) & (m < len(x) - right)]
    if len(m) < 3:
        return filtered, xica
    beats = filtered[m[:, None] + off]
    segments = [
        np.flatnonzero(mask)
        for mask in (
            off < -0.055 * fs,
            (off >= -0.055 * fs) & (off <= 0.075 * fs),
            off > 0.075 * fs,
        )
    ]
    est = np.zeros_like(filtered)
    weight = np.zeros((len(x), 1))
    taper = signal.windows.tukey(len(off), alpha=0.16)[:, None]
    for i, t in enumerate(m):
        neighbors = np.arange(max(0, i - 10), min(len(m), i + 11))
        neighbors = neighbors[neighbors != i]
        template = np.median(beats[neighbors], axis=0)
        for seg in segments:
            segtemp = template[seg]
            scale = np.clip(
                np.sum(beats[i, seg] * segtemp, axis=0)
                / np.maximum(np.sum(segtemp**2, axis=0), 1e-12),
                -0.5,
                2.0,
            )
            template[seg] *= scale
        est[t + off] += template * taper
        weight[t + off] += taper
    residual = filtered - est / np.maximum(1.0, weight)
    return residual, _ica(residual)


def le(x, fs):
    residual, components = _synthetic_template_frontend(x, fs)
    y = _band(np.column_stack((residual, components)), fs, [10, 60])
    return select_fetal_lead(y, fs)[1] / fs


def _qrs_shape(t, width):
    q = t / width
    e = np.exp(-0.5 * q * q)
    return np.column_stack((e, q * e, (1 - q * q) * e))


def _initial(y, fs, start=0):
    stop = min(len(y), start + round(10 * fs))
    block = y[start:stop]
    lead, seeds = select_fetal_lead(_band(block, fs, [20, min(0.43 * fs, 65)]), fs)
    half = round(0.02 * fs)
    t = np.arange(-half, half + 1)
    seeds = seeds[(seeds >= half) & (seeds < len(block) - half)]
    if len(seeds) < 3:
        return None
    # The wavelet-like seed detector can select either QRS flank or an initial
    # noise excursion. Recover a common fiducial before fitting the QRS model.
    patches = block[seeds[:, None] + t, lead]
    polarity = np.sign(patches.ravel()[np.argmax(np.abs(patches))]) or 1.0
    seeds = seeds + t[np.argmax(patches * polarity, axis=1)]
    seeds = seeds[(seeds >= half) & (seeds < len(block) - half)]
    if len(seeds) < 3:
        return None
    intervals = np.diff(seeds)
    typical = np.median(intervals)
    regular = np.abs(intervals - typical) < 0.15 * typical
    start_idx = np.flatnonzero(regular[:-1] & regular[1:])
    if len(start_idx):
        seeds = seeds[start_idx[0] :]
    median = np.median(block[seeds[:, None] + t], axis=0)
    best = np.inf
    for width_ms in (3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0):
        width = fs * width_ms / 1000.0
        b = _qrs_shape(t, width)
        a = np.linalg.lstsq(b, median, rcond=None)[0]
        err = np.sum((b @ a - median) ** 2)
        if err < best:
            best, z = err, np.r_[a.T.ravel(), width]
    n = y.shape[1]
    sig_a = max(1e-5 * float(np.max(np.square(median))), 1e-8)
    qrs_q = np.diag(np.r_[np.full(3 * n, sig_a), 0.0])
    qrs_p = 3 * qrs_q
    qrs_p[-1, -1] = (0.15 * z[-1]) ** 2
    rr = np.diff(seeds) / fs
    rr = rr[(rr >= 60 / 210) & (rr <= 60 / 50)]
    if not len(rr):
        rr = np.array([0.43])
    history = np.full(6, np.median(rr))
    theta = np.full(6, 1 / 6.0)
    if len(rr) > 7:
        design = np.lib.stride_tricks.sliding_window_view(rr, 7)
        reg = design[:, :6][:, ::-1]
        ii = np.eye(6) * 0.1
        theta = np.linalg.solve(reg.T @ reg + ii, reg.T @ design[:, -1] + ii @ theta)
    v = np.eye(6) * (1e-4 * max(abs(theta)) ** 2)
    q = v.copy()
    return int(start + seeds[0]), t, z, qrs_p, qrs_q, history, theta, v, q


def _wave_jac(t, z, nlead):
    width = z[-1]
    b = _qrs_shape(t, width)
    a = z[:-1].reshape(nlead, 3)
    mean = b @ a.T
    step = max(0.001, 0.001 * width)
    db = (_qrs_shape(t, width + step) - _qrs_shape(t, width - step)) / (2 * step)
    jac = np.zeros((len(t), nlead, 3 * nlead + 1))
    for c in range(nlead):
        jac[:, c, 3 * c : 3 * c + 3] = b
    jac[:, :, -1] = db @ a.T
    return mean, jac


def _track(y, fs, capture=False, reset=True):
    n, leads = y.shape
    # A recording can start with dropout. Initialize in the first usable block.
    state = None
    for first_sample in range(0, n - round(2 * fs), round(5 * fs)):
        state = _initial(y, fs, first_sample)
        if state is not None:
            break
    if state is None:
        return np.empty(0, dtype=int), []
    current, t, z, p, q_w, history, theta, v, q = state
    half = len(t) // 2
    picks = [current]
    failed = 0
    r = 0.010**2
    posteriors = []
    failures = []
    signal_windows = np.lib.stride_tricks.sliding_window_view(
        y, len(t), axis=0
    ).transpose(0, 2, 1)
    while current + int(0.28 * fs) < n - half:
        predicted = np.clip(history @ theta, 60 / 210, 60 / 50)
        lower = max(current + int(np.ceil(fs * max(60 / 210, predicted - 0.20))), half)
        upper = min(
            current + int(np.floor(fs * min(60 / 50, predicted + 0.20))), n - half - 1
        )
        if upper < lower:
            break
        times = np.arange(lower, upper + 1)
        window = y[lower - half : upper + half + 1]
        trim = np.quantile(np.abs(window), 0.90, axis=0)
        mask = np.abs(window) <= trim
        noise = (np.where(mask, window, 0.0) ** 2).sum(0) / np.maximum(mask.sum(0), 1)
        noise = np.maximum(noise, 0.02)
        inv_noise = 1 / noise
        wave, j = _wave_jac(t, z, leads)
        flat = j.reshape(-1, len(z))
        wvec = np.tile(inv_noise, len(t))
        weighted = flat * wvec[:, None]
        # Woodbury form of Gamma_QRS^{-1} for all candidate positions, in the same patch.
        prior = p + q_w
        post = np.linalg.inv(np.linalg.inv(prior) + flat.T @ weighted)
        segments = signal_windows[lower - half : upper - half + 1].reshape(
            len(times), -1
        )
        weighted_wave = (wave * inv_noise).ravel()
        gradient = (segments - wave.ravel()) @ weighted
        improvement = (
            2 * (segments @ weighted_wave)
            - wave.ravel() @ weighted_wave
            + np.einsum("ij,ij->i", gradient @ post, gradient)
        )
        vp = v + q
        hr_var = max(float(history @ vp @ history + r), r)
        errors = (times - current) / fs - predicted
        log_score = improvement - errors**2 / hr_var
        idx = int(np.argmax(log_score))
        if improvement[idx] > 0:
            next_peak = int(times[idx])
            delta = (next_peak - current) / fs
            z += post @ gradient[idx]
            z[-1] = np.clip(z[-1], 0.002 * fs, 0.015 * fs)
            p = (post + post.T) * 0.5
            gain = vp @ history / hr_var
            error = delta - history @ theta
            q_next = max(0.0, error**2 - (history @ v @ history + r)) / max(
                history @ history, 1e-10
            )
            theta += gain * error
            v = vp - np.outer(gain, history @ vp)
            v = (v + v.T) * 0.5
            q = np.eye(6) * min(0.04, q_next)
            history = np.r_[delta, history[:-1]]
            failed = 0
            failures = []
        else:
            next_peak = int(np.clip(round(current + fs * predicted), lower, upper))
            failed += 1
            failures.append((len(picks), times, log_score))
        if capture:
            posteriors.append((next_peak, times, log_score))
        picks.append(next_peak)
        current = next_peak
        if reset and failed > 5 and n - current > round(10 * fs):
            state = _initial(y, fs, current)
            if state is None:
                continue
            current, t, z, p, q_w, history, theta, v, q = state
            # Equation 28: start on the following 10 seconds, track backwards
            # over the failed run, multiply forward and reverse local posteriors.
            first = max(0, picks[failures[0][0]] - round(1.3 * fs))
            last = min(n, current + round(10 * fs))
            _, reverse = _track(
                np.ascontiguousarray(y[first:last][::-1]), fs, capture=True, reset=False
            )
            if reverse:
                bt = last - 1 - np.array([p for p, _, _ in reverse])
                for event, grid, score in failures:
                    bid = int(np.argmin(np.abs(bt - picks[event])))
                    if abs(bt[bid] - picks[event]) > round(0.20 * fs):
                        continue
                    _, rb, rs = reverse[bid]
                    lo = max(
                        int(grid[0]),
                        last - 1 - int(rb[-1]),
                        picks[event - 1] + round(0.28 * fs),
                    )
                    hi = min(int(grid[-1]), last - 1 - int(rb[0]))
                    if event + 1 < len(picks):
                        hi = min(hi, picks[event + 1] - round(0.28 * fs))
                    if hi >= lo:
                        common = np.arange(lo, hi + 1)
                        picks[event] = int(
                            common[
                                np.argmax(
                                    score[common - grid[0]]
                                    + rs[last - 1 - common - rb[0]]
                                )
                            ]
                        )
            if current > picks[-1] + round(0.28 * fs):
                picks.append(current)
            else:
                current = picks[-1]
            failed = 0
            failures = []
    return np.asarray(picks), posteriors


def warmerdam_from_components(y, fs):
    """Paper's HR Kalman, shared-width QRS extended Kalman, MAP and backward recovery."""
    if np.iscomplexobj(y):
        # The physical residual leads and the hierarchical likelihood are real.
        if np.max(np.abs(y.imag)) > 1e-10 * max(1.0, np.max(np.abs(y.real))):
            raise ValueError("Varanini residual ICA unexpectedly returned complex leads")
        y = y.real
    y = np.ascontiguousarray(y / (np.maximum(np.std(y, axis=0), 1e-10)))
    times, _ = _track(y, fs)
    return times / fs


def warmerdam(x, fs):
    # Match the source paper's input after Varanini maternal suppression and ICA.
    from baseline_classic import fetal_components

    y, rate = fetal_components(x, fs)
    if rate != fs:
        from fractions import Fraction

        ratio = Fraction(float(fs) / float(rate)).limit_denominator(100)
        y = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=0)
    return warmerdam_from_components(y, fs)


def predict(method, x, fs):
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] != 4
        or not np.isfinite(x).all()
        or len(x) < int(3 * fs)
    ):
        raise ValueError("Expected at least 3 s of finite four-lead ECG")
    if method == "le":
        return le(x, fs)
    if method == "warmerdam":
        return warmerdam(x, fs)
    if method == "fm_fecg":
        import torch

        from other_methods.neural import peaks

        torch.set_num_threads(4)
        return peaks(x, fs, "mps" if torch.backends.mps.is_available() else "cpu")
    raise ValueError(f"Unknown comparator: {method}")
