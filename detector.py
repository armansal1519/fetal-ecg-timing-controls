"""Offline four-lead fetal beat estimation. All inference inputs are abdominal voltages."""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d


def bandpass(x, fs, lo, hi):
    sos = signal.butter(
        3, [lo, min(hi, fs * 0.46)], btype="bandpass", fs=fs, output="sos"
    )
    return signal.sosfiltfilt(sos, x, axis=0)


def nearest_distance(locations, events):
    locations = np.asarray(locations)
    if not len(events):
        return np.full(len(locations), np.inf)
    index = np.searchsorted(events, locations)
    a = events[np.maximum(index - 1, 0)]
    b = events[np.minimum(index, len(events) - 1)]
    return np.minimum(np.abs(locations - a), np.abs(locations - b))


def strongest_separated(times, scores, distance):
    """Greedy nonmaximum suppression, for a waveform-only detector."""
    times = np.asarray(times)
    taken = np.zeros(len(times), dtype=bool)
    selected = []
    for j in np.argsort(-np.asarray(scores), kind="stable"):
        if taken[j]:
            continue
        selected.append(j)
        left, right = np.searchsorted(
            times, [times[j] - distance + 1, times[j] + distance]
        )
        taken[left:right] = True
    return np.array(sorted(selected), dtype=int)


def maternal_residual(x, fs):
    qrs = bandpass(x, fs, 8, 38)
    scale = np.maximum(np.median(np.abs(qrs), axis=0), 1e-9)
    power = np.linalg.norm(qrs / scale, axis=1)
    smooth = gaussian_filter1d(power, max(1, 0.012 * fs))
    locs = signal.find_peaks(
        smooth,
        distance=max(1, int(0.32 * fs)),
        height=np.quantile(smooth, 0.65),
        prominence=np.median(smooth) * 0.6,
    )[0]
    if len(locs):
        heights = smooth[locs]
        locs = locs[heights >= 0.29 * np.quantile(heights, 0.85)]
    radius = int(0.024 * fs)
    locs = np.array(
        [
            max(0, k - radius) + np.argmax(power[max(0, k - radius) : k + radius + 1])
            for k in locs
        ],
        int,
    )
    pre, post = round(0.22 * fs), round(0.30 * fs)
    offsets = np.arange(-pre, post + 1)
    weight = signal.windows.tukey(len(offsets), 0.25)[:, None]
    good = locs[(locs >= pre) & (locs < len(x) - post)]
    mean = np.zeros_like(x)
    timed = np.zeros_like(x)
    width = np.zeros_like(x)
    if len(good) >= 4:
        snippets = x[good[:, None] + offsets]
        for j, k in enumerate(locs):
            pos = np.searchsorted(good, k)
            neigh = np.arange(max(0, pos - 12), min(len(good), pos + 13))
            neigh = neigh[np.abs(good[neigh] - k) > 0.32 * fs]
            if not len(neigh):
                continue
            template = np.median(snippets[neigh], axis=0) * weight
            derivative = np.gradient(template, axis=0)
            inds = k + offsets
            keep = (inds >= 0) & (inds < len(x))
            mean[inds[keep]] += template[keep]
            timed[inds[keep]] += derivative[keep]
            width[inds[keep]] += (offsets[:, None] * derivative)[keep]
    return x - mean, (mean, timed, width), locs


def fetal_seeds(residual, maternal, fs):
    qrs = bandpass(residual, fs, 9, 45)
    subsample = qrs[:: max(1, len(qrs) // 25000)]
    eig, vec = eigh(subsample.T @ subsample / len(subsample))
    transform = vec / np.sqrt(np.maximum(eig, max(eig[-1] * 1e-5, 1e-12)))
    w = subsample @ transform
    # Fourth-order eigenvectors reveal non-Gaussian impulse directions.
    _, kurt = eigh((w.T * np.minimum((w * w).sum(axis=1), 100)) @ w / len(w))
    projections = np.column_stack((qrs, qrs @ transform, qrs @ (transform @ kurt)))
    locations = np.arange(len(qrs))
    distance = nearest_distance(locations, maternal)
    best = None
    for c in range(projections.shape[1]):
        envelope = np.abs(projections[:, c])
        masked = envelope.copy()
        masked[distance <= 0.065 * fs] = 0
        peaks = signal.find_peaks(
            masked, distance=max(1, int(0.22 * fs)), height=np.quantile(masked, 0.87)
        )[0]
        if len(peaks) < 4:
            continue
        peaks = peaks[masked[peaks] >= np.quantile(masked[peaks], 0.68) * 0.35]
        if len(peaks) < 4:
            continue
        autoc = signal.correlate(
            masked - masked.mean(), masked - masked.mean(), method="fft", mode="full"
        )
        autoc = autoc[len(masked) - 1 :]
        lmin, lmax = int(0.285 * fs), int(0.64 * fs)
        rr = lmin + int(np.argmax(autoc[lmin : lmax + 1]))
        rate_score = autoc[rr] / max(autoc[0], 1e-15)
        phase_error = np.abs(np.diff(peaks) / rr - np.round(np.diff(peaks) / rr))
        score = rate_score + 0.20 * np.mean(phase_error < 0.14)
        if best is None or score > best[0]:
            best = score, peaks, rr, c
    if best is None:
        return np.array([], int), int(0.43 * fs)
    _, peaks, rr, _ = best
    return peaks, rr


def learn_fallback(records):
    """Fit a temporal fallback pulse only to source-separated training files."""
    pulses = []
    for row in records:
        with np.load(row["path"]) as d:
            wave = d["fetal_component"]
            fs = float(d["fs"])
            r = np.rint(d["fetus"] * fs).astype(int)
            h = round(0.04 * fs)
            r = r[(r >= h) & (r < len(wave) - h)]
            if len(r) == 0:
                continue
            mean = np.median(
                wave[r[:: max(1, len(r) // 60), None] + np.arange(-h, h + 1)], axis=0
            )
            lead = mean[:, np.argmax(np.max(np.abs(mean), axis=0))]
            if np.max(np.abs(lead)) > 0:
                lead = (
                    lead * np.sign(lead[np.argmax(np.abs(lead))]) / np.max(np.abs(lead))
                )
                pulses.append(
                    np.interp(
                        np.linspace(-0.04, 0.04, 41), np.arange(-h, h + 1) / fs, lead
                    )
                )
    if not pulses:
        raise ValueError("Missing labeled source-separated training pulses")
    return np.median(pulses, axis=0)


def whiten_fit(reference):
    # Estimate AR(1) and channel covariance on separate samples.
    limit = np.maximum(np.quantile(np.abs(reference), 0.96, axis=0), 1e-10)
    q = np.clip(reference, -limit, limit)
    phi = np.clip(
        np.sum(q[1:] * q[:-1], axis=0) / (np.sum(q[:-1] ** 2, axis=0) + 1e-15), 0, 0.94
    )
    innovations = q[1:] - phi * q[:-1]
    cov = innovations.T @ innovations / max(1, len(innovations))
    val, vec = eigh(cov * 0.92 + 0.08 * np.diag(np.diag(cov)))
    W = vec / np.sqrt(np.maximum(val, max(1e-12, val[-1] * 1e-5)))
    return phi, W


def innovations(a, phi, w):
    return (a - np.vstack((a[:1], a[:-1])) * phi) @ w


def sequence(times, scores, fs, rr):
    if not len(times):
        return np.array([], int)
    lo, hi = np.quantile(scores, [0.25, 0.85])
    evidence = np.clip((scores - lo) / max(1e-6, hi - lo), -1, 2.5) + 0.20
    value = np.zeros(len(times))
    back = np.full(len(times), -1, int)
    for j, t in enumerate(times):
        a, b = np.searchsorted(times, [t - int(3.8 * rr), t - int(0.22 * fs) + 1])
        if a < b:
            dt = (t - times[a:b]) / rr
            steps = np.clip(np.rint(dt), 1, 4)
            penalty = 4.5 * (dt - steps) ** 2 + 0.45 * (steps - 1)
            choices = value[a:b] - penalty
            q = int(np.argmax(choices))
            if choices[q] > 0:
                back[j] = a + q
                value[j] = choices[q]
        value[j] += evidence[j]
    j = int(np.argmax(value))
    selected = []
    while j >= 0:
        selected.append(j)
        j = back[j]
    return np.asarray(selected[::-1], int)


def detect_all(x, fs, fallback):
    """Return call tables for the proposed detector and its four controls."""
    x = np.asarray(x, dtype=float)
    fs = float(fs)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) < 3 * fs or not np.isfinite(x).all():
        raise ValueError(
            "Need at least three seconds of finite four-lead abdominal signal"
        )
    names = [
        "proposed",
        "waveform_only",
        "matched_filter",
        "subtraction",
        "rhythm_only",
    ]
    keys = [
        "time_s",
        "waveform",
        "k10",
        "snr",
        "rhythm",
        "maternal_distance",
        "low_quality",
    ]
    out = {name: {key: np.array([]) for key in keys} for name in names}
    if np.all(np.ptp(x, axis=0) == 0):
        return out
    x = bandpass(x, fs, 3, 70)
    residual, tangents, mother = maternal_residual(x, fs)
    seeds, rr = fetal_seeds(residual, mother, fs)
    patch = round(0.12 * fs)
    pulse = round(0.04 * fs)
    span = np.arange(-pulse, pulse + 1)
    rival = round(0.06 * fs)
    # Leave space for the entire 80-ms pulse even at the outermost 60-ms rival.
    peak_radius = max(1, patch - rival - pulse)
    # Extra rivals on both sides of the fitted delay, all inside one fixed patch.
    shifts = np.arange(-rival - peak_radius, rival + peak_radius + 1)
    eligible = np.flatnonzero(np.abs(shifts) <= peak_radius)
    offsets = np.arange(-patch, patch + 1)
    seeds = seeds[(seeds > patch) & (seeds < len(x) - patch)]
    samples = []
    size = int(12 * fs)
    for start in range(0, len(x), size):
        stop = min(len(x), start + size)
        outer = np.r_[
            np.arange(max(0, start - 2 * size), max(0, start - patch)),
            np.arange(min(len(x), stop + patch), min(len(x), stop + 2 * size)),
        ]
        if len(outer) < 2 * fs:
            outer = np.setdiff1d(np.arange(len(x)), np.arange(start, stop))
        if len(outer) < 4:
            continue
        phi, W = whiten_fit(residual[outer])
        support = seeds[(seeds < start - 2 * patch) | (seeds > stop + 2 * patch)]
        support = support[np.argsort(abs(support - (start + stop) // 2))[:100]]
        sparse_support = len(support) < 5
        poor = sparse_support or len(mother) < 3
        if sparse_support:
            shape = np.interp(
                span / fs, np.linspace(-0.04, 0.04, len(fallback)), fallback
            )
            ev, vec = eigh(np.cov(residual[outer].T))
            template = shape[:, None] * vec[:, -1] * np.sqrt(max(ev[-1], 1e-12))
        else:
            template = np.median(residual[support[:, None] + span], axis=0)
        template = innovations(
            template * signal.windows.tukey(len(span), 0.28)[:, None], phi, W
        )
        rawenergy = np.sum(template**2)
        variance = float(np.clip(rawenergy, 1, 1e5))
        template = template / max(np.sqrt(rawenergy), 1e-10)
        bank = np.zeros((len(shifts), len(offsets), 4))
        for j, s in enumerate(shifts):
            active = (offsets >= s - pulse) & (offsets <= s + pulse)
            bank[j, active] = template[offsets[active] - s + pulse]
        bank = bank.reshape(len(shifts), -1)
        l = max(0, start - patch - pulse - 2)
        h = min(len(x), stop + patch + pulse + 2)
        block = innovations(residual[l:h], phi, W)
        maternal = np.stack([innovations(u[l:h], phi, W) for u in tangents], axis=-1)
        corr = sum(
            signal.correlate(block[:, k], template[:, k], mode="same", method="auto")
            for k in range(4)
        )
        env = gaussian_filter1d(np.linalg.norm(block, axis=1), max(1, 0.004 * fs))
        maxima = signal.find_peaks(abs(corr), distance=max(1, int(0.13 * fs)))[0] + l
        energy_peaks = signal.find_peaks(env, distance=max(1, int(0.22 * fs)))[0] + l
        coarse = np.arange(start, stop, max(1, int(0.14 * fs)))
        coarse = np.array(
            [
                max(l, int(q - 0.035 * fs))
                + np.argmax(
                    abs(
                        corr[
                            max(0, int(q - 0.035 * fs) - l) : min(
                                len(corr), int(q + 0.035 * fs) - l + 1
                            )
                        ]
                    )
                )
                for q in coarse
            ],
            int,
        )
        times = np.unique(
            np.r_[
                maxima, energy_peaks, coarse, seeds[(seeds >= start) & (seeds < stop)]
            ]
        )
        times = times[(times >= max(start, patch + l)) & (times < min(stop, h - patch))]
        if len(times):
            times = times[
                strongest_separated(
                    times, abs(corr[times - l]), max(1, int(0.065 * fs))
                )
            ]
        for center in times:
            local = center - l + offsets
            y = block[local].ravel()
            columns = maternal[local].reshape(-1, 3)
            U, sv, _ = np.linalg.svd(columns, full_matrices=False)
            U = U[:, sv > max(1e-8, sv[0] * 1e-4)]
            # Rank-three products are faster than constructing a square projector.
            by = bank @ y
            coeff = bank @ U
            proj_corr = by - coeff @ (U.T @ y)
            energy = np.maximum(
                0, np.einsum("ij,ij->i", bank, bank) - np.sum(coeff**2, axis=1)
            )
            amp = variance
            bf = 0.5 * (
                amp * proj_corr**2 / (1 + amp * energy) - np.log1p(amp * energy)
            )
            pick = eligible[np.argmax(bf[eligible])]
            time = center + shifts[pick]
            mf = proj_corr**2 / np.maximum(energy, 1e-9)
            pm = eligible[np.argmax(mf[eligible])]
            plain = eligible[np.argmax(abs(by[eligible]))]
            # Analytic Gram inner products yield the same KL as timing_separation.
            dot = bank @ bank[pick] - coeff @ coeff[pick]
            rivals = (abs(shifts - shifts[pick]) * 1000 > 10 * fs) & (
                abs(shifts - shifts[pick]) * 1000 <= 60 * fs
            )
            v0 = amp * energy[pick]
            ev = energy[rivals]
            b = (v0 - amp * ev) / (1 + amp * ev)
            kl = 0.5 * (
                b
                - np.log1p(b)
                + amp**2
                * np.maximum(energy[pick] * ev - dot[rivals] ** 2, 0)
                / (1 + amp * ev)
            )
            k10 = min(0.5 * (v0 - np.log1p(v0)), np.min(kl))
            baseline = float(
                np.median(
                    abs(
                        corr[
                            max(0, center - l - round(0.3 * fs)) : center
                            - l
                            + round(0.3 * fs)
                            + 1
                        ]
                    )
                )
            )
            snr = abs(by[pick]) / max(0.1, baseline)
            samples.append(
                (
                    time,
                    bf[pick],
                    k10,
                    snr,
                    center + shifts[pm],
                    mf[pm],
                    center + shifts[plain],
                    abs(by[plain]),
                    float(poor),
                )
            )
    if not samples:
        return out
    rows = np.asarray(samples)
    for name in names[:-1]:
        tc, sc = {"matched_filter": (4, 5), "subtraction": (6, 7)}.get(name, (0, 1))
        order = np.argsort(rows[:, tc], kind="stable")
        r = rows[order]
        times = r[:, tc].astype(int)
        selected = (
            strongest_separated(times, r[:, sc], int(0.25 * fs))
            if name == "waveform_only"
            else sequence(times, r[:, sc], fs, rr)
        )
        t = times[selected]
        dt = np.r_[rr, np.diff(t)] if len(t) else np.array([])
        dist = (
            nearest_distance(t, mother) / fs
            if len(mother)
            else np.full(len(t), len(x) / fs)
        )
        vals = [
            t / fs,
            r[selected, sc],
            r[selected, 2],
            r[selected, 3],
            abs(dt / rr - 1),
            dist,
            r[selected, 8],
        ]
        out[name] = dict(zip(keys, vals))
    anchor = int(seeds[0] if len(seeds) else rr)
    rhythm = np.arange(anchor % rr, len(x), rr) / fs
    out["rhythm_only"] = {
        key: (rhythm if key == "time_s" else np.zeros(len(rhythm))) for key in keys
    }
    return out
