"""Simulated patient test of the original V2 model and timing information."""

import hashlib
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import baseline_classic
from scoring import match, score_record
from v2 import model
from .geometry import field, information, maternal_basis, qrs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / "results"
FS = 500
DURATION = 72
METHODS = ("power_mf", "raw_matched_filter", "matched_filter", "v2")


def verify():
    files = json.loads((ROOT / "v2/state/model_lock.json").read_text())["files"]
    seal = {
        "v2/model.py": files["v2/model.py"],
        "v2/state/model_weights.json": files["v2/state/model_weights.json"],
        "q1/theory/protocol.md": "4caa7f3666749a94e74b7340d49021e8d3149b7902837cac63938d8e6ad640a7",
    }
    for path, expected in seal.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file mismatch: {path}")
    return seal


def events(rng, bpm, width, fast):
    phase = rng.uniform(0, 2*np.pi)
    period = rng.uniform(3, 8)
    time = rng.uniform(0.16, 0.57)
    times, widths, amplitudes = [], [], []
    while time < DURATION - 0.12:
        times.append(time)
        widths.append(width * rng.lognormal(0, 0.035))
        amplitudes.append(rng.lognormal(0, 0.08))
        interval = 60 / bpm * (1 + 0.04*np.sin(2*np.pi*time/period + phase))
        time += interval * rng.lognormal(0, 0.025 if fast else 0.045)
    return np.array(times), np.array(widths), np.array(amplitudes)


def add_beats(y, times, widths, amplitudes, direction, fast):
    for instant, width, amplitude in zip(times, widths, amplitudes):
        left = max(0, int((instant - .25) * FS))
        right = min(len(y), int((instant + .35) * FS))
        t = (np.arange(left, right) / FS - instant) * 1000
        pulse = qrs(t, width)[0]
        scale = .8 if fast else 1.0
        p = .07 * np.exp(-.5 * ((t + 130*scale) / (20*scale))**2)
        twave = .19 * np.exp(-.5 * ((t - 180*scale) / (35*scale))**2)
        y[left:right] += amplitude * (pulse + p + twave)[:, None] * direction


def partition(rng, cosine):
    fetal = rng.normal(size=4)
    fetal /= np.linalg.norm(fetal)
    auxiliary = rng.normal(size=4)
    auxiliary -= (auxiliary @ fetal) * fetal
    auxiliary /= np.linalg.norm(auxiliary)
    return fetal, cosine*fetal + np.sqrt(1 - cosine**2)*auxiliary


def simulate(i):
    rng = np.random.default_rng(np.random.SeedSequence([80742619, i]))
    cosine = (0., .55, .88, .98)[i % 4]
    sigma = (.18, .40)[(i // 4) % 2]
    fetal_direction, mother_direction = partition(rng, cosine)
    f, fw, fa = events(rng, rng.uniform(120, 154), rng.uniform(.84, 1.10), True)
    m, mw, ma = events(rng, rng.uniform(63, 90), rng.uniform(1.12, 1.40), False)
    x = sigma * rng.normal(size=(FS*DURATION, 4))
    add_beats(x, f, fw, fa, fetal_direction, True)
    add_beats(x, m, mw, ma * rng.uniform(4.5, 6.0), mother_direction, False)
    # Original code expects nominal microvolt ranges.
    x *= 100
    bounds, fraction, separations = [], [], []
    for instant, width, amplitude in zip(f, fw, fa):
        nearest = np.argmin(abs(m - instant))
        center = (m[nearest] - instant) * 1000
        # Sample the same clock as the recording, retaining sub-grid fiducials.
        k = int(np.rint(instant * FS))
        t = ((np.arange(-50, 51) + k) / FS - instant) * 1000
        wave, slope = qrs(t, width)
        nuisance = maternal_basis(t, center, mw[nearest], mother_direction)
        bound, retained, _ = information(
            field(wave, fetal_direction), field(-slope, fetal_direction), nuisance, amplitude, sigma
        )
        bounds.append(bound)
        fraction.append(retained)
        separations.append(center)
    return (x, f, m, np.array(bounds), np.array(fraction), np.array(separations), cosine, sigma)


def patient(i):
    x, f, m, bound, fraction, distance, cosine, sigma = simulate(i)
    failed = ""
    with tempfile.TemporaryDirectory() as scratch:
        baseline_classic.CACHE = Path(scratch)
        try:
            outputs = model.predict(x, FS)
        except (RuntimeError, ValueError) as exc:
            # A native detector failure represents zero predictions for every
            # anchor-dependent method under the original study convention.
            if "FecgQRSmCanc at line 120" not in str(exc) and "Too few preliminary" not in str(exc):
                raise
            failed = str(exc).splitlines()[-1]
            outputs = {method: np.array([]) for method in METHODS}
    label = f"qrs_{i+1:02d}"
    np.savez_compressed(
        DEST / "synthetic_predictions" / (label + ".npz"), **outputs, fetal=f, maternal=m,
        bound_ms=bound, retention=fraction, separation_ms=distance, cosine=cosine, sigma=sigma,
    )
    collision = np.rint(abs(distance) * 1000) <= 40_000
    severe = collision & (fraction < .65)
    mild = collision & ~severe
    anchors = outputs["power_mf"]
    pa = match(f, anchors, 28)
    # Avoid spurious sub-ULP differences when comparing errors or ties.
    picked = pa[collision[pa[:, 0]]]
    rows = []
    for name in METHODS:
        pred = outputs[name]
        s, _, _, _ = score_record(f, m, [], pred, DURATION)
        pairs = match(f, pred, 10)
        hit = np.zeros(len(f), bool)
        hit[pairs[:, 0]] = True
        assert int(sum(hit & collision)) == s["collision_tp"]
        rows.append(dict(
            subject=label, method=name, mixing_cosine=cosine, noise_sd=sigma,
            native_failure=failed, n_ref=len(f), n_pred=len(pred),
            n_collision=int(sum(collision)), hit_collision=int(sum(hit & collision)),
            severe_n=int(sum(severe)), severe_hit=int(sum(hit & severe)),
            mild_n=int(sum(mild)), mild_hit=int(sum(hit & mild)),
            collision_recall=s["collision_recall"], f1_50=s["f1_50"], fp_min50=s["fp_min50"],
        ))
    relative = pd.DataFrame({
        "subject": label, "reference_id": picked[:, 0], "power_mf_anchor_id": picked[:, 1],
        "bound_ms": bound[picked[:, 0]], "retention": fraction[picked[:, 0]],
    })
    for name in METHODS:
        relative[name + "_abs_ms"] = abs(outputs[name][picked[:, 1]] - f[picked[:, 0]]) * 1000
    return rows, relative


def contrast(value):
    delta = np.asarray(value.dropna(), dtype=float)
    if not len(delta):
        return dict(subjects=0, average=None, ci_low=None, ci_high=None)
    rng = np.random.default_rng(433719)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
    return dict(subjects=len(delta), average=delta.mean(), ci_low=low, ci_high=high)


def summarize(frame, anchor):
    frame.to_csv(DEST / "transfer_subjects.csv", index=False)
    anchor.to_csv(DEST / "transfer_anchor_conditioned.csv", index=False)
    contrasts = []
    for c in ("severe", "mild"):
        frame[c+"_recall"] = frame[c+"_hit"] / frame[c+"_n"].replace(0, np.nan)
    frame["severe_minus_mild"] = frame.severe_recall - frame.mild_recall
    for metric in ("collision_recall", "f1_50", "fp_min50", "severe_recall",
                   "mild_recall", "severe_minus_mild"):
        table = frame.pivot(index="subject", columns="method", values=metric)
        unit = 1 if metric == "fp_min50" else 100
        for control in METHODS[:-1]:
            row = contrast(unit * (table.v2 - table[control]))
            contrasts.append(dict(control=control, metric=metric, **row))
    pd.DataFrame(contrasts).to_csv(DEST / "transfer_contrasts.csv", index=False)
    frame.groupby("method")[["collision_recall", "f1_50", "fp_min50"]].mean().to_csv(
        DEST / "transfer_means.csv"
    )
    correlations = []
    for subject, subset in [("all", anchor), *anchor.groupby("subject")]:
        for name in METHODS:
            err = subset[name+"_abs_ms"]
            rho = spearmanr(subset.bound_ms, err).statistic if len(err) >= 3 else np.nan
            correlations.append(dict(subject=subject, method=name, beats=len(err), rho=rho))
    pd.DataFrame(correlations).to_csv(DEST / "transfer_bound_correlation.csv", index=False)


def main():
    seal = verify()
    (DEST / "synthetic_predictions").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(patient, range(32)))
    frame = pd.DataFrame(row for rows, _ in outcomes for row in rows)
    anchor = pd.concat([a for _, a in outcomes], ignore_index=True)
    summarize(frame, anchor)
    (DEST / "verified_model_hashes.json").write_text(json.dumps(seal, indent=2) + "\n")


if __name__ == "__main__":
    main()
