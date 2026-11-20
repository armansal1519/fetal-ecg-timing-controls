"""Generate and verify an independent synthetic four-lead fetal ECG stress test.

One independently drawn virtual geometry, maternal rhythm and fetal rhythm per
recording. All amplitudes are in nominal microvolts. Arrays use time from sample
zero in seconds. No detector or historical subject waveform is imported.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import signal
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "v2" / "data" / "sim"
FS = 500
DURATION = 120
# The two namespaces and all sample sizes are fixed before any detector evaluation.
NAMESPACES = {"train": (35148701, 16), "test": (97401163, 24)}
REGIMES = ("stationary", "fetal_motion", "rate_modulation", "ventricular_ectopy")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unit(v):
    return v / np.linalg.norm(v)


def rhythms(rng, base_bpm, fast, regime):
    event = rng.uniform(0.12, 0.72)
    times, ectopic = [], []
    period, slow_period = rng.uniform(2.9, 5.2), rng.uniform(14, 33)
    phase = rng.uniform(0, 2 * np.pi, 2)
    respiratory = rng.uniform(0.006, 0.027) if fast else rng.uniform(0.018, 0.067)
    slow = rng.uniform(0.014, 0.047)
    transient_at, transient_width = rng.uniform(24, 96), rng.uniform(4, 12)
    transient = (
        rng.choice([-1, 1]) * rng.uniform(0.08, 0.20)
        if regime == "rate_modulation"
        else 0.0
    )
    ar = 0.0
    last_premature = False
    while event < DURATION:
        times.append(round(event * FS) / FS)
        ectopic.append(last_premature)
        ar = 0.82 * ar + rng.normal(0, 0.009 if fast else 0.012)
        variation = respiratory * np.sin(2 * np.pi * event / period + phase[0])
        variation += slow * np.sin(2 * np.pi * event / slow_period + phase[1]) + ar
        variation += transient * np.exp(
            -0.5 * ((event - transient_at) / transient_width) ** 2
        )
        interval = np.clip(
            60 / base_bpm * (1 + variation),
            0.27 if fast else 0.43,
            0.78 if fast else 1.35,
        )
        premature = False
        if regime == "ventricular_ectopy":
            if last_premature:
                interval *= 1.24
            elif rng.random() < (0.046 if fast else 0.014):
                premature = True
                interval *= 0.72
        event += interval
        last_premature = premature
    return np.array(times), np.array(ectopic, dtype=bool)


def dipole(rng, times, ectopic, fast):
    """PQRST landmark vectors interpolated with shape-preserving cubic polynomials."""
    q, s = rng.uniform(0.11, 0.32), rng.uniform(0.16, 0.51)
    p, t = rng.uniform(0.035, 0.13), rng.uniform(0.10, 0.28)
    # The first coordinate defines the latent reference-lead R apex at time zero.
    knots = np.array(
        [
            -0.235,
            -0.168,
            -0.104,
            -0.058,
            -0.025,
            -0.012,
            0,
            0.012,
            0.026,
            0.051,
            0.084,
            0.179,
            0.300,
            0.360,
        ]
    )
    knots *= rng.uniform(0.58, 0.77) if fast else rng.uniform(0.87, 1.13)
    shape = np.array(
        [
            [0, 0, 0],
            [p, 0.45 * p, -0.2 * p],
            [0, 0, 0],
            [0, 0, 0],
            [-q, 0.22, -0.14],
            [0.11, 0.28, 0.21],
            [1, 0, 0],
            [0.19, -0.25, 0.18],
            [-s, -0.32, -0.19],
            [0.015, -0.01, -0.01],
            [0.03, 0.02, 0.01],
            [t, -0.35 * t, 0.35 * t],
            [0.018, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    shape[:, 1:] *= rng.uniform(0.45, 1.25, (1, 2))
    normal = PchipInterpolator(knots, shape, axis=0, extrapolate=False)
    alternate = shape.copy()
    alternate[4:9] *= np.array([0.86, -1.10, 0.60])
    alternate[11] *= -0.62
    ect = PchipInterpolator(knots, alternate, axis=0, extrapolate=False)
    source = np.zeros((FS * DURATION, 3))
    resp_phase = rng.uniform(0, 2 * np.pi)
    resp_period = rng.uniform(3.2, 5.4)
    for instant, abnormal in zip(times, ectopic):
        width = rng.lognormal(0, 0.034) * (1.45 if abnormal else 1.0)
        amp = (
            1 + 0.07 * np.sin(2 * np.pi * instant / resp_period + resp_phase)
        ) * rng.lognormal(0, 0.052)
        lo = max(0, int(np.ceil((instant + width * knots[0]) * FS)))
        hi = min(len(source), int(np.floor((instant + width * knots[-1]) * FS)) + 1)
        where = np.clip((np.arange(lo, hi) / FS - instant) / width, knots[0], knots[-1])
        source[lo:hi] += amp * (ect if abnormal else normal)(where)
    return source


def projections(rng, xm, xf, regime, t):
    a = unit(rng.normal(size=4))
    trial = rng.normal(size=4)
    orth = unit(trial - np.dot(trial, a) * a)
    rho = rng.uniform(0.12, 0.96) * rng.choice([-1, 1])
    b = rho * a + np.sqrt(1 - rho * rho) * orth
    ma = np.column_stack((2 * a, rng.normal(0, 0.79, (4, 2))))
    fa = np.column_stack((2 * b, rng.normal(0, 0.79, (4, 2))))
    fetal_gain = rng.uniform(0.10, 0.30)
    maternal_gain = rng.uniform(255.0, 760.0)
    maternal = maternal_gain * (xm @ ma.T)
    fetal = (maternal_gain * fetal_gain) * (xf @ fa.T)
    drift = np.sin(2 * np.pi * t / rng.uniform(8.0, 17.0) + rng.uniform(0, 2 * np.pi))
    maternal += (
        0.04 * maternal_gain * drift[:, None] * (xm @ rng.normal(0, 0.6, (4, 3)).T)
    )
    motion = 0.08 * np.sin(
        2 * np.pi * t / rng.uniform(11.0, 27.0) + rng.uniform(0, 2 * np.pi)
    )
    if regime == "fetal_motion":
        motion += rng.uniform(0.45, 0.90) * np.tanh(
            (t - rng.uniform(25, 85)) / rng.uniform(1.8, 7.0)
        )
    fetal += (
        maternal_gain
        * fetal_gain
        * motion[:, None]
        * (xf @ rng.normal(0, 0.85, (4, 3)).T)
    )
    return maternal, fetal, maternal_gain, fetal_gain, float(rho)


def disturbance(rng, t, scale):
    n = len(t)

    def filtered(lo, hi):
        sos = signal.butter(3, [lo, hi], fs=FS, btype="bandpass", output="sos")
        y = signal.sosfiltfilt(sos, rng.normal(size=(n, 4)), axis=0)
        return y / np.std(y, axis=0)

    muscle = filtered(12, 86)
    envelope = np.full(n, rng.uniform(0.014, 0.055))
    for center in rng.uniform(0, DURATION, rng.integers(4, 10)):
        envelope += rng.uniform(0.035, 0.12) * np.exp(
            -0.5 * ((t - center) / rng.uniform(0.3, 1.9)) ** 2
        )
    noise = muscle * envelope[:, None] + rng.uniform(0.01, 0.032) * filtered(0.7, 13)
    noise += rng.uniform(0.026, 0.08) * filtered(0.06, 0.65)
    noise += rng.uniform(0.003, 0.012) * rng.normal(size=(n, 4))
    mains = rng.choice([50, 60])
    noise += rng.uniform(0.001, 0.012, (1, 4)) * np.sin(
        2 * np.pi * mains * t[:, None] + rng.uniform(0, 2 * np.pi, (1, 4))
    )
    # Short electrode transients with independent signed amplitude.
    for _ in range(rng.integers(2, 8)):
        j = rng.integers(0, 4)
        elapsed = t - rng.uniform(0, DURATION)
        keep = elapsed > 0
        noise[keep, j] += (
            rng.normal(0, 0.22)
            * (1 - np.exp(-elapsed[keep] / 0.015))
            * np.exp(-elapsed[keep] / 0.43)
        )
    return noise * scale


def make_case(namespace, index):
    geometry, maternal_rng, fetal_rng, noise_rng = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence([namespace, index]).spawn(4)
    ]
    regime = REGIMES[(index - 1) % len(REGIMES)]
    m_bpm = geometry.uniform(61, 98)
    f_bpm = geometry.uniform(116, 160)
    mother, me = rhythms(maternal_rng, m_bpm, False, regime)
    fetus, fe = rhythms(fetal_rng, f_bpm, True, regime)
    t = np.arange(FS * DURATION) / FS
    dm = dipole(maternal_rng, mother, me, False)
    df = dipole(fetal_rng, fetus, fe, True)
    maternal, fetal, scale, ratio, rho = projections(geometry, dm, df, regime, t)
    noise = disturbance(noise_rng, t, scale).astype("float32")
    maternal, fetal = maternal.astype("float32"), fetal.astype("float32")
    x = (maternal + fetal) + noise
    arrays = dict(
        x=x,
        fs=np.int32(FS),
        mother=mother,
        fetus=fetus,
        invalid=np.empty((0, 2)),
        fetal_component=fetal,
        maternal_component=maternal,
        noise_component=noise,
        maternal_ectopic=me,
        fetal_ectopic=fe,
    )
    parameters = dict(
        regime=regime,
        maternal_bpm=float(m_bpm),
        fetal_bpm=float(f_bpm),
        nominal_fetal_gain_ratio=ratio,
        primary_mixing_cosine=rho,
    )
    return arrays, parameters


def generate(dest=DEST):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (namespace, count) in NAMESPACES.items():
        seen = set()
        for index in range(1, count + 1):
            key = f"hermite_{split}_{index:02d}"
            path = dest / (key + ".npz")
            arrays, parameters = make_case(namespace, index)
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as stream:
                np.savez(stream, **arrays)
            tmp.replace(path)
            checksum = digest(path)
            assert checksum not in seen
            seen.add(checksum)
            rows.append(
                dict(
                    id=key,
                    subject=key,
                    group="independent_synthetic",
                    split=split,
                    path=path.relative_to(ROOT).as_posix(),
                    sha256=checksum,
                    seed_namespace=namespace,
                    seed_index=index,
                    **parameters,
                )
            )
    source_hash = digest(Path(__file__))
    spec = dict(
        generator="Independent cubic-Hermite vector cardiac stress test v1",
        generator_sha256=source_hash,
        frequency_hz=FS,
        duration_s=DURATION,
        split_counts={k: v[1] for k, v in NAMESPACES.items()},
        seed_namespaces={k: v[0] for k, v in NAMESPACES.items()},
        regimes=REGIMES,
        numpy=np.__version__,
        scipy=scipy.__version__,
        label="Generator R-wave apex in latent first dipole coordinate, on native sample grid.",
        scope="Separately seeded virtual subjects generated by this project. One record per virtual subject; no human participant data or FECGSYN waveforms used.",
    )
    for name, value in [("manifest.json", rows), ("protocol.json", spec)]:
        path = dest / name
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    verify(dest)


def verify(dest=DEST):
    rows = json.loads((dest / "manifest.json").read_text())
    spec = json.loads((dest / "protocol.json").read_text())
    assert digest(Path(__file__)) == spec["generator_sha256"]
    assert (
        len(set(r["subject"] for r in rows))
        == sum(v[1] for v in NAMESPACES.values())
        == len(rows)
    )
    assert len(set(r["sha256"] for r in rows)) == len(rows)
    for split, (_, count) in NAMESPACES.items():
        assert sum(r["split"] == split for r in rows) == count
    for row in rows:
        path = ROOT / row["path"]
        assert digest(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as d:
            assert d["x"].shape == (FS * DURATION, 4) and d["fs"].item() == FS
            assert np.isfinite(d["x"]).all()
            assert np.array_equal(
                d["x"],
                d["maternal_component"] + d["fetal_component"] + d["noise_component"],
            )
            for name in ("mother", "fetus"):
                times = d[name]
                assert 0 <= times[0] < times[-1] < DURATION and np.all(
                    np.diff(times) > 0
                )
                np.testing.assert_allclose(
                    times * FS, np.round(times * FS), atol=1e-9, rtol=0
                )
    print(
        f"Verified {len(rows)} unique simulations. Manifest SHA256 {digest(dest / 'manifest.json')}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check existing sources and hashes without generating.",
    )
    args = parser.parse_args()
    verify() if args.verify else generate()
