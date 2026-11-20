"""Prepare the NIFERCA v1 external validation set, using the raw abdominal leads."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from download import RAW, SIZE, MD5, digest, download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "processed"
FS = 500
SAMPLES = 600000
LEADS = np.array([1, 4, 7, 10]) - 1
# Acquisition offsets in the Figshare v1 README, in seconds.
TRIM = (2.1, 2.1, 1.9, 2.21, 3.15, 2.59, 2.83, 2.46, 3.016, 3.2, 3.0, 2.8)
WINDOWS_S = [120, 360, 600, 840, 1080]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


def variable(bundle, case, name):
    member = f"NIFERCA_DATASET/r{case}/{name}_{case}.mat"
    content = bundle.read(member)
    obj = loadmat(
        io.BytesIO(content), variable_names=["QRS" if name.endswith("QRS") else name]
    )
    return obj["QRS" if name.endswith("QRS") else name], {
        "path": member,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def verify_alignment(raw, filtered, sample_offset):
    """Compare abdominal signals only, on independent windows throughout the recording."""
    half = 15  # search +/-30 ms beyond the documented trim offset
    sos = butter(3, [8, 40], btype="bandpass", fs=FS, output="sos")
    lag_axis = np.arange(-half, half + 1)
    lags, correlations, at_zero = [], [], []
    for t in WINDOWS_S:
        start = FS * t
        stop = start + 20 * FS
        wave = sosfiltfilt(
            sos,
            raw[:, sample_offset + start - half : sample_offset + stop + half],
            axis=1,
        )[:, FS:-FS]
        template = sosfiltfilt(sos, filtered[:, start:stop], axis=1)[:, FS:-FS]
        template -= template.mean(axis=1, keepdims=True)
        candidates = []
        for shift in lag_axis:
            v = wave[:, half + shift : half + shift + template.shape[1]]
            v = v - v.mean(axis=1, keepdims=True)
            norm = np.sqrt(np.sum(template**2, axis=1) * np.sum(v**2, axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.sum(template * v, axis=1) / norm
            candidates.append(np.nanmedian(corr))
        idx = np.argmax(candidates)
        lags.append(int(lag_axis[idx]))
        correlations.append(float(candidates[idx]))
        at_zero.append(float(candidates[half]))
    # Channel-level artifacts may be common, but the robust consensus lag must
    # be consistent throughout the record to identify the acquisition origin.
    consensus = int(np.median(lags))
    if np.max(np.abs(np.asarray(lags) - consensus)) > 1 or abs(consensus) >= half:
        raise ValueError(f"Inconsistent raw/filtered acquisition alignment: {lags}")
    return sample_offset + consensus, dict(
        reference="Publisher-filtered abdominal ECG; no beat labels or detector outputs",
        filter="zero-phase third-order Butterworth 8-40 Hz",
        window_starts_s=WINDOWS_S,
        window_s=20,
        aggregate="median Pearson correlation of 12 leads",
        search_radius_samples=half,
        relative_lags_samples=lags,
        peak_correlation=correlations,
        declared_lag_correlation=at_zero,
        median_relative_lag_samples=consensus,
        documented_raw_offset_samples=sample_offset,
        applied_raw_offset_samples=sample_offset + consensus,
    )


def beat_times(a):
    if a.ndim != 2 or a.shape[0] != 2 or a.dtype.kind not in "iuf":
        raise ValueError("Expected the publisher's 2-by-N QRS matrix")
    indices, flags = a
    if (
        np.any(indices != np.rint(indices))
        or np.any(np.diff(indices.astype(float)) <= 0)
        or indices.min() < 1
        or indices.max() > SAMPLES
        or not np.all(np.isin(flags, [0, 1]))
    ):
        raise ValueError("Unexpected indexing, ordering, flags, or study interval")
    return (indices.astype(float) - 1) / FS, flags.astype(np.uint8)


def flag_intervals(times, flags):
    edges = np.r_[0, (times[1:] + times[:-1]) / 2, SAMPLES / FS]
    return [(float(edges[i]), float(edges[i + 1])) for i in np.flatnonzero(flags == 0)]


def merge_ranges(pairs):
    merged = []
    for a, b in sorted(pairs):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def prepare_case(bundle, case):
    raw, raw_meta = variable(bundle, case, "aECG_raw")
    filtered, filtered_meta = variable(bundle, case, "aECG_filtered")
    if filtered.shape != (12, SAMPLES) or raw.shape[0] != 12:
        raise ValueError(f"Unexpected dimensions for r{case}")
    declared = round(FS * TRIM[case - 1])
    start, audit = verify_alignment(raw, filtered, declared)
    if start + SAMPLES > raw.shape[1]:
        raise ValueError("Inconsistent acquisition length")
    x = np.ascontiguousarray(raw[LEADS, start : start + SAMPLES].T, dtype=np.float32)
    if not np.isfinite(x).all() or np.any(np.ptp(x, axis=0) == 0):
        raise ValueError(f"Invalid four-lead signal r{case}")
    f, f_meta = variable(bundle, case, "fQRS")
    m, m_meta = variable(bundle, case, "mQRS")
    fetus, ff = beat_times(f)
    mother, mf = beat_times(m)
    masked = merge_ranges(flag_intervals(fetus, ff) + flag_intervals(mother, mf))
    record = f"NIFERCA_r{case:02d}"
    path = OUTPUT / f"{record}.npz"
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as fd:
        np.savez(
            fd,
            x=x,
            fs=np.array(FS),
            fetus=fetus,
            mother=mother,
            invalid=np.empty((0, 2), float),
            invalid_flag1=masked,
            fetus_flag1=fetus[ff == 1],
            mother_flag1=mother[mf == 1],
            fetal_flags=ff,
            maternal_flags=mf,
        )
    os.replace(tmp, path)
    details = dict(
        id=record,
        subject=record,
        group="niferca",
        split="test",
        path=str(path.relative_to(ROOT)),
        sha256=digest(path),
        fs=FS,
        samples=len(x),
        duration_s=len(x) / FS,
        excluded_s=0,
        fetal_labels=len(fetus),
        maternal_labels=len(mother),
        fetal_flags_0=int(np.sum(ff == 0)),
        maternal_flags_0=int(np.sum(mf == 0)),
        flag1_mask_seconds=float(np.diff(masked, axis=1).sum()),
        original_leads_1based=(LEADS + 1).tolist(),
        first_fetal_sample=int(f[0, 0]),
        last_fetal_sample=int(f[0, -1]),
        max_fetal_gap_s=float(np.max(np.diff(fetus))),
        first_maternal_sample=int(m[0, 0]),
        last_maternal_sample=int(m[0, -1]),
        max_maternal_gap_s=float(np.max(np.diff(mother))),
    )
    return details, dict(
        id=record, **audit, source_files=[raw_meta, filtered_meta, f_meta, m_meta]
    )


def prepare(archive):
    if archive.stat().st_size != SIZE or digest(archive, "md5") != MD5:
        raise ValueError("The full archive must match the publisher's size and MD5")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    with zipfile.ZipFile(archive) as bundle:
        readme = bundle.read("NIFERCA_DATASET/readme.txt").decode("utf-8-sig")
        (HERE / "release_readme.txt").write_text(readme)
        all_offsets = []
        for match in re.finditer(
            r"^\s*(\d+)\s+([\d.]+)\s*-\s*([\d.]+)\s+", readme, flags=re.M
        ):
            all_offsets.append((int(match[1]), float(match[2]), float(match[3])))
        if len(all_offsets) != 12 or any(
            n != i or start != TRIM[i - 1] or abs(stop - start - 1200) > 1e-6
            for i, (n, start, stop) in enumerate(all_offsets, 1)
        ):
            raise ValueError("Acquisition offsets do not match original README")
        for i in range(1, 13):
            info, clock = prepare_case(bundle, i)
            rows.append(info)
            timing.append(clock)
            print(
                f"Prepared {info['id']}: acquisition alignment residual in samples "
                f"{clock['relative_lags_samples']}",
                flush=True,
            )
    write_json(
        HERE / "clock_audit.json",
        {
            "archive": str(archive.relative_to(ROOT)),
            "publisher_md5": MD5,
            "sha256": digest(archive),
            "sampling_hz": FS,
            "convention": "MATLAB QRS sample 1 denotes time zero of trimmed signal",
            "flag_policy": "All released marks in primary; flag-1-only secondary, masking flag-0 midpoint cells for both sources",
            "per_record": timing,
        },
    )
    write_json(HERE / "records.json", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    prepare(RAW / "NIFERCA_DATASET.zip" if args.offline else download())
