"""Seal, produce and verify the new V3 simulation cohort without running a detector."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from q1.theory import locked_transfer as gaussian
from v2 import simulation as hermite


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = (
    "v3/holdout/build.py",
    "v3/holdout/protocol.md",
    "v3/holdout/endpoint_amendment.md",
    "v3/holdout/history/pre_endpoint/design_lock.json",
    "v3/holdout/history/pre_endpoint/build.py",
    "v2/simulation.py",
    "q1/theory/geometry.py",
    "q1/theory/locked_transfer.py",
)
HERMITE_NAMESPACE = 2630911876
GAUSSIAN_NAMESPACE = 80742619


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    value = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != value:
            raise ValueError(f"Existing artifact differs from rebuilt file: {path}")
        return
    tmp = path.with_suffix(".part")
    tmp.write_text(value)
    tmp.replace(path)


def write_npz(path, **arrays):
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    checksum = digest(tmp)
    if path.exists():
        if digest(path) != checksum:
            raise ValueError(f"Existing recording differs: {path}")
        tmp.unlink()
    else:
        tmp.replace(path)
    return checksum


def design():
    roster = []
    for i in range(80):
        for family, namespace, index, stratum in [
            ("hermite", HERMITE_NAMESPACE, 1 + i, hermite.REGIMES[i % 4]),
            (
                "gaussian",
                GAUSSIAN_NAMESPACE,
                100000 + i,
                f"cos{(0, 55, 88, 98)[i % 4]:02d}_sd{(18, 40)[(i // 4) % 2]}",
            ),
        ]:
            key = f"v3_{family}_{1 + i:03d}"
            duration = hermite.DURATION if family == "hermite" else gaussian.DURATION
            roster.append(dict(
                id=key, subject=key, group="independent_synthetic", family=family,
                stratum=stratum, seed_index=index, seed_namespace=namespace,
                duration_s=duration, fs=500, maternal_reference="simulator_truth",
                signal=f"v3/holdout/signals/{key}.npz",
                reference=f"v3/holdout/references/{key}.npz",
            ))
    assert HERMITE_NAMESPACE not in {x[0] for x in hermite.NAMESPACES.values()}
    assert gaussian.FS == hermite.FS == 500
    assert len({row["subject"] for row in roster}) == 160
    historical = json.loads((HERE / "history/pre_endpoint/design_lock.json").read_text())
    assert roster == historical["roster"]
    for item in FILES:
        if item in historical["sources"]:
            old_source = "v3/holdout/history/pre_endpoint/build.py" if item == "v3/holdout/build.py" else item
            assert digest(ROOT / old_source) == historical["sources"][item]
    return dict(
        sources={name: digest(ROOT / name) for name in FILES},
        numpy=np.__version__, scipy=scipy.__version__, roster=roster,
    )


def verify_design():
    lock = json.loads((HERE / "design_lock.json").read_text())
    lock.pop("sealed_utc")
    if lock != design():
        raise ValueError("Generator, environment or roster differs from the design lock")
    return lock["roster"]


def make_record(row):
    i = row["seed_index"]
    if row["family"] == "hermite":
        arrays, params = hermite.make_case(row["seed_namespace"], i)
        if params["regime"] != row["stratum"]:
            raise ValueError("Hermite regime mismatch")
        x, f, m = arrays["x"], arrays["fetus"], arrays["mother"]
    else:
        x, f, m, _, _, _, cosine, noise = gaussian.simulate(i)
        if f"cos{round(cosine * 100):02d}_sd{round(noise * 100)}" != row["stratum"]:
            raise ValueError("Gaussian stratum mismatch")
    if x.shape != (row["fs"] * row["duration_s"], 4) or not np.isfinite(x).all():
        raise ValueError("Invalid four-channel observation")
    for events in (f, m):
        if (not len(events) or not np.isfinite(events).all() or
                not np.all(np.diff(events) > 0) or not (0 <= events[0] <= events[-1] <= row["duration_s"])):
            raise ValueError("Invalid reference stream")
    f, m = f[f < row["duration_s"]], m[m < row["duration_s"]]
    signal_hash = write_npz(ROOT / row["signal"], x=x.astype(np.float32), fs=np.int32(row["fs"]))
    ref_hash = write_npz(ROOT / row["reference"], fetus=f, mother=m, invalid=np.empty((0, 2)))
    return dict(row, signal_sha256=signal_hash, reference_sha256=ref_hash)


def build(workers):
    roster = verify_design()
    for name in ("signals", "references"):
        (HERE / name).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(make_record, roster))
    write_json(HERE / "records.json", rows)
    write_json(HERE / "manifest_lock.json", dict(
        design_sha256=digest(HERE / "design_lock.json"), records_sha256=digest(HERE / "records.json")
    ))
    verify()


def verify():
    roster = verify_design()
    paths = {key: HERE / path for key, path in (
        ("design_sha256", "design_lock.json"), ("records_sha256", "records.json"),
    )}
    seal = json.loads((HERE / "manifest_lock.json").read_text())
    if {key: digest(path) for key, path in paths.items()} != seal:
        raise ValueError("Cohort manifest checksum mismatch")
    rows = json.loads((HERE / "records.json").read_text())
    if len(rows) != len(roster):
        raise ValueError("Missing subject")
    counts = {"all": Counter(), "hermite": Counter(), "gaussian": Counter()}
    for planned, actual in zip(roster, rows):
        if planned != {key: actual[key] for key in planned}:
            raise ValueError("Subject no longer matches plan")
        for key in ("signal", "reference"):
            if digest(ROOT / actual[key]) != actual[f"{key}_sha256"]:
                raise ValueError(f"File checksum mismatch: {actual[key]}")
        with np.load(ROOT / actual["signal"], allow_pickle=False) as obs:
            if set(obs.files) != {"x", "fs"} or obs["x"].shape != (actual["duration_s"] * actual["fs"], 4):
                raise ValueError("Invalid observation file schema")
        with np.load(ROOT / actual["reference"], allow_pickle=False) as ref:
            if set(ref.files) != {"mother", "fetus", "invalid"} or ref["invalid"].shape != (0, 2):
                raise ValueError("Invalid reference file schema")
            m, f = ref["mother"], ref["fetus"]
            near = np.min(np.abs(m[:, None] - f[None, :]), axis=0)
            overlaps = int(np.count_nonzero(np.rint(near * 1e6) <= 40000))
            entry = dict(records=1, maternal_beats=len(m), fetal_beats=len(f),
                         overlaps=overlaps, seconds=actual["duration_s"])
        for group in ("all", actual["family"]):
            counts[group].update(entry)
    for key in ("signal_sha256", "reference_sha256"):
        if len({r[key] for r in rows}) != len(roster):
            raise ValueError(f"Duplicate {key}")
    write_json(HERE / "reference_counts.json", counts)
    print(json.dumps(counts, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["seal", "build", "verify"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "seal":
        if (HERE / "design_lock.json").exists():
            verify_design()
        else:
            write_json(HERE / "design_lock.json", dict(
                sealed_utc=datetime.now(timezone.utc).isoformat(), **design()
            ))
    elif args.stage == "build":
        build(args.workers)
    else:
        verify()


if __name__ == "__main__":
    main()
