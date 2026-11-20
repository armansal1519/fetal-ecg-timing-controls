"""Replay the frozen study model from public inputs, then score saved calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

from data_io import PROCESSED, clinical_record
from download_data import CLINICAL, FIG, download, file_hash
from scoring import keep_times, match, score_record
from v4.detect import predict as infer_signal

ROOT = Path(__file__).resolve().parent
COHORTS = ("poland", "simulation", "vietnam", "italy")
POLAND_HASH = "a7da9beefe1ce94f86ee99515848b8b853a44d1758cf30e7c7d87546536d77f6"


def manifest(cohort, supplied_zip):
    if cohort == "poland":
        path = supplied_zip or CLINICAL
        if supplied_zip is None:
            download(path, [FIG], POLAND_HASH)
        if file_hash(path) != POLAND_HASH:
            raise ValueError("Matonia archive fingerprint mismatch")
        PROCESSED.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            return [clinical_record(archive, "b2", i) for i in range(1, 13)]
    if cohort == "simulation":
        from v4.data.simulation import verify_design
        verify_design()
        return json.loads((ROOT / "v4/data/simulation_records.json").read_text())
    manifest_file = ROOT / ("v2/data_external/records.json" if cohort == "vietnam" else "q1/data/records.json")
    if not manifest_file.exists():
        steps = ([["v2/data_external/prepare.py"]] if cohort == "vietnam" else
                 [["-m", "q1.data.fetch_archive"], ["-m", "q1.data.fetch_italy"], ["-m", "q1.data.prepare_italy"]])
        for step in steps:
            subprocess.run([sys.executable, *step], cwd=ROOT, check=True)
    return json.loads(manifest_file.read_text())


def inputs(cohort, row):
    signal_path = ROOT / (row["signal"] if "signal" in row else row["path"])
    ref_path = ROOT / (row["reference"] if "reference" in row else row["path"])
    if cohort == "simulation" and not (signal_path.exists() and ref_path.exists()):
        from v3.holdout.build import make_record
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        make_record(row)
    expected_sig = row.get("signal_sha256", row.get("sha256"))
    expected_ref = row.get("reference_sha256", row.get("sha256"))
    if expected_sig and file_hash(signal_path) != expected_sig:
        raise ValueError(f"Waveform hash mismatch for {row['id']}")
    if expected_ref and file_hash(ref_path) != expected_ref:
        raise ValueError(f"Reference hash mismatch for {row['id']}")
    return signal_path, ref_path


def project(anchor, proposed):
    matched = match(anchor, proposed, 28)
    result = anchor.copy()
    result[matched[:, 0]] = proposed[matched[:, 1]]
    return np.sort(result)


def predict(row, signal_file, out):
    with np.load(signal_file, allow_pickle=False) as source:
        result, issues = infer_signal(source["x"], float(source["fs"]))
    anchor = result["power_mf"]
    streams = {name: result[name] for name in ("power_mf", "v4", "sts", "bank_cal", "bank_plain")}
    for method in ("v4", "bank_cal"):
        streams["paired_" + method] = project(anchor, result[method])
    for shift in (-4, -2, 0, 2, 4):
        streams[f"grid_{shift:+d}ms"] = anchor + shift / 1000
    destination = out / (row["id"] + ".npz")
    np.savez_compressed(destination, **streams)
    return dict(file=destination.name, sha256=file_hash(destination), issues=issues)


def site_shift(target, training):
    ids = sorted(set(training) - {target})
    return np.clip(np.median([training[i] for i in ids]), -28000, 28000) / 1e6


def summarize(rows, paths, seal, out, compute_site):
    training = {}
    if compute_site:
        for row in rows:
            rid = row["id"]
            with np.load(paths[rid][1]) as ref, np.load(out / seal[rid]["file"]) as pred:
                f = ref["fetus"][keep_times(ref["fetus"], ref["invalid"])]
                p = pred["power_mf"][keep_times(pred["power_mf"], ref["invalid"])]
                pairs = match(f, p, 50)
                training[rid] = np.median(np.rint(1e6*f[pairs[:, 0]]) - np.rint(1e6*p[pairs[:, 1]]))
    output = []
    for row in rows:
        rid = row["id"]
        file = out / seal[rid]["file"]
        if file_hash(file) != seal[rid]["sha256"]:
            raise ValueError("Predictions changed before scoring")
        with np.load(paths[rid][1], allow_pickle=False) as ref, np.load(file, allow_pickle=False) as source:
            streams = dict(source)
            if compute_site:
                streams["site_loo"] = streams["power_mf"] + site_shift(rid, training)
            for method, times in streams.items():
                times = times[(times >= 0) & (times < row["duration_s"])]
                scores = score_record(ref["fetus"], ref["mother"], ref["invalid"], times, row["duration_s"])[0]
                output.append(dict(record=rid, method=method, **scores))
    return pd.DataFrame(output)


def compare_archived(current, out):
    baseline = pd.read_csv(ROOT / "v4/results/records.csv")
    follow = pd.read_csv(ROOT / "clinical_followup/records.csv").query("scope == 'all' and method in ['paired_v4','paired_bank_cal']")
    site = pd.read_csv(ROOT / "manuscript_round/latency/site_records.csv")
    grid = site[site.method.str.startswith("grid_") | site.method.eq("site_loo")]
    expected = pd.concat([baseline, follow, grid], ignore_index=True)
    keys = ["record", "method"]
    metrics = ["n_ref", "n_collision", "collision_tp", "tp50", "fp50"]
    paired = current[keys + metrics].merge(expected[keys + metrics], on=keys, suffixes=("_new", "_archive"), validate="one_to_one")
    if paired.empty:
        raise ValueError("No common archived metrics to check")
    diff = pd.concat([paired[keys], pd.DataFrame({k:paired[k+"_new"] - paired[k+"_archive"] for k in metrics})], axis=1)
    diff.to_csv(out / "differences_from_archive.csv", index=False)
    count = diff[metrics].ne(0).any(axis=1).sum()
    print(f"{len(diff)} common method-record results, {count} with count differences. See {out/'differences_from_archive.csv'}.")
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", choices=COHORTS)
    p.add_argument("--record", help="A single public record ID for a shorter replay")
    p.add_argument("--matonia", type=Path, help="Use an already downloaded Polish archive")
    p.add_argument("--out", type=Path, required=True, help="New directory for signals' predictions, seal and scores")
    args = p.parse_args()
    if args.matonia and args.cohort != "poland":
        p.error("--matonia applies to the Polish cohort")
    rows = manifest(args.cohort, args.matonia)
    if args.record:
        rows = [r for r in rows if r["id"] == args.record]
        if not rows:
            p.error("Unknown record")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = {r["id"]: inputs(args.cohort, r) for r in rows}
    seal = {}
    for r in rows:
        print("Inferring", r["id"], flush=True)
        seal[r["id"]] = predict(r, paths[r["id"]][0], out)
    (out / "prediction-seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    scores = summarize(rows, paths, seal, out, args.cohort == "poland" and len(rows) == 12)
    scores.to_csv(out / "scores.csv", index=False)
    return int(compare_archived(scores, out) > 0)


if __name__ == "__main__":
    sys.exit(main())
