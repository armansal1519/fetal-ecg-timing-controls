"""Regenerate cohort outcomes and figures from the distributed record-level results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from download_data import CLINICAL, FIG, download, file_hash
from manuscript_round.analysis import make_tables_figures as analysis

ROOT = Path(__file__).resolve().parent


def verify_inputs():
    manifest = ROOT / "SHA256SUMS"
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", maxsplit=1)
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Release checksum mismatch: {name}")


def verify_frame(generated, target, identity):
    def ordered(frame):
        return frame.set_index(identity).sort_index().sort_index(axis=1).rename_axis(columns=None)
    pd.testing.assert_frame_equal(ordered(generated), ordered(pd.read_csv(target)), atol=1e-9, rtol=0,
                                  check_dtype=False)


def tables(out):
    v4 = ROOT / "v4/results"
    follow = ROOT / "clinical_followup"
    manuscript = ROOT / "manuscript_round"
    primary = pd.read_csv(v4 / "records.csv")
    pairs = pd.read_csv(follow / "records.csv")
    data = analysis.source_rows(primary, pairs)
    if data.duplicated(["cohort", "record", "method"]).any():
        raise ValueError("Duplicate recording-method result")
    means, effects = analysis.summaries(data)
    checks = dict(v4means=v4 / "means.csv", v4effects=v4 / "contrasts.csv",
                  paired_effects=follow / "contrasts.csv", ablation=follow / "frozen_ablation_vs_power.csv")
    checked = analysis.validate(means, effects, checks)
    verify_frame(means, manuscript / "analysis/method_outcomes.csv", ["cohort", "method"])
    verify_frame(effects, manuscript / "analysis/paired_effects.csv", ["cohort", "test", "control", "metric"])
    polish = analysis.polish_summary(data).reset_index()
    verify_frame(polish, manuscript / "analysis/polish_records.csv", ["record"])
    means.to_csv(out / "method_outcomes.csv", index=False)
    effects.to_csv(out / "paired_effects.csv", index=False)
    polish.to_csv(out / "polish_records.csv", index=False)
    site = pd.read_csv(manuscript / "latency/site_records.csv")
    controls = site.groupby("method")[analysis.METRICS].mean()
    controls[["collision_recall", "f1_50"]] *= 100
    controls["pooled_overlap_hits"] = site.groupby("method").collision_tp.sum()
    controls["pooled_fp50"] = site.groupby("method").fp50.sum()
    controls = controls.reset_index()
    verify_frame(controls, manuscript / "latency/site_means.csv", ["method"])
    controls.to_csv(out / "polish_latency.csv", index=False)

    reference = means.query("method == 'power_mf'").set_index("cohort")
    cohorts = reference[["participants", "records", "valid_minutes", "overlaps"]]
    cohorts.to_csv(out / "cohort_exposure.csv")
    residual = pd.read_csv(manuscript / "residual/records.csv")
    totals = residual.groupby("cohort")[["minutes", "overlaps", "holes", "recovered", "added", "new_unmatched50"]].sum()
    totals["added_per_min"] = totals.added / totals.minutes
    totals["unmatched50_per_min"] = totals.new_unmatched50 / totals.minutes
    original = pd.read_csv(manuscript / "residual/summary.csv").set_index("cohort")
    pd.testing.assert_frame_equal(totals, original[totals.columns].sort_index(), rtol=1e-12, atol=1e-9, check_dtype=False)
    totals.to_csv(out / "residual_candidates.csv")

    series = effects.set_index(["cohort", "test", "control", "metric"])
    sim = series.loc["simulation", "v4", "power_mf", "collision_recall"]
    filter_effect = series.loc["simulation", "v4", "bank_cal", "collision_recall"]
    report = [
        "# Fetal R-peak timing results", "",
        f"Paired, family-stratified simulation gain: {sim.effect:+.2f} pp (95% CI {sim.lo95:.2f} to {sim.hi95:.2f}).",
        f"Identical candidate bank, V4 minus calibrated filter: {filter_effect.effect:+.2f} pp ({filter_effect.lo95:.2f} to {filter_effect.hi95:.2f}).",
        "", "## Polish labour", "", "| Method | Mean recording recall (%) | Pooled hits / overlaps | FP / 60 min |",
        "|---|---:|---:|---:|"]
    for method in ["power_mf", "paired_v4", "paired_bank_cal", "grid_-2ms", "site_loo"]:
        row = controls.set_index("method").loc[method]
        report.append(f"| {method} | {row.collision_recall:.2f} | {int(row.pooled_overlap_hits)} / {int(cohorts.loc['historical_b2','overlaps'])} | {int(row.pooled_fp50)} |")
    report += ["", "site_loo uses labels from the other eleven records at the same centre.", "",
               f"{len(effects)} paired effects reproduced; {checked} also agree with archived comparator tables.",
               "See paired_effects.csv for every cohort and metric. Uncertainty uses 10,000 paired cluster draws."]
    (out / "results.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reproduced")
    raw = parser.add_mutually_exclusive_group()
    raw.add_argument("--download-matonia", action="store_true", help="Download the original CC0 archive for Figure 5")
    raw.add_argument("--matonia", type=Path, help="An existing original Figshare ZIP for Figure 5")
    args = parser.parse_args()
    verify_inputs()
    out = args.out.resolve()
    out.mkdir(exist_ok=True, parents=True)
    tables(out)
    archive = args.matonia
    if args.download_matonia:
        expected = json.loads((ROOT / "data/provenance/clinical_download.json").read_text())["sha256"]
        download(CLINICAL, [FIG], expected)
        archive = CLINICAL
    if archive is not None:
        expected = json.loads((ROOT / "data/provenance/clinical_download.json").read_text())["sha256"]
        if file_hash(archive) != expected:
            raise ValueError("The archive for Figure 5 differs from the published input")
    figures = ["1", "2", "3", "4", "ga"]
    command = [sys.executable, str(ROOT / "render_figures.py"), "--root", str(ROOT), "--output", str(out)]
    if archive is not None:
        figures.insert(4, "5")
        command += ["--matonia", str(archive.resolve())]
    subprocess.run(command + ["--figures", *figures], check=True)
    print("Wrote", out)


if __name__ == "__main__":
    main()
