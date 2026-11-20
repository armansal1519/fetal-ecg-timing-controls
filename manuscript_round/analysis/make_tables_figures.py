"""Rebuild manuscript estimates, validation precision scenarios and vector figures."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
METRICS = ["collision_recall", "f1_50", "fp_min50"]
COHORTS = ["simulation", "historical_b2", "niferca", "italy"]
LABELS = ["Simulation\n320 pregnancies", "Poland, labour\n12 records", "Vietnam\n12 participants", "Italy\n13 participants"]
BLUE = "#005B86"
ORANGE = "#A84B06"
GREY = "#535C62"
SEED = 433719


def intervals(values, family=None):
    values = values.dropna()
    rng = np.random.default_rng(SEED)
    groups = [values] if family is None else [g for _, g in values.groupby(family)]
    total = np.zeros(10000, dtype=float)
    for subset in groups:
        arr = subset.to_numpy()
        total += arr[rng.integers(len(arr), size=(10000, len(arr)))].sum(axis=1)
    return np.r_[values.mean(), np.quantile(total / len(values), [0.025, 0.975])]


def source_rows(primary, follow):
    # Include the clinical paired streams exactly once.
    own = follow[(follow.scope == "all") & follow.method.isin(["paired_v4", "paired_bank_cal"])]
    return pd.concat([primary[primary.cohort.isin(COHORTS)], own[own.cohort.isin(COHORTS)]],
                     ignore_index=True)


def summaries(data):
    estimates, effects = [], []
    for cohort, grp in data.groupby("cohort"):
        patients = grp.groupby(["method", "subject"])[METRICS].mean()
        family = grp.groupby("subject").family.first() if cohort == "simulation" else None
        for method in ["power_mf", "v4", "bank_cal", "bank_plain", "v2", "fixed_count",
                       "paired_v4", "paired_bank_cal"]:
            group = grp[grp.method == method]
            if group.empty:
                continue
            unit = patients.loc[method]
            estimates.append(dict(cohort=cohort, method=method, records=len(group),
                                  participants=len(unit), participants_with_overlaps=unit.collision_recall.count(),
                                  overlaps=group.n_collision.sum(), tight_hits=group.collision_tp.sum(),
                                  valid_minutes=group.observed_s.sum()/60, false_positives=group.fp50.sum(),
                                  overlap_recall_pct=unit.collision_recall.mean()*100,
                                  f1_pct=unit.f1_50.mean()*100, fp_min=unit.fp_min50.mean()))
            controls = ["power_mf"] if method != "power_mf" else []
            if method == "v4":
                controls += ["bank_plain", "bank_cal"]
            if method == "paired_v4":
                controls += ["paired_bank_cal"]
            for control in controls:
                delta = unit - patients.loc[control]
                for metric in METRICS:
                    scale = 1 if metric == "fp_min50" else 100
                    point, lo, hi = intervals(scale * delta[metric], family)
                    effects.append(dict(cohort=cohort, test=method, control=control, metric=metric,
                                        n=delta[metric].count(), effect=point, lo95=lo, hi95=hi))
    return pd.DataFrame(estimates), pd.DataFrame(effects)


def validate(estimates, effects, official_files):
    ref = pd.read_csv(official_files["v4means"]).set_index(["cohort", "method"])
    fields = ["overlap_recall_pct", "f1_pct", "fp_min"]
    for row in estimates.itertuples(index=False):
        if (row.cohort, row.method) in ref.index:
            elem = ref.loc[row.cohort, row.method]
            np.testing.assert_allclose([getattr(row, f) for f in fields],
                                       elem[METRICS] * [100, 100, 1], rtol=0, atol=1e-9)
            assert row.tight_hits == elem.collision_hits and row.false_positives == elem.fp50
    oficial = pd.read_csv(official_files["v4effects"]).set_index(["cohort", "test", "control", "metric"])
    pairs = pd.read_csv(official_files["paired_effects"]).query('scope == "all"').set_index(["cohort","test","control"])
    ablation = pd.read_csv(official_files["ablation"]).set_index(["cohort","model","metric"])
    verified = 0
    for row in effects.itertuples(index=False):
        key = (row.cohort, row.test, row.control, row.metric)
        checks = []
        if key in oficial.index:
            checks.append(oficial.loc[key, ["difference","lo95","hi95"]])
        if key[:3] in pairs.index:
            checks.append(pairs.loc[key[:3], [row.metric, row.metric+"_lo95", row.metric+"_hi95"]])
        if row.control == "power_mf" and (row.cohort,row.test,row.metric) in ablation.index:
            checks.append(ablation.loc[(row.cohort,row.test,row.metric),["difference","lo95","hi95"]])
        for stored in checks:
            np.testing.assert_allclose([row.effect,row.lo95,row.hi95],stored.astype(float),rtol=0,atol=1e-9)
        verified += int(bool(checks))
    return verified


def save_fig(fig, stem):
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight", pad_inches=.055,
                metadata={"CreationDate":None, "ModDate":None})
    fig.savefig(OUT / (stem + ".png"), dpi=220, bbox_inches="tight", pad_inches=.055)
    plt.close(fig)


def cohort_fig(effects):
    grp = effects[(effects.control == "power_mf") & effects.test.isin(["v4", "paired_v4"])]
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.15), sharey=True)
    fig.subplots_adjust(left=.175, bottom=.15, right=.98, top=.79, wspace=.22)
    order = {name: 3-i for i,name in enumerate(COHORTS)}
    for ax, metric, title, label in zip(axs, METRICS,
                                        ["Overlap timing", "All-beat F1", "False positives"],
                                        ["Difference (pp)", "Difference (pp)", "Difference (calls/min)"]):
        ax.axvline(0, color="#7D878B", lw=.9)
        if metric == "collision_recall":
            ax.axvline(2, color="#627885", ls=(0,(2,3)), lw=.7)
        for item in grp[grp.metric==metric].itertuples(index=False):
            paired = item.test == "paired_v4"
            y = order[item.cohort] + (-.12 if paired else .12)
            ax.plot([item.lo95, item.hi95], [y, y], color=ORANGE if paired else BLUE, lw=1.4)
            ax.plot(item.effect, y, marker="D" if paired else "o", color=ORANGE if paired else BLUE,
                    markersize=4.4, linestyle="none")
        ax.set_xlabel(label)
        ax.set_title(title, fontsize=9, weight="bold", loc="left", pad=7)
        ax.set_ylim(-.45,3.46)
        ax.spines[["top","left","right"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.yaxis.grid(True, color="#DDE3E4", lw=.5)
        ax.set_axisbelow(True)
        ax.set_yticks([3,2,1,0],LABELS)
    legend = [Line2D([],[], color=BLUE, marker="o", lw=1.4),
              Line2D([],[], color=ORANGE, marker="D", lw=1.4)]
    fig.legend(legend, ["Full decoder", "Paired retiming"], frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(.55,1))
    save_fig(fig, "cohort_effects")


def polish_summary(data):
    group = data[data.cohort=="historical_b2"]
    rates = group.pivot(index="record", columns="method", values="collision_recall")*100
    rates = rates[["power_mf","paired_v4","paired_bank_cal","v4","v2","fixed_count"]].copy()
    rates["overlaps"] = group[group.method=="power_mf"].set_index("record").n_collision
    rates["paired_effect_pp"] = rates.paired_v4-rates.power_mf
    rates["leave_one_out_mean_pp"] = (rates.paired_effect_pp.sum()-rates.paired_effect_pp)/(len(rates)-1)
    return rates


def polish_fig(rates):
    fig, ax = plt.subplots(figsize=(5.8, 3.58))
    fig.subplots_adjust(left=.26,bottom=.15,right=.96,top=.87)
    n = len(rates)
    for y, (_, row) in enumerate(rates.iloc[::-1].iterrows()):
        ax.plot([row.power_mf,row.paired_v4], [y,y], c="#9AA4A8", lw=1.7)
        ax.plot(row.power_mf,y, marker="o", markerfacecolor="white", markeredgecolor=BLUE,
                markersize=5.9, zorder=3)
        ax.plot(row.paired_v4,y, marker="D", color=ORANGE, markersize=3.7, zorder=4)
    ax.set_yticks(range(n), [f"{ix.split('_')[-1]}    ({int(row.overlaps)})"
                           for ix,row in rates.iloc[::-1].iterrows()])
    ax.text(-.025,1.04,"Record (beats)",transform=ax.transAxes,ha="right",fontsize=8)
    ax.set_xlabel("Overlapping beats timed within ±10 ms (%)")
    ax.set_xlim(-2,102)
    ax.set_ylim(-.65,n-.35)
    ax.set_xticks(np.arange(0,101,20))
    ax.grid(axis="x", color="#DDE3E4", lw=.7)
    ax.spines[["left","top","right"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    legend=[Line2D([],[],ls="none",marker="o",markerfacecolor="white",markeredgecolor=BLUE),
            Line2D([],[],ls="none",marker="D",color=ORANGE,ms=5)]
    fig.legend(legend,["Power-MF","Paired retiming"],loc="upper center",ncol=2,frameon=False,
               bbox_to_anchor=(.62,1.0))
    save_fig(fig, "polish_records")


def precision_scenarios(rates):
    pilot_sd = rates.paired_effect_pp.std(ddof=1)
    rows=[]
    for factor in (.75, 1, 1.25):
        sd = factor*pilot_sd
        for half_width in (2, 3, 5):
            n = 2
            while student_t.ppf(.975,n-1)*sd/np.sqrt(n) > half_width:
                n += 1
            rows.append(dict(sd_factor=factor, assumed_sd_pp=sd,
                             target_half_width_pp=half_width, evaluable_units=n))
    return pd.DataFrame(rows)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8.4,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#8E989C",
                         "text.color":"#192F37","axes.labelcolor":"#192F37","xtick.color":"#263A40",
                         "ytick.color":"#263A40","pdf.fonttype":42,"svg.fonttype":"none"})
    paths=dict(primary=ROOT/"v4/results/records.csv",follow=ROOT/"clinical_followup/records.csv",
               v4means=ROOT/"v4/results/means.csv",v4effects=ROOT/"v4/results/contrasts.csv",
               paired_effects=ROOT/"clinical_followup/contrasts.csv",
               ablation=ROOT/"clinical_followup/frozen_ablation_vs_power.csv")
    data=source_rows(pd.read_csv(paths["primary"]),pd.read_csv(paths["follow"]))
    assert not data.duplicated(["cohort","record","method"]).any()
    estimates,effects=summaries(data)
    checked=validate(estimates,effects,paths)
    polish=polish_summary(data)
    estimates.to_csv(OUT/"method_outcomes.csv",index=False)
    effects.to_csv(OUT/"paired_effects.csv",index=False)
    polish.to_csv(OUT/"polish_records.csv")
    precision_scenarios(polish).to_csv(OUT/"validation_precision.csv",index=False)
    cohort_fig(effects)
    polish_fig(polish)
    manifest={"source_sha256":{str(v.relative_to(ROOT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in paths.values()},
              "contrasts_recomputed":len(effects),"contrasts_with_matching_published_estimates":checked,
              "maximum_matching_tolerance":1e-9,"bootstrap_resamples":10000,"seed":SEED}
    (OUT/"verification.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
