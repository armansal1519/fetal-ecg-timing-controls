"""Render the CMPB analytical figures from archived estimates and the original signal."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BLUE, RUST, TEAL = "#195B83", "#A44815", "#00675F"
INK, GREY, GRID = "#1D3038", "#526369", "#DAE2E4"
PALE, WARM, COOL = "#F3F6F7", "#FCF5ED", "#EDF5F4"
MATONIA_SHA = "a7da9beefe1ce94f86ee99515848b8b853a44d1758cf30e7c7d87546536d77f6"
STYLE = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.8,
         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#88979A",
         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
META = {"CreationDate": None, "ModDate": None, "Creator": "Matplotlib scientific figures"}


def save(fig, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, metadata=META)
    plt.close(fig)


def schematic():
    fig = plt.figure(figsize=(7.15, 4.55))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.plot([.5, .5], [.03, .974], color=GRID, lw=.9)

    def block(x, y, width, height, title, detail=None, fill=PALE, border=GRID):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=0.0025,rounding_size=0.009",
                     fc=fill, ec=border, lw=.75))
        cx, cy = x + width / 2, y + height / 2
        ax.text(cx, cy + (.022 if detail else 0), title, ha="center", va="center",
                weight="bold", size=8.6)
        if detail:
            ax.text(cx, cy - .023, detail, ha="center", va="center", size=7.7,
                    color=GREY, linespacing=1.12)

    def arrow(x, top, bottom):
        ax.annotate("", xy=(x, bottom), xytext=(x, top),
                    arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": .92,
                                "shrinkA": 0, "shrinkB": 1, "mutation_scale": 8})

    def fork(xmid, xleft, xright, ytop, ymid, ybottom, down):
        if down:
            ax.plot([xmid, xmid], [ytop, ymid], color=GREY, lw=.9)
        else:
            ax.plot([xleft, xleft], [ytop, ymid], color=GREY, lw=.9)
            ax.plot([xright, xright], [ytop, ymid], color=GREY, lw=.9)
        ax.plot([xleft, xright], [ymid, ymid], color=GREY, lw=.9)
        for x in ([xleft, xright] if down else [xmid]):
            arrow(x, ymid, ybottom)

    ax.text(.018,.952,"A   Generating a beat sequence",size=11.1,weight="bold", va="center")
    ax.text(.515,.952,"B   Testing the source of timing gain",size=11.1,weight="bold", va="center")
    block(.028,.815,.44,.091,"Four abdominal ECG leads")
    fork(.248,.139,.357,.815,.785,.758,True)
    block(.028,.668,.218,.089,"Power-MF")
    block(.253,.668,.215,.089,"STS + ICA")
    fork(.248,.139,.357,.667,.641,.613,False)
    block(.028,.494,.44,.116,"Shared candidate times",
          "Raw calls + template maxima within ±28 ms")
    arrow(.248,.491,.429)
    block(.028,.302,.44,.124,"Cross-lead waveform evidence",
          "Fit amplitude on three leads and score the fourth", COOL, "#A0C3BF")
    arrow(.248,.3,.243)
    block(.028,.124,.44,.116,"Sequence selection",
          "Three-class score + RR-constrained decoding", COOL, "#A0C3BF")
    ax.text(.248,.066,"Fetal beat times with edits to count and timing",
            size=7.4,ha="center",color=GREY)
    block(.525,.778,.449,.128,"Same candidate times and decoder",
          "Compare the waveform score with plain and\ncalibrated matched-filter scores")
    arrow(.75,.775,.719)
    block(.525,.575,.449,.142,"Keep the Power-MF beat count",
          "Pair within ±28 ms, move each matched call\nand retain every unpaired native call",COOL,"#A0C3BF")
    ax.text(.531,.515,"Compare the resulting times with",size=8.4,weight="bold")
    block(.525,.363,.449,.118,"Common shifts of all Power-MF calls",
          "Sensitivity grid, including the B1-trained offset",WARM,"#D6BFA8")
    block(.525,.213,.449,.118,"Same-site calibration",
          "Estimate a shift from the other 11 B2 records",WARM,"#D6BFA8")
    ax.plot([.525,.975],[.17,.17],color=GRID,lw=.8)
    ax.text(.53,.13,"Held-out electrical references enter scoring",weight="bold",size=8.35)
    ax.text(.53,.068,"Overlap ≤40 ms  •  timing ≤10 ms\nAll-beat F1 and false positives at 50 ms",
            color=GREY,size=8.05,linespacing=1.3,va="center")
    return fig


def cohort_figure(table):
    cohorts = ["simulation", "historical_b2", "niferca", "italy"]
    labels = ["Simulation\n320 pregnancies", "Poland\n12 labour records",
              "Vietnam\n12 participants", "Italy\n13 participants"]
    metrics = ["collision_recall", "f1_50", "fp_min50"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.93), sharey=True)
    fig.subplots_adjust(left=.185, right=.98, bottom=.19, top=.78, wspace=.23)
    data = table[(table.control == "power_mf") & table.test.isin(["v4", "paired_v4"])]
    assert len(data) == 21 and not data.duplicated(["cohort","test","metric"]).any()
    for ax, metric, title, label in zip(axs, metrics, ["Overlap recall", "All-beat F1", "False positives"],
                                        ["Change (pp)", "Change (pp)", "Change (calls/min)"]):
        ax.axvline(0, color="#7C8B8E", lw=.85)
        if metric == "collision_recall":
            ax.axvline(2, color=GREY, ls=(0,(2,2)), lw=.7)
        for row in data[data.metric == metric].itertuples():
            projected = row.test == "paired_v4"
            y = 3 - cohorts.index(row.cohort) + (-.115 if projected else .115)
            color = RUST if projected else BLUE
            ax.hlines(y, row.lo95, row.hi95, color=color, lw=1.3, zorder=2)
            ax.plot(row.effect, y, "D" if projected else "o", ms=4.2, color=color)
        ax.set_yticks([3,2,1,0],labels)
        ax.set_ylim(-.43, 3.42)
        ax.tick_params(axis="y",length=0)
        ax.tick_params(axis="x",length=2.8)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True,color=GRID,lw=.5)
        ax.set_axisbelow(True)
        ax.set_title(title,loc="left",weight="bold",size=9.4)
        ax.set_xlabel(label)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker="o",label="Full decoder",ms=4.5),
                        Line2D([],[],color=RUST,marker="D",label="Count-preserving retiming",ms=4.5)],
               loc="upper center",bbox_to_anchor=(.55,1.00),ncol=2,frameon=False)
    return fig


def polish_figure(rows):
    fig, ax = plt.subplots(figsize=(6.5,3.37))
    fig.subplots_adjust(left=.18,right=.95,bottom=.155,top=.845)
    assert len(rows)==12 and rows.overlaps.sum()==899
    for pos, row in enumerate(rows.iloc[::-1].itertuples()):
        ax.plot([row.power_mf,row.paired_v4],[pos,pos],color="#9AA6A9",lw=1.65)
        ax.plot(row.power_mf,pos,"o",mfc="white",mec=BLUE,ms=6.2,zorder=3)
        ax.plot(row.paired_v4,pos,"D",color=RUST,ms=3.9,zorder=4)
    ax.set_yticks(range(len(rows)),[f"{r.record[-2:]}    ({r.overlaps})" for r in rows.iloc[::-1].itertuples()])
    ax.grid(axis="x",color=GRID,lw=.6)
    ax.spines["left"].set_visible(False)
    ax.set(xlim=(-2,102),ylim=(-.6,11.6),xlabel="Overlapping fetal beats timed within ±10 ms (%)")
    ax.text(-.013,1.035,"Record  (beats)",transform=ax.transAxes,ha="right",size=8)
    ax.tick_params(axis="y",length=0)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker="o",mfc="white",ls="none",label="Power-MF"),
                        Line2D([],[],color=RUST,marker="D",ms=4.4,ls="none",label="Count-preserving retiming")],
               ncol=2,loc="upper center",bbox_to_anchor=(.61,.994),frameon=False)
    return fig


def latency_figure(means, rows, transfers, fits):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.22))
    fig.subplots_adjust(left=.085,right=.98,top=.79,bottom=.19,wspace=.33)
    shifts=np.array([-4,-2,0,2,4])
    curve=means.loc[[f"grid_{s:+}ms" for s in shifts],"collision_recall"]
    ax.plot(shifts,curve,color=GREY,marker="o",ms=4.4,lw=1.35)
    for method,color,style in [("paired_v4",RUST,":"),("paired_bank_cal",BLUE,"--")]:
        ax.axhline(means.loc[method,"collision_recall"],color=color,ls=style,lw=1.05)
    ax.set(xlim=(-4.55,4.55), ylim=(36,99),xticks=shifts,yticks=[40,50,60,70,80,90],
           xlabel="Shift applied to Power-MF (ms)",ylabel="Mean overlap recall (%)")
    fit=transfers["offset_us"] / 1000
    ax.scatter([fit],[curve.iloc[np.flatnonzero(shifts==fit)[0]]],marker="s",color=TEAL,s=26,zorder=3)
    ax.annotate(f"B1 fit ({fit:+g} ms)", (fit,curve.loc[f"grid_{int(fit):+}ms"]),
                xytext=(-5,-13),textcoords="offset points",ha="right",color=TEAL,size=7.2)
    ax.text(4.4,82.0,"Calibrated filter",size=7.2,ha="right",color=BLUE)
    ax.text(4.4,75.1,"Waveform",size=7.2,ha="right",color=RUST)
    ax.set_title("A   Shift sensitivity",size=10,weight="bold",loc="left",pad=9)
    ax.annotate("Native", (0, curve.loc["grid_+0ms"]), xytext=(4,4),textcoords="offset points",size=7.25)
    delta=rows.site_minus_v4_pp
    x=np.arange(1,len(rows)+1)
    np.testing.assert_allclose(rows.applied_shift_ms,fits.loc[rows.index,"applied_us"]/1000)
    for record, fit_row in fits.iterrows():
        assert set(fit_row.training_ids.split()) == set(rows.index) - {record}
    bx.axhline(0,color=GREY,lw=.8)
    bx.axhline(delta.mean(),color=TEAL,lw=.8,ls=(0,(3,2)))
    bx.vlines(x,0,delta,color="#A4B1B4",lw=1)
    for shift,marker,fc in [(-4.5,"o",TEAL),(-9,"s","white")]:
        mask=rows.applied_shift_ms==shift
        bx.scatter(x[mask],delta[mask],s=20,marker=marker,edgecolor=TEAL,facecolor=fc,
                   label=f"{shift:g} ms",zorder=3,linewidths=.9)
    bx.set(xlim=(.5,12.5),ylim=(-13,84),xticks=x,yticks=[0,20,40,60,80],
           xlabel="Held-out B2 recording",ylabel="Site shift − waveform (pp)")
    bx.text(12.3,10.6,f"Mean +{delta.mean():.1f} pp",color=TEAL,ha="right",size=7.3)
    bx.annotate(f"+{delta.max():.1f} pp",(x[np.argmax(delta)],delta.max()),
                xytext=(4,3),textcoords="offset points",size=7.3,color=TEAL)
    bx.set_title("B   Fit to 11 other B2 records",size=9.7,weight="bold",loc="left",pad=9)
    bx.legend(loc="upper right",frameon=False,fontsize=7.25,ncol=2,handlelength=.9,
              columnspacing=.8,borderaxespad=.18)
    bx.tick_params(axis="x",labelsize=7.6)
    for axis in (ax,bx):
        axis.grid(axis="y",lw=.5,color=GRID)
        axis.tick_params(length=2.4)
        axis.set_axisbelow(True)
    return fig


def source_waveforms(archive, record):
    """Read exactly the four measured channels and the two reference clocks."""
    idx = record[-2:]
    prefix = f"Data Records/B2_Labour_dataset/{record}/B2_"
    def payload(name):
        return archive.read(prefix + f"{name}_{idx}.txt")
    data = np.fromstring(payload("abSignals").replace(b",",b"."),sep=" ",dtype=np.float32)
    fetal = np.loadtxt(io.BytesIO(payload("Fetal_R")),ndmin=2,dtype=int)[:,0]
    maternal = np.loadtxt(io.BytesIO(payload("Maternal_R")),ndmin=2,dtype=int)[:,0]
    return data.reshape(-1,8)[:,:4], fetal, maternal


def examples_figure(examples, path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != MATONIA_SHA:
        raise ValueError("Matonia v1 source checksum mismatch")
    styles = [("Fetal scalp R",BLUE,"-"),("Maternal R",GREY,(0,(1,2))),
              ("Power-MF",RUST,(0,(5,2))),("Waveform retiming",TEAL,(0,(4,2,1,2)))]
    fig, axs=plt.subplots(4,2,figsize=(7.15,3.9),sharex="col")
    fig.subplots_adjust(left=.093,right=.986,top=.728,bottom=.13,hspace=.205,wspace=.24)
    assert list(examples.status)==["gained","lost"]
    with zipfile.ZipFile(path) as archive:
        data={name:source_waveforms(archive,name) for name in examples.record.unique()}
    for col,case in enumerate(examples.itertuples()):
        signal,fetal,maternal=data[case.record]
        t0=int(round(case.fetal_s * 1e6))
        assert t0//1000 in fetal and round(case.maternal_s*500) in maternal
        errors=np.array([0,round(case.maternal_s*1e6)-t0,
                         round(case.power_s*1e6)-t0,round(case.refined_s*1e6)-t0])/1000
        np.testing.assert_allclose(errors[2:],[case.native_error_ms,case.refined_error_ms],atol=1e-9)
        assert abs(errors[1])<=40 and (abs(errors[2])<=10) == (col==1)
        assert (abs(errors[3])<=10) == (col==0)
        idx=np.arange(int(np.ceil((t0-95000)/2000)),int(np.floor((t0+95000)/2000))+1)
        for lead,axis in enumerate(axs[:,col]):
            axis.axvspan(-10,10,color="#E1EEF3",zorder=0)
            for delta,(_,color,ls) in zip(errors,styles):
                axis.axvline(delta,color=color,ls=ls,lw=1.12,zorder=1)
            axis.plot(idx*2 - t0/1000,signal[idx,lead],color="#253238",lw=1.02,zorder=2)
            axis.set_xlim(-95,95)
            axis.set_xticks([-75,-50,-25,0,25,50,75])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=2,min_n_ticks=2))
            axis.tick_params(labelsize=7,length=2,pad=2)
            if lead<3:
                axis.tick_params(bottom=False)
                axis.spines["bottom"].set_visible(False)
            if col==0:
                axis.text(-.165,.48,f"L{lead+1}",transform=axis.transAxes,size=8.2,
                          ha="right",va="center",weight="bold")
        axs[-1,col].set_xlabel("Time from fetal scalp R (ms)",size=8.1)
        left=axs[0,col].get_position().x0
        fig.text(left,.862,["A   Strict match gained","B   Strict match lost"][col],
                 size=10.3,weight="bold")
        fig.text(left,.811,f"B2 record {case.record[-2:]}   •   fetal R at {case.fetal_s:.3f} s",size=8.0)
        fig.text(left,.763,f"Error (ms)    Power-MF {errors[2]:+.1f}    Retimed {errors[3]:+.1f}",size=7.9)
    fig.legend(handles=[Line2D([],[],color=c,ls=d,lw=1.4,label=l) for l,c,d in styles],frameon=False,
               loc="upper center",bbox_to_anchor=(.52,1.003),handlelength=2.7,ncol=4,columnspacing=1.65)
    fig.text(.093,.014,"Four abdominal leads in µV. Shading marks the inclusive ±10 ms timing window.",size=7.6)
    return fig


def graphical_abstract(means):
    """An ordinary vector data plot with values read directly from the clinical means."""
    fig=plt.figure(figsize=(7.5,3.0))
    fig.text(.039,.892,"One sampling interval changed fetal R-peak timing recall",
             fontsize=14.3,weight="bold",color=INK)
    fig.text(.04,.799,"Four abdominal ECG leads  •  12 Polish labour recordings  •  899 overlapping beats",
             size=9.25,color=GREY)
    ax=fig.add_axes([.345,.25,.423,.432])
    labels=["Native Power-MF","Waveform retiming","Power-MF shifted −2 ms"]
    methods=["power_mf","paired_v4","grid_-2ms"]
    colors=[GREY,RUST,BLUE]
    fig.text(.783,.706,"False positives",ha="left",size=9.5,weight="bold",color=INK)
    fig.text(.783,.644,"in 60 minutes",ha="left",size=8.5,color=GREY)
    ax.axis("off")
    ax.set_xlim(0,100)
    ax.set_ylim(-.5,2.5)
    for i,(label,method,color) in enumerate(zip(labels,methods,colors)):
        y=2-i
        ypos=.25 + .432*(y+.5)/3
        rate=means.loc[method,"collision_recall"]
        fig.text(.039,ypos,label,size=9.65,va="center",weight="bold")
        ax.plot([0,rate],[y,y],lw=9.0,color=color,solid_capstyle="butt")
        ax.text(rate+1.6,y,f"{rate:.2f}%",size=9.1,color=color,va="center",weight="bold")
        fig.text(.827,ypos,str(int(means.loc[method,"pooled_fp50"])),ha="center",
                 va="center",size=11,weight="bold",color=INK)
    fig.text(.345,.705,"Mean recall within ±10 ms",size=9.5,weight="bold",color=INK)
    fig.text(.04,.119,"Overlap = fetal reference within 40 ms of a maternal mark. Recall is averaged across recordings.",
             fontsize=8.0,color=GREY)
    fig.text(.04,.054,"The common shift moves every detection one 500 Hz sample earlier.",fontsize=8.0,color=GREY)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd(),help="Study or replication directory with manuscript_round/")
    parser.add_argument("--output",type=Path,required=True,help="Parent of source/ and ancillary/ outputs")
    parser.add_argument("--matonia",type=Path,help="Official Matonia v1 ZIP for the raw waveforms in Figure 5")
    parser.add_argument("--figures",nargs="+",choices=["1","2","3","4","5","ga"],
                        default=["1","2","3","4","5","ga"])
    args=parser.parse_args()
    plt.rcParams.update(STYLE)
    folder=args.root/"manuscript_round"
    analysis=folder/"analysis"
    latency=folder/"latency"
    for key in args.figures:
        if key=="1":
            fig=schematic()
        elif key=="2":
            fig=cohort_figure(pd.read_csv(analysis/"paired_effects.csv"))
        elif key=="3":
            fig=polish_figure(pd.read_csv(analysis/"polish_records.csv"))
        elif key=="4":
            fig=latency_figure(pd.read_csv(latency/"site_means.csv",index_col="method"),
                               pd.read_csv(analysis/"latency_record_differences.csv",index_col=0),
                               json.loads((latency/"fit.json").read_text()),
                               pd.read_csv(latency/"site_fits.csv",index_col="omitted_record"))
        elif key=="5":
            fig=examples_figure(pd.read_csv(folder/"examples.csv"),
                                args.matonia or args.root/"data/raw/clinical/matonia.zip")
        elif key=="ga":
            fig=graphical_abstract(pd.read_csv(latency/"site_means.csv",index_col="method"))
        path=args.output/("ancillary/graphical-abstract.pdf" if key=="ga" else f"source/fig{key}.pdf")
        save(fig,path)
        print(path)


if __name__=="__main__":
    main()
