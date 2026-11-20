"""Link labelled Cagliari snippets to the 24-lead recordings using chest ECG.

No fetal annotations, Doppler frames, model outputs or abdominal lead choices
enter the linkage. The signed offset places sample zero of the labelled
snippet at that sample of the host recording (both native 2,048 Hz).
"""
import argparse
import csv
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import signal


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
WAVES = RAW / "wfdb_format_ecg_and_respiration"
FS = 2048
STRIDE = 8
MIN_S = 4
MIN_R = 0.985


def source_zip():
    for path in [RAW / "italy_verified.zip", RAW / "italy.zip"]:
        if path.exists() and path.stat().st_size == 81344334:
            return path
    raise FileNotFoundError("Need the verified 2020 Mendeley ZIP")


def read_csv(bundle, series, idx, columns=None):
    name = f"{series}_{idx}.csv".lower()
    matches = [s for s in bundle.namelist() if Path(s).name.lower() == name]
    if len(matches) != 1:
        raise ValueError((name, matches))
    with bundle.open(matches[0]) as file:
        return np.loadtxt(file, delimiter=",", usecols=columns)


def wfdb_channels(record, columns):
    lines = (WAVES / f"{record}.hea").read_text().splitlines()
    _, channels, rate, samples = lines[0].split()[:4]
    if int(channels) != 34 or int(rate) != FS:
        raise ValueError("Unexpected NInFEA header")
    raw = np.memmap(WAVES / f"{record}.dat", dtype="<i4", mode="r").reshape(int(samples), int(channels))
    x = np.empty((len(raw), len(columns)))
    for i, ch in enumerate(columns):
        info = lines[ch+1].split()
        if info[1] != "32":
            raise ValueError("Not signed 32-bit interleaved WFDB")
        gain, base = re.fullmatch(r"([.0-9eE+-]+)\(([-\d]+)\)/uV", info[2]).groups()
        if np.any(raw[:,ch] == np.iinfo(np.int32).min):
            raise ValueError(f"WFDB missing sample in {record}, lead {ch+1}; audit before use")
        x[:, i] = (raw[:, ch].astype(float) - int(base)) / float(gain)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite physical waveform")
    return x


def band(x):
    sos = signal.butter(3, [8, 65], fs=FS, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def correlate_snippets(host, snippet):
    """Rank by correlations on all three high-passed maternal leads."""
    n, m = len(host), len(snippet)
    lags = signal.correlation_lags(n, m)
    begin = np.maximum(0, lags)
    end = np.minimum(n, lags + m)
    ok = end - begin >= MIN_S * FS // STRIDE
    lag, begin, end = lags[ok], begin[ok], end[ok]
    if not len(lag):
        return 0., 0
    score = np.ones(len(lag))
    for k in range(3):
        x, y = host[:, k], snippet[:, k]
        prod = signal.correlate(x, y, method="fft")[ok]
        xx = np.r_[0., np.cumsum(x*x)]
        yy = np.r_[0., np.cumsum(y*y)]
        denom = (xx[end]-xx[begin]) * (yy[end-lag]-yy[begin-lag])
        s = prod / np.sqrt(np.maximum(denom, 1.e-20))
        score = np.minimum(score, s)
    idx = int(np.argmax(score))
    return float(score[idx]), int(lag[idx]) * STRIDE


def sample_alignment(host, snippet, approx):
    scored = []
    # Exclude filter transients when verifying identical acquisition samples.
    for lag in range(approx-STRIDE, approx+STRIDE+1):
        start, stop = max(0, lag), min(len(host), len(snippet)+lag)
        margin = int(.20*FS)
        a = host[start+margin:stop-margin]
        b = snippet[start+margin-lag:stop-margin-lag]
        if len(a) < (MIN_S-.4)*FS-1:
            continue
        corr = (a*b).sum(axis=0) / np.sqrt((a*a).sum(axis=0)*(b*b).sum(axis=0))
        scored.append((float(min(corr)), lag, start, stop))
    return max(scored, default=(0., 0, 0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="Report acquisition progress on available hosts")
    partial = parser.parse_args().partial
    hosts = sorted(int(p.stem) for p in WAVES.glob("*.dat") if p.with_suffix(".hea").exists())
    if not partial and hosts != list(range(1, 61)):
        raise ValueError("Full release needed to finalize linkage")
    with ZipFile(source_zip()) as f:
        snippets = {i: band(read_csv(f, "Real_Traces", i, [0, 1, 2])) for i in range(1, 22)}
    accepted = []
    candidates = {i:[] for i in snippets}
    for n in hosts:
        host = band(wfdb_channels(n, [24,25,26]))
        for i, snippet in snippets.items():
            score, lag = correlate_snippets(host[::STRIDE], snippet[::STRIDE])
            candidates[i].append((score,n))
            if score < .90:
                continue
            exact, lag, start, stop = sample_alignment(host, snippet, lag)
            if exact >= MIN_R:
                accepted.append(dict(snippet=i, host=n, offset_samples=lag, start=start, stop=stop,
                                     min_chest_corr=round(exact,7)))
    if not partial:
        with (HERE / "linkage.csv").open("w",newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["snippet","host","offset_samples","start","stop","min_chest_corr"])
            writer.writeheader()
            writer.writerows(accepted)
        with (HERE / "linkage_rank.csv").open("w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["snippet","best_host","best_coarse_corr","second_host","second_coarse_corr","eligible_hosts"])
            for i, options in candidates.items():
                (first,host1),(second,host2) = sorted(options,reverse=True)[:2]
                valid = sorted(m["host"] for m in accepted if m["snippet"]==i)
                writer.writerow([i,host1,round(first,7),host2,round(second,7)," ".join(map(str,valid))])
    print(f"{len(hosts)} hosts; {len(accepted)} verified mappings")
    for row in accepted:
        print(row)


if __name__ == "__main__":
    main()
