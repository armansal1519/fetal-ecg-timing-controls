# Fetal R-peak timing during maternal QRS overlap

Version 1.0.0 accompanies *Fixed time shifts and waveform-based fetal R-peak refinement during maternal QRS overlap*. This release lets a reader reconstruct study estimates and figures from recording-level statistics, replay the frozen four-lead detector on source waveforms, and inspect the reference clocks, candidate bank and signed-shift controls.

## Rebuild the paper results

Use Python 3.11 on macOS or Linux. The analysis was run with Python 3.11.15 and the versions in `requirements.txt`. From the extracted release directory:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --download-matonia
```

The last command verifies the release checksums, recomputes participant-level estimates, 10,000-draw paired intervals, pooled counts, residual-proposal rates and figures 1–5 plus the graphical abstract. The clinical example panel reads the four abdominal channels directly from Matonia's 195 MB CC0 archive. Use `python reproduce.py --matonia /path/to/matonia.zip` with an existing download. Running `python reproduce.py` works offline after Python dependencies have been installed and omits the raw-waveform example panel. Paper figures go into `reproduced/source/`, the graphical abstract into `reproduced/ancillary/`, and numeric results into `reproduced/*.csv` and `reproduced/results.md`. Results include native operational detector failures on the original valid exposure.

The verification compares 111 paired estimates against the archived table and 90 against earlier comparator output, covers all 12 Polish records and compares fixed shifts against recording-level counts. Simulated pregnancies are resampled within generator family. Clinical records are first averaged within participant when linkage is available. The pooled numerator/denominator weights beats equally and can order techniques differently from the mean across records. Positive changes favour recall and F1, negative changes favour false positives per minute. See `v4/protocol.md` for the original decision criterion and `manuscript_round/latency/site_protocol.md` for the retrospective calibration.

## Replay from original signals

Waveform inference additionally uses GNU Octave and its development headers with a C++ compiler. The Apple Silicon installer has a pinned local GNU Octave 10.3.0 runtime. On another platform install `octave-cli` and matching `mkoctfile` before setup. The classic front end downloads reference algorithms and dependencies from their original providers and verifies every pinned archive. It may reuse a local cache and outputs can vary with an eigensolver/BLAS implementation. The scorer writes count differences against the release for each tested method and record, returning failure if these differ.

```sh
python classic/setup.py
python -m unittest v4.test_model
python raw_replay.py poland --out raw-runs/poland
python raw_replay.py simulation --out raw-runs/simulation
python raw_replay.py vietnam --out raw-runs/vietnam
python raw_replay.py italy --out raw-runs/italy
```

Each invocation expects a new output directory. Add `--record B2_Labour_04` to the Polish invocation for a single-record smoke test. Poland accepts `--matonia /path/to/matonia.zip` and otherwise fetches the original. The simulation is regenerated from the versioned design and verified against signal/reference SHA-256 values. Its 320 virtual pregnancies are distinct from the FECGSYNDB development subjects. Vietnam and Italy obtain source files and create local four-lead recordings using the bundled importers. The Italian source linkage uses public ECG record and excerpt numbers and a thoracic maternal proxy. The supplied four-lead waveform and prediction streams use seconds since sample zero. Reference files supply eligible times and the two evaluation tolerances. Cohort runs first save predictions and their fingerprints, then score them. An operational model call on a waveform with arrays `x` (samples × 4) and `fs` (sampling rate in Hz) is `python -m v4.detect input.npz output.npz`.

Raw replay covers the proposed decoder, its two filters, both parent detectors, paired projections and simple time shifts. A full 12-record Polish replay also refits the eleven-record leave-one-out reference offset. Simulation fixed-count projections in the convenience replay are diagnostic outputs; the archived clinical follow-up did not evaluate paired projection in the simulation. The 14-method historical recording metrics are provided under `v4/results/`. Additional benchmark neural-network retraining, the full historical research workspace and its execution caches are outside this release. This package distributes the fitted V4 weights. Retraining on the released test outcomes would invalidate the study comparison.

## Inputs and attribution

| Source | Version and use |
|---|---|
| [Matonia et al.](https://doi.org/10.6084/m9.figshare.10311029.v1) | CC0 version 1. The ZIP SHA-256 and the member hashes are in `data/provenance/`. B2 scalp fetal marks are on a 1 kHz clock and abdominal/maternal samples are on a 500 Hz clock. Figure 5 uses two record-relative example locators on the public time base. |
| [Andreotti, Behar & Clifford](https://physionet.org/content/fecgsyndb/1.0.0/) | FECGSYNDB 1.0.0 under ODC Attribution 1.0. The development source manifest indexes every selected file, URL and SHA-256. Download using `python data_io.py` if the development inputs are needed. Cite [the generating framework](https://doi.org/10.1088/0967-3334/37/5/627) and PhysioNet. |
| [NIFERCA](https://doi.org/10.6084/m9.figshare.31912092.v1) | Figshare version 1. Pinned publisher MD5 and study SHA-256 appear in `data/provenance/niferca-archive.json`. The prepared input uses four prespecified abdominal channels and expert-reviewed abdominal fetal marks. |
| [NInFEA](https://physionet.org/content/ninfea/1.0.0/) and [Baldazzi et al.](https://doi.org/10.17632/64zy9f2dkf.1) | The versioned Italian inputs combine NInFEA 1.0.0 (ODC Attribution 1.0) and Mendeley version 1. The fetchers validate the waveform hashes and labelled ZIP hash. Use the original repositories for data terms and attribution. |
| [PhysioNet Challenge 2013](https://physionet.org/content/challenge-2013/1.0.0/) and [Tanasković & Miljković](https://zenodo.org/records/7848240) | The auxiliary historical comparison used ten anonymous clips. It remains descriptive. Source sets with unknown pregnancy identity must not be added to the clinical participant counts. |

No raw clinical traces, images, published manuscripts or participant annotation streams are distributed here. Figure 5 is regenerated from its original repository. Read `NOTICES.md` for source-specific terms. The archive contains published dataset record identifiers and derived counts. Human-subject study findings remain retrospective.
