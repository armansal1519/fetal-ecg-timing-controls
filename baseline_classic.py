"""Python interface for the original Varanini/Behar code and Power-MF paper detector.

Input shape (samples, 4); output seconds relative to the first acquired sample.
Shared maternal cancellation and ICA are cached by the content of the signal.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import io, signal

ROOT = Path(__file__).resolve().parent / "classic"
CACHE = ROOT / "cache"


def _key(x, fs):
    h = hashlib.sha256()
    h.update(str(float(fs)).encode())
    h.update(str(x.shape).encode())
    h.update(memoryview(np.ascontiguousarray(x)).cast("B"))
    for path in [
        Path(__file__),
        ROOT / "run_classic.m",
        ROOT / "setup.py",
        ROOT / "prepare_overlays.py",
        ROOT / "qrs_scan.cc",
        ROOT / "runtime-osx-arm64.lock",
        ROOT / "compat/qrs_scan.oct",
        ROOT / "compat/medfilt1.oct",
    ]:
        h.update(path.read_bytes())
    return h.hexdigest()


def _run(method, source, destination):
    octave = ROOT / "runtime/bin/octave-cli"
    if not octave.exists():
        found = shutil.which("octave-cli")
        if not found:
            raise RuntimeError(
                "Prepare the reference implementation with python3 classic/setup.py"
            )
        octave = Path(found)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        OCTAVE_HISTFILE=os.devnull,
    )
    if octave.parent == ROOT / "runtime" / "bin":
        env.update(
            OCTAVE_HOME=str(octave.parent.parent),
            OCTAVE_EXEC_HOME=str(octave.parent.parent),
        )
    command = [
        str(octave),
        "--no-gui",
        "--no-init-file",
        "--quiet",
        str(ROOT / "run_classic.m"),
        method,
        str(source),
        str(destination),
    ]
    proc = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
        check=False,
    )
    log = destination.parent / (method + ".log")
    log.write_text(proc.stdout)
    flags = [
        line
        for line in proc.stdout.splitlines()
        if "REFERENCE_CATCH" in line
        or "just guessing" in line
        or "There was an error!" in line
    ]
    if proc.returncode or not destination.is_file() or (flags and method != "behar"):
        message = "\n".join(flags[:4]) or "\n".join(proc.stdout.splitlines()[-12:])
        raise RuntimeError(
            f"{method} failed; {destination.parent.parent / log.name}\n{message}"
        )


def _inputs(x, fs):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all() or fs <= 0:
        raise ValueError(
            "Expected a finite four-channel signal and positive sampling rate"
        )
    # Original filtering includes a 150 Hz low pass in some records. A 250 Hz
    # synthetic input cannot represent that cutoff. Resample the synthetic input
    # to the native clinical common rate, retaining time zero.
    if fs == 250:
        return signal.resample_poly(x, 2, 1, axis=0), 500.0
    return x, float(fs)


def _segments(x):
    split = len(x) // 2
    return [(0, x[:split]), (split, x[split:])] if len(x) > 500000 else [(0, x)]


def _materialize(method, x, fs):
    key = _key(x, fs)
    folder = CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (method + ".mat")
    if target.exists():
        return target
    # Atomic materialization permits independent worker processes to share this
    # cache even if the same input arrives twice. Each worker has a private cwd.
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        work = Path(scratch)
        if method == "varanini":
            source = _materialize("pre", x, fs)
        else:
            source = work / "input.mat"
            io.savemat(source, {"x": x, "fs": fs})
        output = work / "output.mat"
        try:
            _run(method, source, output)
        finally:
            log = work / (method + ".log")
            if log.exists():
                log.replace(folder / log.name)
        output.replace(target)
    return target


def _power_mf(components, fs):
    """Jaeger PowerMF.m after residual ICA, preserving the indexed zero template rows."""
    standardized = (components - components.mean(axis=0)) / components.std(
        axis=0, ddof=1
    )
    nu = int(np.ceil(0.005 * fs))
    nz = int(np.floor(0.003 * fs / 2)) * 2 + 1
    b = np.r_[np.ones(nu), np.zeros(nz), -np.ones(nu)]
    delay = len(b) // 2
    padded = np.pad(standardized, ((delay, delay), (0, 0)), mode="edge")
    derivative = signal.lfilter(b, [1], padded, axis=0)[2 * delay :]
    b, a = signal.butter(1, [0.7, 8], btype="bandpass", fs=fs)
    envelope = signal.filtfilt(
        b, a, np.abs(derivative), axis=0, padlen=3 * (max(len(a), len(b)) - 1)
    )
    win_len = min(
        2 ** int(np.ceil(np.log2(round(fs * 15 * (60 / 110))))), len(envelope) // 2
    )
    window = signal.windows.gaussian(win_len, (win_len - 1) / (2 * 2.5))
    freq, power = signal.welch(
        envelope - envelope.mean(axis=0),
        fs=fs,
        window=window,
        nperseg=win_len,
        noverlap=win_len // 2,
        nfft=win_len,
        detrend=False,
        axis=0,
    )
    band = power[(freq > 1.8) & (freq < 3)]

    def strongest(column):
        local = signal.find_peaks(column)[0]
        return max(column[local], default=0.0)

    channel = int(np.argmax([strongest(col) for col in band.T]))
    distance = 0.340 * fs
    peaks = signal.find_peaks(envelope[:, channel], distance=distance)[0]
    if len(peaks) < 5:
        raise ValueError("Too few preliminary Power-MF peaks")
    half = int(np.floor(np.median(np.diff(peaks)) / 2 + 0.5))
    templates = np.zeros((len(peaks) - 2, 2 * half + 1))
    for j in range(2, len(peaks) - 2):
        templates[j] = components[peaks[j] - half : peaks[j] + half + 1, channel]
    template = np.median(templates, axis=0)
    # MATLAB matched_filter crops starting at index w, with w=floor(L/2).
    conv = signal.oaconvolve(components[:, channel], template[::-1])
    filtered = conv[half - 1 : half - 1 + len(components)]
    return signal.find_peaks(filtered, distance=distance)[0] / fs


def fetal_components(x, fs):
    x, fs = _inputs(x, fs)
    segments = [
        io.loadmat(_materialize("pre", chunk, fs))["components"]
        for _, chunk in _segments(x)
    ]
    return np.concatenate(segments) if len(segments) > 1 else segments[0], 4 * fs


def predict(method, x, fs):
    """Seconds on a physical time origin, excluding the historical B1 join offset."""
    if method not in {"behar", "varanini", "power_mf"}:
        raise ValueError(f"Unknown classical ECG baseline {method!r}")
    x, fs = _inputs(x, fs)
    result = []
    for start, chunk in _segments(x):
        if method == "power_mf":
            data = io.loadmat(_materialize("pre", chunk, fs))
            secs = _power_mf(data["components"], float(data["fs"].item()))
        else:
            peaks = io.loadmat(_materialize(method, chunk, fs))["peaks"].reshape(-1)
            secs = (peaks - 1) / (fs if method == "behar" else 4 * fs)
        # An inner Behar error branch emits a guessed series with fractional
        # indices. Exclude physically out-of-record calls and retain the error log.
        secs = secs[(secs >= 0) & (secs < len(chunk) / fs)]
        result.append(secs + start / fs)
    return np.concatenate(result)


def reference_events(method, x, fs):
    """Original caught errors and fallback diagnostics by input segment."""
    x, fs = _inputs(x, fs)
    logs = []
    for start, chunk in _segments(x):
        stages = (
            ["pre"]
            if method == "power_mf"
            else ["pre", "varanini"]
            if method == "varanini"
            else ["behar"]
        )
        for stage in stages:
            path = CACHE / _key(chunk, fs) / (stage + ".log")
            if path.is_file():
                events = [
                    line.strip()
                    for line in path.read_text().splitlines()
                    if any(
                        token in line
                        for token in [
                            "REFERENCE_CATCH",
                            "just guessing",
                            "There was an error!",
                            "Convergence failure",
                        ]
                    )
                ]
                if events:
                    logs.append(
                        {"start_s": start / fs, "stage": stage, "events": events}
                    )
    return logs
