"""Development-selected alternative fetal QRS detector, four abdominal leads in.

All public call times are measured in seconds from acquired sample zero.
"""

import json
from pathlib import Path

import numpy as np
from v2.model import RATE, resample


LOCK = Path(__file__).with_name("selection_lock.json")


def candidate_anchors(x, fs, method):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError("Expected four finite abdominal channels (samples, channels)")
    if not np.isfinite(fs) or fs <= 0 or len(x) / fs < 5:
        raise ValueError("Expected positive acquisition frequency and at least five seconds")
    y = resample(x, fs)
    if method == "le":
        from other_refs import le

        times = le(y, RATE)
    elif method == "fm_fecg":
        from other_methods.neural import peaks
        import torch

        torch.set_num_threads(1)
        times = peaks(y, RATE, "cpu")
    else:
        raise ValueError(f"Unknown second-detector candidate {method}")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Detector generated invalid fetal times")
    return times[(times >= 0) & (times < len(x) / fs)]


def anchors(x, fs):
    """Return the previously selected second detector's calls, without annotations."""
    selection = json.loads(LOCK.read_text())
    return candidate_anchors(x, fs, selection["selected"])
