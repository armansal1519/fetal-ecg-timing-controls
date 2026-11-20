"""One-to-one fetal peak matching and recording-level statistics. Times in seconds."""

import numpy as np


def match(ref_s, pred_s, tolerance_ms):
    """Ordered maximum cardinality, then minimum sum absolute errors via a sparse DP."""
    ref = np.rint(np.asarray(ref_s) * 1e6).astype(np.int64)
    pred = np.rint(np.asarray(pred_s) * 1e6).astype(np.int64)
    if np.any(np.diff(ref) < 0) or np.any(np.diff(pred) < 0):
        raise ValueError("Call streams must be sorted")
    radius = round(tolerance_ms * 1000)
    nodes = [(0, 0, -1, -1, 0)]  # count, negative error, ref, prediction, predecessor
    tree = [0] * (len(pred) + 1)
    left = np.searchsorted(pred, ref - radius)
    right = np.searchsorted(pred, ref + radius, side="right")

    def better(a, b):
        return a if nodes[a][:2] >= nodes[b][:2] else b

    def query(end):
        best = 0
        while end:
            best = better(best, tree[end])
            end -= end & -end
        return best

    for i, (a, b) in enumerate(zip(left, right)):
        new = []
        for j in range(a, b):
            prev = query(j)
            count, neg, *_ = nodes[prev]
            nodes.append((count + 1, neg - abs(int(ref[i] - pred[j])), i, j, prev))
            new.append((j + 1, len(nodes) - 1))
        for k, v in new:
            while k < len(tree):
                tree[k] = better(tree[k], v)
                k += k & -k
    end = query(len(pred))
    pairs = []
    while end:
        _, _, i, j, end = nodes[end]
        pairs.append((i, j))
    return np.array(pairs[::-1], int).reshape(-1, 2)


def keep_times(times, invalid):
    t = np.asarray(times)
    good = np.ones(len(t), bool)
    for a, b in np.asarray(invalid).reshape(-1, 2):
        good &= (t < a) | (t >= b)
    return good


def score_record(ref, mother, invalid, pred, duration):
    from detector import nearest_distance

    ref = np.asarray(ref)
    pred = np.asarray(pred)
    keep = keep_times(pred, invalid)
    p = pred[keep]
    r = ref[keep_times(ref, invalid)]
    intervals = np.asarray(invalid).reshape(-1, 2)
    observed = duration - np.sum(
        np.maximum(
            0, np.minimum(duration, intervals[:, 1]) - np.maximum(0, intervals[:, 0])
        )
    )
    if observed <= 0:
        raise ValueError("record has no valid time")
    lag = np.rint(nearest_distance(r, np.asarray(mother)) * 1e6)
    stats = dict(
        n_ref=len(r),
        n_pred=len(p),
        n_collision=int(np.sum(lag <= 40000)),
        masked_s=duration - observed,
        observed_s=observed,
    )
    bins = []
    for tol in (10, 50):
        pairs = match(r, p, tol)
        tp = len(pairs)
        fn = len(r) - tp
        fp = len(p) - tp
        stats.update(
            {
                f"tp{tol}": tp,
                f"fn{tol}": fn,
                f"fp{tol}": fp,
                f"f1_{tol}": 2 * tp / max(1, len(r) + len(p)),
                f"precision{tol}": tp / max(1, len(p)),
                f"recall{tol}": tp / max(1, len(r)),
                f"fp_min{tol}": 60 * fp / observed,
            }
        )
        if tol == 50:
            err = abs(r[pairs[:, 0]] - p[pairs[:, 1]]) * 1000
            stats["timing_median_ms"] = np.median(err) if tp else np.nan
            stats["timing_p90_ms"] = np.quantile(err, 0.9) if tp else np.nan
        else:
            hit = np.zeros(len(r), bool)
            hit[pairs[:, 0]] = True
            good_prediction = np.zeros(len(p), int)
            good_prediction[pairs[:, 1]] = 1
            overlap = lag <= 40000
            stats["collision_tp"] = int(hit[overlap].sum())
            stats["collision_recall"] = hit[overlap].mean() if overlap.any() else np.nan
            stats["noncollision_recall"] = (
                hit[~overlap].mean() if (~overlap).any() else np.nan
            )
            for name, mask in [
                ("0–20", lag <= 20000),
                ("20–40", (lag > 20000) & (lag <= 40000)),
                ("40–80", (lag > 40000) & (lag <= 80000)),
                (">80", lag > 80000),
            ]:
                bins.append(
                    dict(
                        lag_band_ms=name,
                        reference=int(mask.sum()),
                        matched=int(hit[mask].sum()),
                    )
                )
    return stats, bins, keep, good_prediction


def paired_interval(delta, seed=20261119):
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    draws = np.mean(d[rng.integers(len(d), size=(10000, len(d)))], axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return np.mean(d), lo, hi


def checks():
    import itertools

    rng = np.random.default_rng(18)
    for _ in range(80):
        a = np.sort(rng.choice(80, 5, replace=False)) / 1000.0
        b = np.sort(rng.choice(80, 5, replace=False)) / 1000.0
        best = (0, 0)
        for k in range(1, 6):
            for ii in itertools.combinations(range(len(a)), k):
                for jj in itertools.combinations(range(len(b)), k):
                    errs = np.rint(abs(a[list(ii)] - b[list(jj)]) * 1e6).astype(int)
                    if all(errs <= 10000):
                        best = max(best, (k, -errs.sum()))
        pairs = match(a, b, 10)
        assert (
            len(pairs),
            -int(round(np.sum(abs(a[pairs[:, 0]] - b[pairs[:, 1]])) * 1e6)),
        ) == best
    assert len(match([1.0], [1.010], 10)) == 1 and not len(match([1.0], [1.012], 10))
    assert len(match([1.0], [0.990], 10)) == 1 and not len(match([1.0], [0.988], 10))
    assert keep_times([1.0, 1.2, 2.0], [[1, 2]]).tolist() == [False, False, True]
    # A fetal reference at the same physical time on 500 and 1000 Hz clocks.
    assert len(match([1251 / 1000], [625 / 500], 10)) == 1
    assert len(match([1.0], [252 / 250], 10)) == 1 and not len(
        match([1.0], [253 / 250], 10)
    )
    print("Exact matcher, tolerance boundaries, annotation clocks and masks passed.")


if __name__ == "__main__":
    checks()
