"""Prepare four-lead, native-rate records without supplying reference leads to inference.

Run python3 data_io.py to fetch both datasets and build data/records.json.
Use --ready to prepare only already-verified source files during initial transfer.
"""
import argparse
import io
import json
import re
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from download_data import (CLINICAL, DATA, PROV, SYN, FIG_MD5, INDEX_SHA256, download_clinical,
                           download_synthetic, file_hash, json_write, synthetic_files, data_lock)

PROCESSED = DATA / "processed"
LEADS = np.array([0, 10, 21, 31])
EMPTY = np.empty((0, 2), dtype=float)
IS_BEAT = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 30, 34, 35, 38, 41}


def wfdb_beats(path, fs):
    """Read standard MIT annotations, including signed SKIP and AUX padding."""
    data = path.read_bytes()
    position, sample = 0, 0
    beats = []
    while position < len(data):
        word = struct.unpack_from("<H", data, position)[0]
        position += 2
        if not word:
            break
        typ, delta = divmod(word, 1024)
        if typ == 59:
            high, low = struct.unpack_from("<hH", data, position)
            sample += (high << 16) + low
            position += 4
        elif typ == 63:
            aux = data[position:position + delta].decode("latin1")
            if aux.startswith("## time resolution:") and float(aux.split(":")[1]) != fs:
                raise ValueError(f"Inconsistent annotation clock in {path}.")
            position += delta + delta % 2
        elif typ < 59:
            sample += delta
            if typ in IS_BEAT:
                beats.append(sample)
        # SUB, CHN and NUM (60..62) only annotate the preceding beat.
    beats = np.array(beats, dtype=np.float64) / fs
    if len(beats) == 0 or np.any(np.diff(beats) <= 0):
        raise ValueError(f"Invalid WFDB beat series in {path}")
    return beats


def read_component(base):
    lines = base.with_suffix(".hea").read_text().splitlines()
    _, nchan, rate, length = lines[0].split()[:4]
    if (int(nchan), int(rate), int(length)) != (34, 250, 75000):
        raise ValueError(f"Unexpected FECGSYNDB header in {base}")
    data = np.memmap(base.with_suffix(".dat"), dtype="<i2", mode="r")
    raw = np.array(data.reshape(int(length), int(nchan))[:, LEADS], dtype=np.float32, order="C")
    for col, lead in enumerate(LEADS):
        description = lines[1 + lead].split()
        gain = re.fullmatch(r"([0-9.eE+\-]+)\(([-+]?\d+)\)/nu", description[2])
        if description[1] != "16" or gain is None:
            raise ValueError(f"Unsupported signal encoding in {base}")
        if np.any(raw[:, col] == -32768):
            raise ValueError(f"WFDB missing-sample sentinel found at {base} lead {lead + 1}")
        raw[:, col] = (raw[:, col] - int(gain[2])) / float(gain[1])
    return raw


def mask_unverified(times, verified, duration):
    """Midpoint partition of the recording assigned to each released annotation."""
    boundaries = np.r_[0.0, (times[1:] + times[:-1]) / 2, duration]
    intervals = []
    for i in np.flatnonzero(~verified):
        start, stop = boundaries[i:i + 2]
        if intervals and start == intervals[-1][1]:
            intervals[-1][1] = stop
        else:
            intervals.append([start, stop])
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def save_record(record_id, group, split, x, fs, fetus, mother, invalid=EMPTY, **extra):
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError(f"Malformed four-lead mixture {record_id}: {x.shape}")
    duration = len(x) / fs
    for label in (fetus, mother):
        if not len(label) or label.min() < 0 or label.max() >= duration or np.any(np.diff(label) <= 0):
            raise ValueError(f"Reference times outside signal or unordered for {record_id}")
    path = PROCESSED / (record_id + ".npz")
    tmp = path.with_suffix(".npz.part")
    with tmp.open("wb") as handle:
        np.savez(handle, x=np.ascontiguousarray(x, dtype=np.float32), fs=np.array(fs),
                 fetus=fetus, mother=mother, invalid=invalid, **extra)
    tmp.replace(path)
    return dict(id=record_id, group=group, split=split,
                path=path.relative_to(DATA.parent).as_posix(), fs=fs, samples=len(x),
                duration_s=duration, excluded_s=float(np.diff(invalid, axis=1).sum()),
                fetal_labels=len(fetus), maternal_labels=len(mother))


def import_synthetic(record, sources):
    base = SYN / record
    names = sorted(sources)
    signal = np.zeros((75000, 4), dtype=np.float32)
    for name in names:
        part = read_component(base.with_name(base.name + "_" + name))
        signal += part
        if name == "fecg1":
            fetal = part
    number = int(base.name[3:5])
    split = "train" if number <= 6 else ("validation" if number <= 8 else "test")
    fetus = wfdb_beats(base.with_name(base.name + "_fecg1.qrs"), 250)
    mother = wfdb_beats(base.with_name(base.name + "_mecg.qrs"), 250)
    # Source simulator can put the first QRS at sample -1, outside its 75,000-sample file.
    duration = len(signal) / 250
    fvalid = (fetus >= 0) & (fetus < duration)
    mvalid = (mother >= 0) & (mother < duration)
    row = save_record(base.name, "synthetic", split, signal, 250, fetus[fvalid], mother[mvalid],
                      fetal_component=fetal)
    row.update(pregnancy=number, sources=names, leads_1based=[1, 11, 22, 32],
               out_of_signal_fetal=int((~fvalid).sum()), out_of_signal_maternal=int((~mvalid).sum()))
    return row


def clinical_record(archive, group, number):
    group_code = group.upper()
    stage = "Pregnancy" if group == "b1" else "Labour"
    ref_id = f"{group_code}_{stage}_{number:02d}"
    prefix = f"Data Records/{group_code}_{stage}_dataset/{ref_id}/{group_code}_"
    def payload(label):
        return archive.read(prefix + f"{label}_{number:02d}.txt")
    # Only the first four columns are measured abdominal leads. The remaining
    # columns contain a supplied vendor-derived fetal waveform.
    raw = np.fromstring(payload("abSignals").replace(b",", b"."), sep=" ", dtype=np.float32)
    x = np.ascontiguousarray(raw.reshape(-1, 8)[:, :4])
    dur = len(x) / 500
    maternal_values = np.loadtxt(io.BytesIO(payload("Maternal_R")), ndmin=2, dtype=int)
    fetal_values = np.loadtxt(io.BytesIO(payload("Fetal_R")), ndmin=2, dtype=int)
    all_mother = maternal_values[:, 0].astype(float) / 500
    # B2_Labour_10 has one mark at exactly 300 s, after its final sampled instant.
    m_in_range = (all_mother >= 0) & (all_mother < dur)
    mother = all_mother[m_in_range]
    reference_fs = 500 if group == "b1" else 1000
    times = fetal_values[:, 0].astype(float) / reference_fs
    verified = np.ones(len(times), dtype=bool)
    if fetal_values.shape[1] == 2:
        if not np.isin(fetal_values[:, 1], [0, 1]).all():
            raise ValueError(f"Unexpected reliability flag in {ref_id}")
        verified = fetal_values[:, 1].astype(bool)
    elif group == "b1" or fetal_values.shape[1] != 1:
        raise ValueError(f"Unexpected annotation layout in {ref_id}")
    invalid = mask_unverified(times, verified, dur)
    row = save_record(ref_id, group, "test", x, 500, times[verified], mother, invalid,
                      fetal_all=times, fetal_verified=verified)
    row.update(reference_fs=reference_fs, excluded_fetal_labels=int((~verified).sum()),
               out_of_signal_maternal=int((~m_in_range).sum()),
               units="microvolts", leads_1based=[1, 2, 3, 4])
    return row


def prepare(ready=False):
    with data_lock("prepare"):
        _prepare(ready)


def _prepare(ready):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = DATA / "records_ready.json"
    previous = {row["id"]: row for row in json.loads(cache.read_text())} if cache.exists() else {}
    inventory_path = PROV / "processed_checksums.json"
    if not ready and inventory_path.is_file():
        for item in json.loads(inventory_path.read_text()):
            path = DATA.parent / item["path"]
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or file_hash(path) != item["sha256"]:
                previous.pop(path.stem, None)
    if CLINICAL.is_file():
        with zipfile.ZipFile(CLINICAL) as archive:
            for group, count, stage in (("b1", 10, "Pregnancy"), ("b2", 12, "Labour")):
                for number in range(1, count + 1):
                    key = f"{group.upper()}_{stage}_{number:02d}"
                    if key not in previous or not (PROCESSED / (key + ".npz")).is_file():
                        previous[key] = clinical_record(archive, group, number)
        json_write(cache, sorted(previous.values(), key=lambda r: r["id"]))
    if (PROV / "SHA256SUMS.txt").is_file() and (PROV / "RECORDS").is_file():
        sources = defaultdict(set)
        required = defaultdict(list)
        for path in synthetic_files():
            stem, _ = path.rsplit(".", 1)
            record, component = stem.rsplit("_", 1)
            sources[record].add(component)
            required[record].append(SYN / path)
        for record, names in sorted(sources.items()):
            key = Path(record).name
            if key in previous and (PROCESSED / (key + ".npz")).is_file():
                continue
            if ready and not all(p.exists() for p in required[record]):
                continue
            previous[key] = import_synthetic(record, names)
            json_write(cache, sorted(previous.values(), key=lambda r: r["id"]))
    rows = sorted(previous.values(), key=lambda r: r["id"])
    if not ready and len(rows) != 172:
        raise ValueError(f"Expected 172 records, obtained {len(rows)}.")
    json_write(cache, rows)
    if len(rows) == 172:
        inventory = []
        for row in rows:
            path = DATA.parent / row["path"]
            inventory.append(dict(path=row["path"], sha256=file_hash(path),
                                  size_bytes=path.stat().st_size))
        json_write(PROV / "processed_checksums.json", inventory)
        json_write(DATA / "records.json", rows)
    print(f"Prepared {len(rows)}/172 verified-source records.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="all", choices=["all"])
    parser.add_argument("--ready", action="store_true", help="Convert only files already present.")
    parser.add_argument("--offline", action="store_true", help="Use completely downloaded sources.")
    args = parser.parse_args()
    if not args.ready and not args.offline:
        download_clinical()
        download_synthetic()
    if args.offline:
        for name, digest in INDEX_SHA256.items():
            if file_hash(PROV / name) != digest:
                raise ValueError(f"Unrecognized FECGSYNDB source index {name}")
        if file_hash(CLINICAL, "md5") != FIG_MD5:
            raise ValueError("Clinical input archive does not match publisher MD5.")
        # Aliased WFDB records are hard links; hash their content only once per inode.
        verified_inodes = {}
        for name, digest in synthetic_files().items():
            path = SYN / name
            inode = path.stat().st_ino
            if inode not in verified_inodes:
                verified_inodes[inode] = file_hash(path)
            if verified_inodes[inode] != digest:
                raise ValueError(f"WFDB input checksum mismatch for {name}")
    prepare(args.ready)
    if not args.ready:
        from data_audit import main as inventory_archive
        inventory_archive()
