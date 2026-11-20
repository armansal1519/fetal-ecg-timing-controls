"""Fetch the public, fixed Matonia and FECGSYNDB releases into data/raw."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
PROV = DATA / "provenance"
SYN = DATA / "raw" / "synthetic"
CLINICAL = DATA / "raw" / "clinical" / "matonia.zip"
PHYS = "https://physionet.org/files/fecgsyndb/1.0.0/"
MIRROR = "https://physionet-open.s3.amazonaws.com/fecgsyndb/1.0.0/"
FIGMETA = "https://api.figshare.com/v2/articles/10311029/versions/1"
FIG = "https://ndownloader.figshare.com/files/18736784"
FIG_MD5 = "7a8c5402c9e100ef1b9a3f63f4b7b039"
INDEX_SHA256 = {
    "SHA256SUMS.txt": "81ebd5ec9f45546977809916877ef9ed59f0fb30f1cd5b9494443391a255e413",
    "RECORDS": "ab0e0e8f0a982af2539d0c97f011189b75931be031754e51a7115b4075f79748",
}
SELECT = re.compile(r"sub(0[1-9]|10)/snr(00|06|12)dB/sub\1_snr\2dB_l1_c[0-4]_(fecg1|mecg|noise[1-3])\.(dat|hea|qrs)$")


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def download(path, urls, expected=None, algorithm="sha256", end_byte=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (not expected or file_hash(path, algorithm) == expected):
        return
    partial = path.with_suffix(path.suffix + ".part")
    if expected and partial.is_file() and file_hash(partial, algorithm) == expected:
        partial.replace(path)
        return
    for attempt in range(5):
        range_args = ["--continue-at", "-"] if end_byte is None else ["--range", f"0-{end_byte}"]
        command = ["curl", "--fail", "--silent", "--show-error", "--location",
                   "--connect-timeout", "15", "--max-time", "450", "--retry", "1",
                   *range_args, urls[attempt % len(urls)], "--output", str(partial)]
        status = subprocess.run(command, capture_output=True)
        if status.returncode == 0 and (not expected or file_hash(partial, algorithm) == expected):
            partial.replace(path)
            return
        if status.returncode in (0, 33, 36) or b"416" in status.stderr:
            # A bad/complete partial cannot be repaired by appending more bytes.
            partial.unlink(missing_ok=True)
        time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Download failed for {path}: {status.stderr.decode(errors='replace')}")


@contextmanager
def data_lock(name):
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / ("." + name + ".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def download_clinical():
    with data_lock("clinical-download"):
        _download_clinical()


def _download_clinical():
    download(PROV / "figshare.json", [FIGMETA])
    meta = json.loads((PROV / "figshare.json").read_text())
    assert meta["files"][0]["computed_md5"] == FIG_MD5
    download(CLINICAL, [FIG], FIG_MD5, "md5")
    assert CLINICAL.stat().st_size == meta["files"][0]["size"]
    legacy = "https://physionet.org/files/adfecgdb/1.0.0/"
    download(PROV / "duplicate_clock_r01.qrs", [legacy + "r01.edf.qrs"],
             "03cc6b7623abdf2eadaf19f561ed8a0cbac4ec97f9ace7ffa7125973fc1f715c")
    download(PROV / "duplicate_clock_r01_first100001.edfpart", [legacy + "r01.edf"],
             "b490611a3720c705e7f10ec15587d4ac0f5539af014698458c50cebc0832d602", end_byte=100000)
    json_write(PROV / "clinical_download.json", {
        "doi": "10.6084/m9.figshare.10311029.v1", "source": FIG,
        "size": CLINICAL.stat().st_size, "sha256": file_hash(CLINICAL),
        "md5": FIG_MD5, "license": "CC0-1.0"})


def synthetic_files():
    checks = {}
    for line in (PROV / "SHA256SUMS.txt").read_text().splitlines():
        digest, path = line.split()
        if SELECT.fullmatch(path):
            checks[path] = digest
    stems = {path for path in (PROV / "RECORDS").read_text().splitlines() if SELECT.fullmatch(path + ".dat")}
    if len(stems) != 630 or len(checks) != 1560:
        raise ValueError("Unexpected FECGSYNDB 1.0.0 manifest size.")
    for stem in stems:
        suffixes = (".hea", ".dat") if "_noise" in stem else (".hea", ".dat", ".qrs")
        if any(stem + s not in checks for s in suffixes):
            raise ValueError(f"Incomplete WFDB record {stem}")
    return checks


def download_synthetic():
    with data_lock("synthetic-download"):
        _download_synthetic()


def _download_synthetic():
    for name, checksum in INDEX_SHA256.items():
        download(PROV / name, [PHYS + name, MIRROR + name], checksum)
    checks = synthetic_files()
    by_digest = defaultdict(list)
    for name, digest in sorted(checks.items()):
        by_digest[digest].append(name)

    def fetch(item):
        digest, names = item
        name = names[0]
        canonical = SYN / name
        download(canonical, [MIRROR + name, PHYS + name], digest)
        for alias in names[1:]:
            target = SYN / alias
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if os.path.samefile(target, canonical) or file_hash(target) == digest:
                    continue
                target.unlink()
            os.link(canonical, target)

    # Schedule in pregnancy order, making whole waveforms ready progressively.
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        jobs = [pool.submit(fetch, pair) for pair in by_digest.items()]
        for n, job in enumerate(concurrent.futures.as_completed(jobs), start=1):
            job.result()
            if n % 100 == 0:
                print(f"verified {n}/{len(jobs)} distinct source objects", flush=True)
    manifest = [{"path": k, "size": (SYN / k).stat().st_size, "sha256": d,
                 "source": PHYS + k} for k, d in sorted(checks.items())]
    json_write(PROV / "synthetic_download.json", {
        "version": "1.0.0", "license": "ODC-By-1.0",
        "doi": "10.13026/C21P4T", "record_manifest_sha256": file_hash(PROV / "RECORDS"),
        "checksum_manifest_sha256": file_hash(PROV / "SHA256SUMS.txt"), "files": manifest})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["synthetic", "clinical", "all"], default="all", nargs="?")
    args = parser.parse_args()
    if args.dataset in ("all", "clinical"):
        download_clinical()
    if args.dataset in ("all", "synthetic"):
        download_synthetic()
