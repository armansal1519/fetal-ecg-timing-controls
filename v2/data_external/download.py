"""Resume and verify the original pinned figshare release of NIFERCA."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
METADATA = "https://api.figshare.com/v2/articles/31912092/versions/1"
SIZE = 910624448
MD5 = "17e2bbff10817cbf195aa53588d67aae"
URL = "https://ndownloader.figshare.com/files/63405588"
CHUNK = 32 * 1024 * 1024


def digest(path, algorithm="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def download_chunk(bounds):
    start, stop = bounds
    size = stop - start
    chunk = RAW / "ranges" / f"{start:012d}-{stop:012d}.bin"
    if chunk.exists() and chunk.stat().st_size == size:
        return chunk
    with tempfile.NamedTemporaryFile(
        dir=chunk.parent, suffix=".partial", delete=False
    ) as handle:
        tmp = Path(handle.name)
    header = tmp.with_suffix(".headers")
    try:
        code = subprocess.check_output(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--retry",
                "4",
                "--range",
                f"{start}-{stop - 1}",
                "--max-filesize",
                str(size),
                "--dump-header",
                str(header),
                "--write-out",
                "%{http_code}",
                "--output",
                str(tmp),
                URL,
            ],
            text=True,
        )
        content_ranges = re.findall(
            r"^content-range:\s*bytes (\d+)-(\d+)/(\d+)",
            header.read_text(),
            re.I | re.M,
        )
        if (
            code != "206"
            or tmp.stat().st_size != size
            or not content_ranges
            or tuple(map(int, content_ranges[-1])) != (start, stop - 1, SIZE)
        ):
            raise IOError(
                f"Figshare range response {bounds}: HTTP {code}, {tmp.stat().st_size} bytes"
            )
        os.replace(tmp, chunk)
    finally:
        tmp.unlink(missing_ok=True)
        header.unlink(missing_ok=True)
    return chunk


def download(workers=8):
    RAW.mkdir(parents=True, exist_ok=True)
    meta = RAW / "figshare.json"
    if not meta.exists():
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                METADATA,
                "--output",
                str(meta),
            ],
            check=True,
        )
    entry = json.loads(meta.read_text())["files"][0]
    if (entry["size"], entry["computed_md5"], entry["download_url"]) != (
        SIZE,
        MD5,
        URL,
    ):
        raise RuntimeError("Unexpected publisher metadata for NIFERCA v1")
    archive = RAW / "NIFERCA_DATASET.zip"
    if (
        archive.exists()
        and archive.stat().st_size == SIZE
        and digest(archive, "md5") == MD5
    ):
        print("Archive matches the publisher's checksum.", flush=True)
        return archive
    prefix = RAW / "NIFERCA_DATASET.zip.part"
    start = prefix.stat().st_size if prefix.exists() else 0
    if start > SIZE:
        raise ValueError("Invalid partial archive")
    (RAW / "ranges").mkdir(exist_ok=True)
    bounds = [(i, min(i + CHUNK, SIZE)) for i in range(start, SIZE, CHUNK)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(download_chunk, bounds))
    tmp = archive.with_suffix(".assembled")
    with tmp.open("wb") as out:
        if start:
            with prefix.open("rb") as f:
                if prefix.stat().st_size != start:
                    raise ValueError("Prior single-stream download still running")
                shutil.copyfileobj(f, out, 2**20)
        for chunk in chunks:
            with chunk.open("rb") as f:
                shutil.copyfileobj(f, out, 2**20)
    if tmp.stat().st_size != SIZE or digest(tmp, "md5") != MD5:
        raise IOError("NIFERCA download checksum mismatch")
    os.replace(tmp, archive)
    # Only these transient inputs, whose bytes are in the verified archive, are removed.
    for path in [prefix, *chunks]:
        path.unlink(missing_ok=True)
    print("Publisher MD5 validated.", flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    download(parser.parse_args().workers)
