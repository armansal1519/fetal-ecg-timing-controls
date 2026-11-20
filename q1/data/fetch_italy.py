"""Download the released waveform files used for cross-dataset signal linkage."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from time import sleep

BASE = "https://physionet-open.s3.amazonaws.com/ninfea/1.0.0/"
RAW = Path(__file__).resolve().parent / "raw"


def fetch(url, path, checksum=None):
    if path.exists() and (not checksum or sha256(path.read_bytes()).hexdigest() == checksum):
        return
    for attempt in range(3):
        try:
            with urlopen(url, timeout=90) as response:
                data = response.read()
            if checksum and sha256(data).hexdigest() != checksum:
                raise ValueError(f"checksum mismatch: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authors-mirror", action="store_true", help="Waveforms only from the authors' byte-identical GitHub mirror")
    mirror = parser.parse_args().authors_mirror
    RAW.mkdir(parents=True, exist_ok=True)
    index = RAW / "ninfea_sha256.txt"
    fetch(BASE + "SHA256SUMS.txt", index)
    tasks = []
    for line in index.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        name = filename.lstrip("*./")
        if name.startswith("wfdb_format_ecg_and_respiration/") and name.endswith((".dat", ".hea")):
            url = BASE + name
            if mirror:
                url = "https://raw.githubusercontent.com/rsameni/NInFEADataset/b8ef8562189fc3864b48fd94a80890d8d2f27d8f/physionet/WFDBFormatData/" + Path(name).name
            tasks.append((url, RAW / name, digest))
    assert len(tasks) == 120, len(tasks)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda args: fetch(*args), tasks))
    print(f"Verified {len(tasks)} files")


if __name__ == "__main__":
    main()
