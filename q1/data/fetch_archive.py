"""Resume a slow public ZIP transfer by ranges; verify the publisher's digest."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen

RAW = Path(__file__).resolve().parent / "raw"
URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/ff15208b-ea4c-443d-9968-2f774b38d822"
SIZE = 81344334
CHUNK = 512 * 1024
DIGEST = "911dc8452a5a79867420ce16f283f37041c07a2b7e11e04f38b662550e5f92e9"


def chunk(start):
    stop = min(start + CHUNK, SIZE)
    path = RAW / "ranges" / str(start)
    if path.is_file() and path.stat().st_size == stop - start:
        return path
    for attempt in range(4):
        try:
            request = Request(URL, headers={"Range": f"bytes={start}-{stop-1}"})
            with urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise ValueError("Range not accepted")
                data = response.read()
            if len(data) != stop - start:
                raise ValueError("Incomplete range")
            path.write_bytes(data)
            return path
        except (ValueError, OSError):
            if attempt == 3:
                raise
            sleep(1 + attempt)


def main():
    target = RAW / "italy_verified.zip"
    if target.is_file() and sha256(target.read_bytes()).hexdigest() == DIGEST:
        print("ZIP verified")
        return
    (RAW / "ranges").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(chunk, range(0, SIZE, CHUNK)))
    archive = b"".join(p.read_bytes() for p in parts)
    if sha256(archive).hexdigest() != DIGEST:
        raise ValueError("ZIP does not match publisher checksum")
    target.write_bytes(archive)
    print("ZIP verified")


if __name__ == "__main__":
    main()
