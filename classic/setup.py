"""Install the pinned reference sources and a local Octave runtime for the ECG study."""

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
SOURCES = {
    "randtx.m": (
        "https://math.aalto.fi/teaching/numsym/04/matlab/moler/m/randtx.m",
        "3c6b189efbaaffd55e0d68700726a2cfc6056f967a9e67f7dda9f33007c3c600",
    ),
    "benchmark.zip": (
        "https://codeload.github.com/mad-lab-fau/fecg-benchmarking/zip/0594916a3f4b78ccb5b06cbadfbb179a524698f4",
        "d6a83372fa7d4fdbbdc0a5f623e9367cc3d13bfc94c8d583a526db5448d1a1c9",
    ),
    "varanini.zip": (
        "https://archive.physionet.org/challenge/2013/sources/pmea/pmea-varanini.zip",
        "180fc53d30a031992cfa663a23e81ae24e2ea10f2ac32ef051eca03ab8715f28",
    ),
    "behar.zip": (
        "https://archive.physionet.org/challenge/2013/sources/joachimbehar.zip",
        "28cdd100c6f2604bf1b7e8ef5f2b7134211faf8eb9bd96e56fdb9cb5d5e02591",
    ),
    "octave-signal.tar.gz": (
        "https://github.com/gnu-octave/octave-signal/releases/download/1.4.8/signal-1.4.8.tar.gz",
        "f5989b4f148e6f5b696a7b7ceee8c610140acf6084eb3e7eafbf0eb38df79a33",
    ),
}


def write_source(path, content):
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)


def fetch(url, filename, digest):
    filename = Path(filename)
    if not filename.exists():
        part = filename.with_suffix(".download")
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", url, "-o", str(part)],
            check=True,
        )
        if hashlib.sha256(part.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Digest mismatch for {part}")
        part.replace(filename)
    if hashlib.sha256(filename.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Digest mismatch for {filename}")


def reference_sources():
    UPSTREAM.mkdir(exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        archive = UPSTREAM / name
        fetch(url, archive, digest)
        if name.endswith(".m"):
            continue
        if name.endswith(".gz"):
            with tarfile.open(archive) as src:
                for member in src.getmembers():
                    if (
                        member.isfile()
                        and ".." not in Path(member.name).parts
                        and not Path(member.name).is_absolute()
                    ):
                        dest = UPSTREAM / member.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        write_source(dest, src.extractfile(member).read())
            continue
        kind = name.split(".")[0]
        with zipfile.ZipFile(archive) as src:
            for member in src.infolist():
                path = Path(member.filename)
                if member.is_dir() or ".." in path.parts or path.is_absolute():
                    continue
                if kind == "benchmark":
                    if path.name not in {
                        "Varanini14.m",
                        "Behar14.m",
                        "PowerMF.m",
                        "matched_filter.m",
                        "LICENSE",
                    }:
                        continue
                elif path.suffix.lower() not in {".m", ".txt"}:
                    continue
                dest = UPSTREAM / kind / Path(*path.parts[1:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_source(dest, src.read(member))
    # Preserve original source files; generate narrowly scoped compatibility overlays.
    compat = ROOT / "compat"
    compat.mkdir(exist_ok=True)
    for path in (UPSTREAM / "behar").rglob("*.m"):
        text = path.read_text(encoding="latin-1")
        # Make fallback branches observable, including inner preprocessing fallbacks.
        text, count = re.subn(
            r"(?m)^(\s*)catch ME\s*$",
            r"\1catch ME\n    fprintf(2,'REFERENCE_CATCH %s:%s\\n',mfilename,ME.message);",
            text,
        )
        if path.name == "assess_regularity.m":
            # MATLAB interp1 defaults to extrapolation for a cubic spline.
            text = text.replace("hr,xi,'spline'", "hr,xi,'spline','extrap'")
        if path.name == "preprocessing.m":
            text = text.replace(
                "'subfunctions/FiltersCoeff/'",
                "fullfile(fileparts(mfilename('fullpath')),'..','upstream','behar','subfunctions','FiltersCoeff')",
            )
            # MATLAB permits the empty subscript expression in this assignment.
            # Octave can reject the empty colon operand even when no points saturate.
            text = re.sub(
                r"(?m)^(\s*)(ecg\(ind_t,ind_ch\)\s*=\s*median\([^\n]+;)",
                r"\1if ~isempty(ind_t); \2 end",
                text,
            )
        if count:
            (compat / path.name).write_text(text)

    entry = UPSTREAM / "benchmark/Code/Behar14.m"
    (compat / "Behar14.m").write_text(
        entry.read_text()
        .replace(
            "addpath(genpath('subfunctions/'));",
            "% Paths initialized by the study driver.",
        )
        .replace(
            "catch ME",
            "catch ME\n    fprintf(2,'REFERENCE_CATCH %s:%s\\n',mfilename,ME.message);",
        )
    )
    shutil.copyfile(UPSTREAM / "randtx.m", compat / "randtx.m")
    for n in ("FecgICAm", "FecgICAf", "coshFpDeIca", "kurtFpDeIca"):
        src = UPSTREAM / "varanini" / (n + ".m")
        (compat / src.name).write_text(src.read_text().replace("rand(", "randtx("))
    from prepare_overlays import create

    create(ROOT)
    # MATLAB accepts prediction order equal to the number of samples and pads.
    lpc = (UPSTREAM / "signal-1.4.8/inst/lpc.m").read_text()
    marker = "  if (nargin < 1 || nargin > 2)"
    lpc = lpc.replace(
        marker,
        """  if nargin == 2 && isvector(x) && p == numel(x)
    x = [x(:); 0];
  endif
"""
        + marker,
    )
    (compat / "lpc.m").write_text(lpc)


def runtime():
    local = ROOT / "runtime/bin/octave-cli"
    if local.exists():
        return local
    if shutil.which("octave-cli"):
        return Path(shutil.which("octave-cli"))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Install GNU Octave with development headers (mkoctfile), then rerun setup."
        )
    binary = ROOT / "micromamba"
    fetch(
        "https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-osx-arm64",
        binary,
        "c547639f5caf742236641c362746abb4a3505774753ecd08fe341643670c4c01",
    )
    binary.chmod(0o755)
    cache = Path(tempfile.gettempdir()) / "fecg-mamba-cache"
    subprocess.run(
        [
            str(binary),
            "create",
            "--no-rc",
            "--yes",
            "--root-prefix",
            str(cache),
            "--prefix",
            str(ROOT / "runtime"),
            "--file",
            str(ROOT / "runtime-osx-arm64.lock"),
        ],
        check=True,
    )
    return local


def compile_functions(octave):
    mkoctfile = octave.parent / "mkoctfile"
    env = dict(os.environ)
    if octave.parent == ROOT / "runtime" / "bin":
        env.update(
            OCTAVE_HOME=str(octave.parent.parent),
            OCTAVE_EXEC_HOME=str(octave.parent.parent),
        )
    env["PATH"] = str(octave.parent) + os.pathsep + env.get("PATH", "")
    if platform.system() == "Darwin":
        env.update(
            CXX="/usr/bin/clang++",
            CC="/usr/bin/clang",
            XTRA_CXXFLAGS="-pthread",
            CXXFLAGS="-O2 -std=c++17 -fPIC -Wno-deprecated-declarations",
            CPPFLAGS="-D_FORTIFY_SOURCE=2",
            OCT_LINK_DEPS=" ",
        )
    for source in [UPSTREAM / "signal-1.4.8/src/medfilt1.cc", ROOT / "qrs_scan.cc"]:
        dest = ROOT / "compat" / (source.stem + ".oct")
        if not dest.exists() or dest.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(
                [str(mkoctfile), str(source), "-o", str(dest)], env=env, check=True
            )


if __name__ == "__main__":
    reference_sources()
    compile_functions(runtime())
    print("ECG reference runtime ready")
