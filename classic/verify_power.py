"""Compare Python Power-MF post-ICA output against the authors' M-file on identical input."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import io, signal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from baseline_classic import _power_mf


def main():
    rng = np.random.default_rng(3297)
    fs = 2000
    n = 90 * fs
    kernel = signal.windows.gaussian(91, 10)
    mu = np.zeros((n, 4))
    for j, (interval, amplitude) in enumerate(
        [(891, 1.2), (860, 0.5), (970, 0.3), (1020, 0.1)]
    ):
        pos = np.cumsum(rng.integers(interval - 30, interval + 30, n // interval + 4))
        pos = pos[pos < n]
        mu[pos, j] = amplitude
    components = signal.oaconvolve(
        mu, kernel[:, None], mode="same", axes=0
    ) + rng.normal(0, 0.018, (n, 4))
    expected = _power_mf(components, fs) * fs
    octave = ROOT / "runtime/bin/octave-cli"
    if not octave.exists():
        octave = Path(shutil.which("octave-cli") or "octave-cli")
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        io.savemat(work / "fixture.mat", {"Ser": components, "fs": float(fs)})
        original = (ROOT / "upstream/benchmark/Code/PowerMF.m").read_text()
        body = original[original.index("    sig = Ser'") : original.index("catch e")]
        # The original channel display uses MATLAB string arrays.
        body = "\n".join(
            line for line in body.splitlines() if 'disp("channel ' not in line
        )
        # Octave's uint16 exponent propagates integer arithmetic into gausswin.
        # Use the same integer window length as a double for the spectral call.
        body = body.replace("gausswin(Nfft)", "gausswin(double(Nfft))").replace(
            "Nfft, fs);", "double(Nfft), double(fs));"
        )
        body = body.replace("2^nextpow2(nsc));", "2^nextpow2(nsc)); Nfft=double(Nfft);")
        fixture = (
            f"""addpath('{ROOT / "upstream/signal-1.4.8/inst"}');
addpath('{ROOT / "upstream/benchmark/Code/helper"}'); addpath('{work}');
pwelch('R12+'); load('{work / "fixture.mat"}'); ms=340;
"""
            + body
            + f"\nsave('-mat-binary','{work / 'out.mat'}','fPeaks');\n"
        )
        (work / "compare.m").write_text(fixture)
        # Octave's stock findpeaks interprets negative data differently. MATLAB
        # greedily retains the largest local maxima within MinPeakDistance.
        (work / "findpeaks.m").write_text("""function [v,k]=findpeaks(x,varargin)
x=x(:); k=find(x(2:end-1)>x(1:end-2)&x(2:end-1)>=x(3:end))+1;
if nargin>1
  d=varargin{2}; [~,order]=sort(x(k),'descend'); blocked=false(length(x),1);
  accept=false(length(k),1);
  for j=order(:)'
    p=k(j); if ~blocked(p), accept(j)=true; blocked(max(1,p-d+1):min(length(x),p+d-1))=true; end
  end
  k=k(accept);
end
v=x(k);
end
""")
        subprocess.run(
            [str(octave), "--no-init-file", "--quiet", str(work / "compare.m")],
            cwd=work,
            env=dict(
                os.environ,
                OPENBLAS_NUM_THREADS="1",
                OCTAVE_HISTFILE=os.devnull,
                OCTAVE_HOME=str(octave.parent.parent),
                OCTAVE_EXEC_HOME=str(octave.parent.parent),
            ),
            check=True,
        )
        found = io.loadmat(work / "out.mat")["fPeaks"].reshape(-1) * 4 - 1
    np.testing.assert_array_equal(found, np.round(expected))
    print(f"Power-MF: {len(found)} identical full-resolution peak indices")


if __name__ == "__main__":
    main()
