"""Infer the locked V4 model on one waveform-only archive, without reference files."""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline_classic import _segments, predict as classic
from q1.clinical import recoverable
from q1.frontend.detector import anchors as sts_detector
from v2.model import resample
from v4.model import make_bank, predict_bank


def predict(x, fs):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=4 or not np.isfinite(x).all() or not np.isfinite(fs) or fs<=0 or not len(x):
        raise ValueError("Expected four finite abdominal channels and a positive sample rate")
    duration=len(x)/fs
    common=resample(x,fs)
    issues=[]
    power=[]
    second=np.empty(0)
    if duration>=5 and np.any(np.ptp(common,axis=0)>0):
        for start,block in _segments(common):
            try:
                t=start/500+classic("power_mf",block,500)
                power.extend(t[t<duration])
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                if not recoverable(exc):
                    raise
                issues.append(dict(parent="power_mf",start_seconds=start/500,message=str(exc)))
        try:
            t=sts_detector(common,500)
            second=t[t<duration]
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            if not recoverable(exc):
                raise
            issues.append(dict(parent="sts",start_seconds=0,message=str(exc)))
    else:
        issues.append(dict(parent="both",start_seconds=0,message="Flat recording or less than 5 seconds"))
    power=np.asarray(power,dtype=float)
    outputs=predict_bank(make_bank(common,500,power,second,duration))
    return {**outputs,"power_mf":power,"sts":second},issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waveform",type=Path,help="NPZ with x [samples,4] and scalar fs")
    parser.add_argument("output",type=Path,help="New NPZ file for times, sources, probabilities and decisions")
    args=parser.parse_args()
    if args.waveform.resolve()==args.output.resolve() or args.output.suffix!=".npz" or args.output.exists():
        parser.error("Choose a new .npz output distinct from the waveform")
    with np.load(args.waveform,allow_pickle=False) as obs:
        streams,failures=predict(obs["x"],float(obs["fs"]))
    encoded=json.dumps(failures,ensure_ascii=False)
    np.savez_compressed(args.output,**streams,parent_diagnostics=np.asarray(encoded))
    print(f"{len(streams['v4'])} selected beats, {len(failures)} parent diagnostics")


if __name__=="__main__":
    main()
