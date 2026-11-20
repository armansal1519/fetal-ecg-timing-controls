"""Freeze the V4 input roster and generate its fresh simulated pregnancies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import scipy

from v3.holdout.build import digest, make_record, write_json
from v2 import simulation as dipole
from q1.theory import locked_transfer as gaussian

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHALLENGE_IDS = ("a01","a02","a06","a07","a09","a10","a11","a16","a18","a21")
DESIGN_FILES = ("v4/data/simulation.py", "v4/data/protocol.md", "v3/holdout/build.py",
                "v2/simulation.py", "q1/theory/geometry.py", "q1/theory/locked_transfer.py",
                "replication/challenge/links_poland.csv")
LOCK = HERE/"design_lock.json"


def row(key, cohort, family, fs, duration, **kwargs):
    return dict(id=key, subject=key, cohort=cohort, family=family, fs=fs, duration_s=duration,
                signal="v4/data/signals/"+key+".npz", reference="v4/data/references/"+key+".npz", **kwargs)


def design():
    rows = []
    for i in range(160):
        families = [
            ("hermite", 3827641019, i + 1, dipole.REGIMES[i % 4], dipole.DURATION),
            ("gaussian", 80742619, 200000 + i, f"cos{(0,55,88,98)[i % 4]:02d}_sd{(18,40)[(i//4) % 2]}", gaussian.DURATION),
        ]
        for family, namespace,index,stratum,duration in families:
            rows.append(row(f"v4_{family}_{i+1:03d}", "simulation", family, 500, duration,
                            seed_namespace=namespace, seed_index=index, stratum=stratum,
                            maternal_reference="simulator_truth"))
    past = json.loads((ROOT/"v3/holdout/records.json").read_text())
    seeds = lambda records: {(r["family"],r["seed_namespace"],r["seed_index"]) for r in records}
    assert len(seeds(rows)) == 320 and not (seeds(rows) & seeds(past))
    assert 3827641019 not in {i[0] for i in dipole.NAMESPACES.values()}
    rows.extend(row("v4_cinc_"+k,"challenge","",1000,60,source_record=k,
                    maternal_reference="Zenodo 7848240 channel 1 calculated annotation",
                    subject_identity="unknown, record used as descriptive unit") for k in CHALLENGE_IDS)
    return dict(roster=rows,numpy=np.__version__,scipy=scipy.__version__,
                files={f:digest(ROOT/f) for f in DESIGN_FILES})


def verify_design():
    stored=json.loads(LOCK.read_text())
    stored.pop("sealed_utc")
    if stored != design():
        raise RuntimeError("V4 roster or generator changed")
    return stored["roster"]


def seal():
    if not LOCK.exists():
        write_json(LOCK, dict(sealed_utc=datetime.now(timezone.utc).isoformat(),**design()))
    return verify_design()


def build(workers):
    rows=[r for r in verify_design() if r["cohort"]=="simulation"]
    for folder in ("signals","references"):
        (HERE/folder).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        final=[]
        for i,item in enumerate(pool.map(make_record,rows),1):
            final.append(item)
            if i % 80==0:
                print(f"Generated {i}/320 independent virtual recordings",flush=True)
    past = json.loads((ROOT/"v3/holdout/records.json").read_text())
    for key in ("signal_sha256","reference_sha256"):
        hashes={r[key] for r in final}
        if len(hashes)!=320 or hashes.intersection(r[key] for r in past):
            raise ValueError("Duplicate simulation waveform/reference")
    write_json(HERE/"simulation_records.json",final)
    return final


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["seal","build"])
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.stage=="seal":
        seal()
    else:
        build(args.workers)
