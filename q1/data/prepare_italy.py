"""Assemble waveform-only inputs, held-out references, and leakage checks.

Run after link_italy.py. All reference sample numbers use the Mendeley
snippet clock plus the chest-only alignment offset.
"""
import csv
import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import signal

from .link_italy import FS, HERE, MIN_R, WAVES, band, correlate_snippets, read_csv, sample_alignment, source_zip, wfdb_channels
from scoring import match


ROOT = HERE.parents[1]
# One-based abdominal lead numbers. No channel or transform was selected on outcomes.
LEADS = [1, 7, 13, 19]
# Published Table 3 groups repeated recordings within women.
GROUPS = [(1,2,3), (4,5), (6,), (7,), (8,), (9,), (10,), (11,), (12,),
          (13,), (14,), (15,16), (17,), (18,), (19,), (20,), (21,)]
PERSON = {i:f"cagliari_{k:02d}" for k, group in enumerate(GROUPS,1) for i in group}
# Sulas et al. Scientific Data, Table 2: host numbers belonging to the same woman.
HOST_FAMILIES = [(1,2,3),(4,5),(6,7),(8,),(9,),(10,11),(12,),(13,14),(15,16),
                 (17,),(18,),(19,),(20,21),(22,),(23,),(24,),(25,),(26,),(27,),
                 (28,29),(30,),(31,32,33),(34,),(35,36),(37,38),(39,),(40,41),
                 (42,),(43,44),(45,),(46,),(47,48),(49,50),(51,52),(53,54),
                 (55,56),(57,),(58,),(59,60)]


def intervals(mask):
    transitions = np.flatnonzero(np.diff(np.r_[0, mask, 0].astype(int)))
    return transitions.reshape(-1, 2)


def fetal_samples(bundle, idx):
    matches = [s for s in bundle.namelist() if re.fullmatch(
        f"real_(?:fecg|ann)_{idx}_ann.csv|real_(?:fecg_ann|ann)_{idx}.csv",
        Path(s).name.lower())]
    if len(matches) != 1:
        raise ValueError((idx,matches,[s for s in bundle.namelist() if "ann" in s.lower()][:8]))
    with bundle.open(matches[0]) as src:
        values = np.atleast_1d(np.loadtxt(src, delimiter=","))
    if values.ndim != 1 or np.any(values != values.astype(int)) or len(np.unique(values)) != len(values):
        raise ValueError("Irregular annotation encoding")
    # MATLAB sample convention, no endpoint or peak-output fitted correction.
    return np.sort(values.astype(int))-1


def independent_chest(t):
    qrs = band(t)
    qrs /= np.maximum(np.quantile(abs(qrs), .97, axis=0), 1e-12)
    def detect(envelope):
        return signal.find_peaks(envelope, distance=round(.33*FS),
                height=.32*np.quantile(envelope,.98),
                prominence=.18*np.quantile(envelope,.98))[0]
    combined = detect(np.sqrt((qrs*qrs).mean(axis=1)))
    channel_checks = []
    for ch in range(3):
        single = detect(abs(qrs[:,ch]))
        times = np.r_[-10*FS, single, (len(t)+10*FS)]
        idx = np.searchsorted(times, combined)
        distance = np.minimum(combined-times[idx-1], times[idx]-combined)
        channel_checks.append(distance*1000/FS)
    return combined, np.transpose(channel_checks)


def alignment_checks(host, fragment, start, stop, lag):
    """Check local delays at opposite ends and chest correlations without labels."""
    offsets = []
    for center in [start+(stop-start)//4, start+3*(stop-start)//4]:
        half = min(FS, (stop-start)//6)
        a = host[center-half:center+half]
        scores = []
        for shift in range(-12,13):
            b = fragment[center-half-lag-shift:center+half-lag-shift]
            agreement = np.sum(a*b,axis=0)/np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0))
            scores.append(np.min(agreement))
        offsets.append(int(np.argmax(scores))-12)
    return offsets


def linked_people(by_host):
    """Union participant groupings from both papers and waveform identities."""
    people = dict(PERSON)
    for family in HOST_FAMILIES:
        matches = [m for host in family for m in by_host.get(host,[])]
        if not matches:
            continue
        group = {people[m["snippet"]] for m in matches}
        representative = min(group)
        for snippet, identity in people.items():
            if identity in group:
                people[snippet] = representative
    return people


def prepare():
    with (HERE/"linkage.csv").open() as f:
        links = [{k:int(v) if k!="min_chest_corr" else float(v) for k,v in row.items()}
                 for row in csv.DictReader(f)]
    if not links:
        raise ValueError("No verified cross-dataset links")
    by_host = defaultdict(list)
    for row in links:
        by_host[row["host"]].append(row)
    people = linked_people(by_host)
    previous = defaultdict(list)
    manifest = []
    checks = []
    repeated_annotations = []
    for folder in ["signals","references"]:
        (HERE/folder).mkdir(exist_ok=True)
    with ZipFile(source_zip()) as bundle:
        for host, matches in sorted(by_host.items()):
            subjects = {people[m["snippet"]] for m in matches}
            if len(subjects)!=1:
                raise ValueError(f"Contradictory participant links: {matches}")
            person = subjects.pop()
            chest = wfdb_channels(host,[24,25,26])
            filtered = band(chest)
            duplicate = np.zeros(len(chest),bool)
            # Remove already counted portions of repeated/relabelled acquisition.
            for earlier, used in previous[person]:
                score, approx = correlate_snippets(filtered[::8], earlier[::8])
                if score < .90:
                    continue
                exact, lag, start, stop = sample_alignment(filtered, earlier, approx)
                if exact >= MIN_R:
                    duplicate[start:stop] |= used[start-lag:stop-lag]
            keep = np.zeros(len(chest),bool)
            fetal = []
            sources = []
            for m in sorted(matches,key=lambda r:r["snippet"]):
                i,start,stop,lag = (m[k] for k in ["snippet","start","stop","offset_samples"])
                src = band(read_csv(bundle,"Real_Traces",i,[0,1,2]))
                offset_checks = alignment_checks(filtered,src,start,stop,lag)
                marks = fetal_samples(bundle,i)
                if np.any(marks<0) or np.any(marks>=len(src)):
                    raise ValueError("Marks outside the supplied snippet")
                marks = marks+lag
                marks = marks[(marks>=start)&(marks<stop)]
                for ni,prevstart,prevstop,prevmarks in sources:
                    lo,hi=max(prevstart,start),min(prevstop,stop)
                    if hi <= lo:
                        continue
                    first=prevmarks[(lo<=prevmarks)&(prevmarks<hi)]/FS
                    second=marks[(lo<=marks)&(marks<hi)]/FS
                    pairs=match(first,second,50)
                    diff=1000*abs(first[pairs[:,0]]-second[pairs[:,1]])
                    repeated_annotations.append(dict(host=host,first_snippet=ni,second_snippet=i,
                        overlap_s=(hi-lo)/FS,first_marks=len(first),second_marks=len(second),
                        matched_50ms=len(pairs),matched_10ms=len(match(first,second,10)),
                        median_difference_ms=np.median(diff) if len(diff) else None))
                sources.append((i,start,stop,marks))
                fetal.extend(marks[~keep[marks]&~duplicate[marks]])
                keep[start:stop] = True
                checks.append(dict(host=host,snippet=i,quarter_offset_samples=offset_checks[0],
                                   three_quarter_offset_samples=offset_checks[1],**{k:v for k,v in m.items() if k not in ["host","snippet"]}))
            keep &= ~duplicate
            if not keep.any():
                continue
            reverse_marks = []
            reverse_used = duplicate.copy()
            for _,start,stop,marks in reversed(sources):
                reverse_marks.extend(marks[~reverse_used[marks]])
                reverse_used[start:stop] = True
            previous[person].append((filtered,keep))
            maternal,disagreement = independent_chest(chest)
            chest_qa = disagreement[keep[maternal]].ravel()
            name = f"ninfea_{host:02d}"
            signal_file,ref_file = (HERE/"signals"/(name+".npz"), HERE/"references"/(name+".npz"))
            abdominal = wfdb_channels(host,[l-1 for l in LEADS])
            np.savez_compressed(signal_file,x=abdominal.astype(np.float32),fs=FS)
            ref_fields = dict(fetus=np.asarray(sorted(fetal),dtype=float)/FS,
                              mother=maternal.astype(float)/FS,invalid=intervals(~keep).astype(float)/FS)
            if not np.array_equal(sorted(fetal),sorted(reverse_marks)):
                ref_fields["fetus_reverse_excerpt_priority"] = np.asarray(sorted(reverse_marks),dtype=float)/FS
            np.savez_compressed(ref_file,**ref_fields)
            manifest.append(dict(
                id=name,subject=person,fs=FS,duration_s=len(chest)/FS,
                signal=str(signal_file.relative_to(ROOT)),reference=str(ref_file.relative_to(ROOT)),
                monitored_s=keep.sum()/FS,maternal_reference="held_out_thoracic_proxy",
                fetus_reference="cardiologist abdominal electrical marks supported by simultaneous fetal Doppler",
                input_leads_1based=LEADS,
                labelled_snippets=sorted(m["snippet"] for m in matches), waveform_host=host,
                thoracic_calls_in_annotated_exposure=int(keep[maternal].sum()),
                thoracic_channel_comparisons=int(len(chest_qa)),
                thoracic_concordant_10ms=int((chest_qa<=10).sum()),
                thoracic_concordant_40ms=int((chest_qa<=40).sum()),
                signal_sha256=sha256(signal_file.read_bytes()).hexdigest(),
                reference_sha256=sha256(ref_file.read_bytes()).hexdigest(),
            ))
    (HERE/"records.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with (HERE/"alignment_checks.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    if repeated_annotations:
        with (HERE/"within_reference_agreement.csv").open("w",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=list(repeated_annotations[0]))
            writer.writeheader()
            writer.writerows(repeated_annotations)
    print(f"{len(manifest)} recordings; {len({r['subject'] for r in manifest})} women; "
          f"{sum(r['monitored_s'] for r in manifest):.3f} adjudicated seconds")


if __name__=="__main__":
    prepare()
