"""Offline fetal beat sequence editing from two independent abdominal detector streams."""
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.special import softmax

from detector import bandpass, nearest_distance

STATE = Path(__file__).resolve().parent/"state/weights.json"
RATE = 500
BLOCK = 10*RATE
SHIFT = np.arange(-14, 15)
CORE = np.arange(-15,16)
WIDE = np.arange(-43,44)
MOTHER = np.arange(-105,126)
PARENTS = ("power","sts")
BASE_NAMES = ("power_distance", "sts_distance", "power_raw", "power_mf", "sts_raw", "sts_mf",
              "local_agreement", "power_regularity", "sts_regularity", "maternal_proximity")
MF_NAMES = ("power_cos","sts_cos","power_cos_sq","sts_cos_sq","power_strength","sts_strength")
LEAD_NAMES = ("predictive_total","predictive_median","predictive_worst","positive_leads",
              "template_amplitude","peak_norm","overlap_gain","evidence_agreement")
FEATURE_NAMES = BASE_NAMES + MF_NAMES + LEAD_NAMES
COLS = {"v4": np.arange(len(FEATURE_NAMES)), "bank_cal": np.arange(len(BASE_NAMES)+len(MF_NAMES)),
        "no_wave": np.arange(len(BASE_NAMES))}
METHODS = ("v4","bank_plain","bank_cal","no_wave")
THRESHOLDS = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])


def _nearest(values, events):
    if not len(events):
        return np.full(np.shape(values), -1, dtype=int)
    i = np.searchsorted(events, values)
    a,b = np.maximum(i-1,0), np.minimum(i,len(events)-1)
    return np.where(np.abs(values-events[a]) <= np.abs(events[b]-values),a,b)


def _mother_locations(y):
    z = bandpass(y,RATE,8,38)
    scale = np.maximum(np.median(np.abs(z[::4]),axis=0),1e-8)
    mag = np.linalg.norm(z/scale,axis=1)
    smooth = gaussian_filter1d(mag,6)
    peaks = signal.find_peaks(smooth,distance=160,height=np.quantile(smooth[::4],.65),
                              prominence=np.median(smooth[::4])*.6)[0]
    radius = np.arange(-11,12)
    if not len(peaks):
        return peaks
    ix = np.clip(peaks[:,None]+radius,0,len(y)-1)
    return ix[np.arange(len(ix)),mag[ix].argmax(1)]


def _support(centers, start, stop, n, pad, mother=None):
    good = ((centers>=pad)&(centers<n-pad)
            & ((centers<start-pad) | (centers>=stop+pad)))
    if mother is not None:
        good &= nearest_distance(centers,mother)>32
    available = centers[good]
    if len(available)>56:
        order = np.argsort(abs(available-(start+stop)*.5),kind="stable")[:56]
        available = available[order]
    return available


def _maternal_template(y, centers, start, stop):
    support = _support(centers,start,stop,len(y),126)
    if len(support)<4:
        return np.zeros((len(MOTHER),4))
    template = np.median(y[support[:,None]+MOTHER],axis=0)
    return template*signal.windows.tukey(len(MOTHER),.25)[:,None]


def _residual(y, centers, offset, mother, mt):
    indices = np.clip(centers[:,None] + offset,0,len(y)-1)
    observations = y[indices]
    if not len(mother) or not np.any(mt):
        return observations
    # Each sample is assigned to its nearest maternal complex. The amplitude is
    # fitted independently for each lead; template morphology stays cross-block.
    group = _nearest(indices,mother)
    ids, back = np.unique(group,return_inverse=True)
    aligned = y[np.clip(mother[ids,None]+MOTHER,0,len(y)-1)]
    amp = np.einsum("nsc,sc->nc",aligned,mt) / np.maximum((mt*mt).sum(0),1e-8)
    lag = indices-mother[group]
    valid = (lag >= MOTHER[0]) & (lag <= MOTHER[-1])
    profile = mt[np.clip(lag-MOTHER[0],0,len(mt)-1)]
    amplitude = np.clip(amp,-.5,2.5)[back.reshape(indices.shape)]
    return observations - profile*amplitude*valid[:,:,None]


def _fetal_template(y, anchor_samples, start, stop, mother, mt):
    support = _support(anchor_samples,start,stop,len(y),44,mother)
    if len(support)<4:
        return np.zeros((len(CORE),4)), 0.
    segments = _residual(y,support,WIDE,mother,mt)
    proto = np.median(segments,axis=0)
    center = (len(WIDE)//2 + SHIFT)[np.argmax(np.linalg.norm(proto[len(WIDE)//2 + SHIFT],axis=1))]
    shape = proto[center+CORE]*signal.windows.tukey(len(CORE),.25)[:,None]
    strength = np.linalg.norm(shape)/max(np.median(np.linalg.norm(segments[:,center+CORE]-shape,axis=(1,2))),1e-8)
    return shape, min(float(strength),10.)


def _scores(patches, template):
    taper = signal.windows.tukey(len(CORE),.25)[:,None]
    snippets = patches*taper
    dot = np.einsum("nsc,sc->nc",snippets,template)
    observed = (snippets**2).sum(1)
    norm = (template**2).sum(0)
    combined = dot.sum(1)/np.sqrt(np.maximum(observed.sum(1)*norm.sum(),1e-14))
    # Unit is the observed squared error in the held-out lead. One shared
    # nonnegative gain is estimated from the three other leads for each rotation.
    a = np.clip((dot.sum(1)[:,None]-dot)/np.maximum(norm.sum()-norm,1e-8),0,3)
    gain = 2*a*dot-a*a*norm
    relative = np.clip(gain/(observed+.04*len(CORE)), -4,1)
    aggregate = gain.sum(1)/np.maximum(observed.sum(1),1e-8)
    return combined, np.column_stack([np.clip(aggregate,-4,1),np.median(relative,axis=1),
                                      relative.min(1),(relative>0).mean(1),np.median(a,axis=1)])


def _regularity(times, start, end):
    local = times[(times>=start-5)&(times<end+5)]
    rr = np.diff(local)
    if len(rr)<3:
        return 0., .43
    period = np.clip(np.median(rr),.28,.70)
    return float(np.mean(np.abs(rr-period)<.06)), float(period)


def _anchor_array(t,duration):
    t = np.asarray(t,dtype=float)
    if (t.ndim!=1 or not np.isfinite(t).all() or np.any(np.diff(t)<=0)
            or np.any((t<0)|(t>=duration))):
        raise ValueError("Invalid detector timestamps")
    return t


def make_bank(x_at_500, fs, power_times, sts_times, duration_s):
    """Generate the identical waveform-only bank used by every sequence method.

    Bit flags 1,2,4,8 indicate native Power, refined Power, native STS and
    refined STS. All timestamps stay in seconds on the original acquisition clock.
    """
    x = np.asarray(x_at_500,dtype=float)
    if (fs!=RATE or x.ndim!=2 or x.shape[1]!=4 or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(duration_s) or duration_s<=0 or abs(len(x)/fs-duration_s)>.003):
        raise ValueError("A finite, four-lead common 500-Hz input with a matching duration is required")
    streams = [_anchor_array(p,duration_s) for p in (power_times,sts_times)]
    bank = dict(power=streams[0],sts=streams[1],duration=np.float64(duration_s))
    samples = [np.rint(t*fs).astype(int) for t in streams]
    y = bandpass(x,fs,7,65) if len(x)>40 else np.zeros_like(x)
    subsample = y[::max(1,len(y)//40000)]
    scale = np.maximum(np.median(abs(subsample),axis=0),1e-8)
    y = y/scale
    mother = _mother_locations(y) if len(y)>40 and np.any(np.ptp(y,axis=0)>1e-7) else np.empty(0,int)
    pieces = []
    for start in range(0, len(y), BLOCK):
        stop = min(start+BLOCK,len(y))
        # Block membership uses time, including anchors rounded across a boundary.
        original = [s[(s>=start/fs)&(s<stop/fs)] for s in streams]
        if not any(len(t) for t in original):
            continue
        mt = _maternal_template(y,mother,start,stop)
        shape_strength = [_fetal_template(y,s,start,stop,mother,mt) for s in samples]
        times, origins = [], []
        for parent, raw in enumerate(original):
            if not len(raw):
                continue
            t = np.rint(raw*fs).astype(int)
            times.extend(raw)
            origins.extend(np.full(len(t),1<<(2*parent)))
            template,strength = shape_strength[parent]
            if not strength:
                continue
            scan = _residual(y,t,np.arange(-29,30),mother,mt)
            pulses = scan[:,SHIFT[:,None]+CORE+29,:]
            profile = _scores(pulses.reshape(-1,len(CORE),4),template)[0].reshape(len(t),-1)
            # Stable timing convention: prefer the smallest move for a flat tie.
            best = SHIFT[np.argmax(profile-(1e-8*np.abs(SHIFT))[None,:],axis=1)]
            proposal = raw+best/fs
            valid = (proposal>=0)&(proposal<duration_s)&(t>=30)&(t<len(y)-30)
            times.extend(proposal[valid])
            origins.extend(np.full(valid.sum(),2<<(2*parent)))
        us, inv = np.unique(np.rint(np.asarray(times)*1e6).astype(np.int64),return_inverse=True)
        source = np.zeros(len(us),np.uint8)
        np.bitwise_or.at(source,inv,np.asarray(origins,np.uint8))
        t = us/1e6
        s = np.rint(t*fs).astype(int)
        patches = _residual(y,s,CORE,mother,mt)
        ok = (s>=16)&(s<len(y)-16)
        cos, feats = zip(*[_scores(patches,tv[0]) for tv in shape_strength])
        cos = np.column_stack(cos)
        cos[~ok] = 0
        geom = np.stack(feats,axis=1)[np.arange(len(s)),np.argmax(cos,axis=1)]
        geom[~ok] = 0
        ss = np.array([pair[1] for pair in shape_strength])
        distances = np.column_stack([nearest_distance(t,st) for st in streams])
        agreement = np.mean(nearest_distance(original[0],streams[1])<.04) if len(original[0]) else 0.
        regs = [_regularity(st,start/fs,stop/fs) for st in streams]
        rr = regs[int(regs[1][0]>regs[0][0])][1]
        proximity = np.maximum(0,1-nearest_distance(s,mother)/35)
        amp = np.sqrt(np.mean(np.max(np.abs(patches),axis=1)**2,axis=1))
        design = np.column_stack([
            np.exp(-distances/.030), *[(source & (1<<k))>0 for k in range(4)],
            np.full(len(t),agreement),np.full(len(t),regs[0][0]),np.full(len(t),regs[1][0]),proximity,
            cos, np.maximum(cos,0)**2, np.broadcast_to(ss,(len(t),2)), geom,
            np.log1p(amp),proximity*geom[:,0],agreement*geom[:,0],
        ])
        support = ok & (np.max(ss)>0) & (np.linalg.norm(patches,axis=(1,2))>1e-6)
        pieces.append((us,source,design,np.full(len(t),rr),support))
    if pieces:
        ticks,source,design,rr,support = [np.concatenate(c) for c in zip(*pieces)]
        order = np.argsort(ticks,kind="stable")
        ticks,order0,inv = np.unique(ticks[order],return_index=True,return_inverse=True)
        flags = np.zeros(len(ticks),np.uint8)
        np.bitwise_or.at(flags,inv,source[order])
        i = order[order0]
        design = design[i]
        design[:,2:6] = np.column_stack([(flags & (1<<b))>0 for b in range(4)])
        bank.update(time=ticks/1e6, source=flags, X=design.astype(np.float32),rr=rr[i],support=support[i])
    else:
        bank.update(time=np.empty(0),source=np.empty(0,np.uint8),X=np.empty((0,len(FEATURE_NAMES)),np.float32),
                    rr=np.empty(0),support=np.empty(0,bool))
    return bank


def _transition(dt,rr):
    steps = np.maximum(1,np.rint(dt/rr))
    return .22*np.minimum(1,np.abs(dt-steps*rr)/(.25*rr)) + .025*np.minimum(3,steps-1)


def graph(times,rr):
    """Packed local predecessor edges and prefix index for an arbitrary long gap."""
    far = np.searchsorted(times,times-1.5,side="left")
    past = np.searchsorted(times,times-.220+1e-9,side="right")
    return [(a-1,a,b,_transition(times[j]-times[a:b],rr[j]))
            for j,(a,b) in enumerate(zip(far,past))]


def decode(times, utility, supported, edges, thresholds):
    """Maximize a sum of suprathreshold node rewards minus finite RR costs.

    The empty path has value zero. Long gaps can jump to any earlier path.
    Several threshold columns can be solved simultaneously for validation.
    """
    thresholds = np.atleast_1d(thresholds)
    n,k = len(times),len(thresholds)
    value = np.full((n,k),-np.inf)
    prefix = np.zeros((n+1,k))
    arg = np.full((n+1,k),-1,int)
    parent = np.full((n,k),-1,int)
    col = np.arange(k)
    for j,(remote,a,b,cost) in enumerate(edges):
        prefix[j+1],arg[j+1] = prefix[j],arg[j]
        allow = supported[j] & (utility[j]>thresholds)
        if not np.any(allow):
            continue
        previous = np.maximum(0,prefix[remote+1]-.25)
        use = np.where(previous>0,arg[remote+1],-1)
        if b>a:
            local = value[a:b]-cost[:,None]
            best = np.argmax(local,axis=0)
            take = local[best,col] > previous
            previous = np.where(take,local[best,col],previous)
            use = np.where(take,a+best,use)
        value[j,allow] = (previous+utility[j]-thresholds)[allow]
        parent[j,allow] = use[allow]
        better = value[j] > prefix[j+1]
        prefix[j+1,better], arg[j+1,better] = value[j,better],j
    results = []
    for c in col:
        ids = []
        j = arg[-1,c]
        while j>=0:
            ids.append(j)
            j = parent[j,c]
        results.append(np.array(ids[::-1],int))
    return results


def _fixed(bank,utility):
    # Constrain one choice to the disjoint Voronoi cell of each Power anchor.
    # A native timing is always available, including an unsupported one.
    power,t = bank["power"],bank["time"]
    if not len(power):
        return np.empty(0)
    positions = _nearest(t,power)
    # Each local group is contiguous so no scan over the entire record is needed.
    cuts = np.searchsorted(positions,np.arange(len(power)+1))
    prev_ids = np.empty(0,int)
    prev_value = np.empty(0)
    histories = []
    for i,center in enumerate(power):
        ids = np.arange(cuts[i],cuts[i+1])
        ids = ids[abs(t[ids]-center)<=.1150005]
        native = abs(t[ids]-center)<0.00000051
        ids = ids[bank["support"][ids]|native]
        if not len(ids):
            return np.asarray(power).copy()
        node = utility[ids].copy()
        node[~bank["support"][ids]] = 0
        if len(prev_ids):
            dt = t[ids,None]-t[prev_ids]
            edges = np.where(dt>=.220-1e-9, prev_value-_transition(dt,bank["rr"][ids,None]),-np.inf)
            ix = np.argmax(edges,axis=1)
            node += edges[np.arange(len(ids)),ix]
        else:
            ix = np.full(len(ids),-1,int)
        histories.append((ids,ix))
        prev_ids,prev_value = ids,node
    if not np.isfinite(prev_value).any():
        return np.asarray(power).copy()
    ix = np.argmax(prev_value)
    result = []
    for ids,back in reversed(histories):
        result.append(t[ids[ix]])
        ix = back[ix]
    return np.asarray(result[::-1])


def scores(bank,state):
    x = bank["X"]
    output = {}
    for method,cols in COLS.items():
        fit = state[method]
        design = np.clip((x[:,cols]-fit["center"])/fit["scale"],-8,8)
        logits = design@np.asarray(fit["coef"]).T + fit["intercept"]
        p = softmax(logits,axis=1)
        output[method] = (p[:,2]+.30*p[:,1],p)
    cosines = x[:,len(BASE_NAMES):len(BASE_NAMES)+2]
    output["bank_plain"] = (np.maximum(0,cosines.max(1)),None)
    return output


def predict_bank(bank,state=None):
    if state is None:
        state = json.loads(STATE.read_text())
    t = bank["time"]
    g = graph(t,bank["rr"])
    evidence = scores(bank,state)
    out = {}
    for method in METHODS:
        utility,_ = evidence[method]
        chosen = decode(t,utility,bank["support"],g,[state["thresholds"][method]])[0]
        out[method] = t[chosen]
        if method=="v4":
            selected = np.zeros(len(t),bool)
            selected[chosen]=True
    raw_p,raw_s = bank["power"],bank["sts"]
    out["naive"] = np.sort(np.r_[raw_p,raw_s[nearest_distance(raw_s,raw_p)>=.220-1e-9]])
    utility,p = evidence["v4"]
    out["fixed_count"] = _fixed(bank,utility)
    out.update(candidate_time=t,candidate_source=bank["source"],candidate_selected=selected,
               candidate_reason=np.select(
                    [selected,~bank["support"],utility<=state["thresholds"]["v4"]],
                    ["selected","no_wave_support","below_threshold"],default="sequence_conflict"),
               candidate_p_tight=p[:,2],candidate_p_loose=p[:,1],candidate_p_absent=p[:,0],
               candidate_utility=utility,candidate_wave_gain=bank["X"][:,len(BASE_NAMES)+len(MF_NAMES)],
               candidate_support=bank["support"])
    return out
