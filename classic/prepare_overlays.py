"""Speed-equivalent native/vectorized scans with unmodified originals retained for parity tests."""

from pathlib import Path


def create(root):
    upstream = root / "upstream/varanini"
    compat = root / "compat"
    for name, maternal in [("QRSdetectorM", 1), ("QRSdetectorF1", 0)]:
        text = (upstream / (name + ".m")).read_text()
        begin = text.index("jq=1;")
        tail = f"QRSref=qrs_scan(vadx,vsdx,Fs,pth,RRts,pmQT,meaD,QRSd,maskQT,nsd,rthT,{maternal});\nend\n"
        (compat / (name + ".m")).write_text(text[:begin] + tail)
    name = "QRSdetectorF2.m"
    text = (upstream / name).read_text()
    text = text.replace(
        "j=1;\nRRsm=",
        """% Cache the exact successive subtractions used in original weighting loops.
weight_lut=ones(1, 1+max(8192,ceil(Fs*4)));
for k=2:length(weight_lut)
    if weight_lut(k-1)-0.02>0.2, weight_lut(k)=weight_lut(k-1)-0.002;
    else, weight_lut(k:end)=weight_lut(k-1); break; end
end
j=1;
RRsm=""",
    )
    a = text.index("    if(j>1)", text.index("while QRSrefj<=length(vdx)"))
    b = text.index("    if(maxs>th", a)
    text = (
        text[:a]
        + """    positions=(1:QRSrefjrma-QRSrefjlmi+1)';
    offsets=max(0,positions-wleft-wrightc);
    if j>1, offsets=max(offsets,wleft-wleftc+2-positions); end
    weightw=reshape(weight_lut(1+min(offsets,length(weight_lut)-1)),[],1);
    [maxs,imaxs]=max(weightw.*vsdx(QRSrefjlmi:QRSrefjrma));
"""
        + text[b:]
    )
    (compat / name).write_text(text)


if __name__ == "__main__":
    create(Path(__file__).resolve().parent)
