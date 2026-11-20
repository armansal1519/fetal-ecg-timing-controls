% Compare accelerated detector loops to original MATLAB algorithms on a seeded
% artificial waveform. The same Octave-compatible lpc implementation is used.
base=fileparts(mfilename('fullpath'));
orig=fullfile(base,'upstream','varanini'); opt=fullfile(base,'compat');
addpath(fullfile(base,'upstream','signal-1.4.8','inst'));
addpath(orig); addpath(opt);
randtx('state',0); assert(abs(randtx(1,1)-0.95012928514718)<1e-13);
rng(823);
fs=1000; N=80*fs; v=0.05*randn(N,1);
locations=cumsum(420+round(12*randn(190,1)));
for j=1:length(locations)
    k=locations(j);
    if k+22 <= N
        v(k-22:k+22) += (-22:22)'/5 .* exp(-0.5*((-22:22)'/5).^2);
    end
end
env=abs(v);
names={'QRSdetectorM','QRSdetectorF1','QRSdetectorF2'};
for n=1:length(names)
    fn=names{n}; args={env,v,fs,.35,.45,.8};
    if n==3, args={env,v,fs,.2,.45,.43,12,0}; end
    accelerated=feval(fn,args{:});
    origtext=fileread(fullfile(orig,[fn,'.m']));
    origtext=strrep(origtext,fn,[fn,'_original']);
    tempfiledir=tempname(); mkdir(tempfiledir);
    filename=fullfile(tempfiledir,[fn,'_original.m']);
    fid=fopen(filename,'w'); fwrite(fid,origtext); fclose(fid);
    addpath(tempfiledir); reference=feval([fn,'_original'],args{:});
    rmpath(tempfiledir); unlink(filename); rmdir(tempfiledir);
    assert(isequal(reference,accelerated), ['Detector scan mismatch: ',fn]);
    fprintf('PASS %s, %d identical indices\n',fn,length(reference));
end
