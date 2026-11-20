function run_classic()
  args = argv();
  location = fileparts(mfilename('fullpath'));
  upstream = fullfile(location, 'upstream');
  addpath(fullfile(upstream, 'signal-1.4.8', 'inst'));
  addpath(fullfile(upstream, 'varanini'));
  addpath(fullfile(upstream, 'benchmark', 'Code'));
  addpath(fullfile(upstream, 'benchmark', 'Code', 'helper'));
  addpath(genpath(fullfile(upstream,'behar')));
  cd(fullfile(location,'compat')); % Make instrumented wrappers visible ahead of originals.
  addpath(fullfile(location, 'compat'));
  fftw('threads',1);
  old = pwelch('R12+'); % MATLAB Welch defaults and overlap in samples
  warning('off','Octave:legacy-function');
  method=args{1}; src=args{2}; out=args{3};
  if strcmp(method,'behar')
    load(src); tm=(0:rows(x)-1)'/fs*1000;
    peaks=Behar14(tm,x,fs);
    save('-mat-binary',out,'peaks');
  elseif strcmp(method,'pre')
    load(src);
    x=FecgImpArtCanc(x,fs,'',0,0);
    x=FecgDetrFilt(x,fs,'',0,0);
    x=FecgNotchFilt(x,fs,'',0,0);
    x=FecgICAm(x,fs,'',0,0,0);
    [x,fs]=FecgInterp(x,fs,4,'',0);
    % Fourier interpolation of real ECG has real output. Octave's interpft
    % leaves round-off imaginary residues; eig can rotate an ICA axis if
    % those residues cause LAPACK to select a complex eigensolver.
    if max(abs(imag(x(:)))) > 1e-10 * max(1,max(abs(real(x(:)))))
      error('Non-numerical imaginary component after real Fourier interpolation');
    end
    x=real(x);
    mother=FecgQRSmDet(x,fs,'',0,0,0,[]);
    x=FecgQRSmCanc(x,mother,fs,'',0,0,0,[]);
    components=FecgICAf(x,fs,'',0,0,0);
    if ~isreal(components), error('Unexpected complex residual ICA'); end
    save('-mat-binary',out,'components','mother','fs');
  elseif strcmp(method,'varanini')
    load(src);
    % Public PMEA version exposes just the QRS output used by the benchmark.
    peak_seconds=FecgQRSfDet(components,fs,'',mother,0,0,0,0,[]);
    peaks=peak_seconds(:)*fs;
    save('-mat-binary',out,'peaks');
  else
    error('Unknown method');
  end
end
run_classic();
