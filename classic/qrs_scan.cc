// Compiled sample scan from Maurizio Varanini's QRSdetectorM/F1 (GPL-2.0-or-later).
// The M-files retain the original initialization and polarity selection.
#include <octave/oct.h>
#include <algorithm>
#include <cmath>
#include <vector>

DEFUN_DLD(qrs_scan,args,,"Varanini adaptive QRS threshold scan") {
  ColumnVector envelope = args(0).column_vector_value();
  ColumnVector derivative = args(1).column_vector_value();
  double fs=args(2).double_value(), pth=args(3).double_value(), rr0=args(4).double_value();
  double qt_factor=args(5).double_value(), mean=args(6).double_value();
  int width=args(7).int_value(), mask=args(8).int_value(), gap=args(9).int_value();
  double refractory_height=args(10).double_value();
  bool maternal=args(11).bool_value();
  double threshold=pth*mean, max_amplitude=0, rr=fs*rr0;
  int start=-mask, finish=start;
  bool active=false;
  std::vector<int> peaks;
  for(int i=1; i<=envelope.numel(); ++i) {
    double sample=envelope(i-1);
    if(active) {
      if(sample<threshold) {
        finish=i;
        int highest=std::max(start,1);
        for(int j=highest+1;j<=finish;++j)
          if(derivative(j-1)>derivative(highest-1)) highest=j;
        peaks.push_back(highest);
        mean=0.97*mean+(1.0-0.97)*std::min(max_amplitude,4*threshold);
        threshold=pth*mean;
        if(maternal && peaks.size()>1) {
          double delta=highest-peaks[peaks.size()-2]-rr;
          rr+=0.97*std::copysign(std::min(std::abs(delta),0.1*rr),delta);
          double seconds=rr/fs;
          if(seconds<rr0*0.4 || seconds>rr0*2.5) {seconds=rr0; rr=rr0*fs;}
          mask=std::round((0.07+qt_factor*0.420*std::sqrt(seconds))*fs);
          gap=std::round(0.078*std::sqrt(seconds)*fs);
        }
        active=false;
      } else if(sample>max_amplitude) max_amplitude=sample;
    } else if(sample>threshold) {
      if(i<start+width || i<finish+gap) {
        if(!peaks.empty()) peaks.pop_back();
        finish=i;
        max_amplitude=std::max(sample,max_amplitude);
        active=true;
      } else if(sample>threshold*(refractory_height-(refractory_height-1)*(i-finish)/mask)) {
        max_amplitude=sample; start=i; active=true;
      }
    }
  }
  ColumnVector out(peaks.size());
  for(size_t j=0;j<peaks.size();++j) out(j)=peaks[j];
  return octave_value(out);
}
