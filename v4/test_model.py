"""Regression and independent numerical fixtures for the two-source candidate graph."""
import itertools
import unittest

import numpy as np

from scoring import match
from v4 import model


def example_bank():
    # Power missed the beat at 1.45, and supplied an extra false peak at 1.70.
    power=np.array([.55,1.,1.70,1.90,2.35])
    sts=np.array([.55,1.,1.45,1.90,2.35])
    t=np.array([.55,1.,1.45,1.46,1.70,1.90,2.35])
    return dict(power=power,sts=sts,time=t,source=np.array([5,5,4,8,1,5,5],np.uint8),
                X=np.zeros((len(t),len(model.FEATURE_NAMES))),rr=np.full(len(t),.45),
                support=np.ones(len(t),bool),duration=3.)


class CandidateGraphTests(unittest.TestCase):
    def test_added_removed_duplicate_and_arbitrary_gap(self):
        b=example_bank()
        t=np.r_[b["time"],30.01]
        score=np.array([.94,.95,.97,.83,.08,.92,.95,.9])
        idx=model.decode(t,score,np.ones(len(t),bool),model.graph(t,np.full(len(t),.45)),[.4])[0]
        np.testing.assert_array_equal(t[idx],[.55,1.,1.45,1.90,2.35,30.01])

    def test_unsupported_wave_and_empty_detectors(self):
        x=np.zeros((14500,4))
        anchors=np.arange(.5123,28,.417)
        bank=model.make_bank(x,500,anchors,[],29)
        out=model.predict_bank(bank)
        self.assertFalse(bank["support"].any())
        self.assertTrue((out["candidate_reason"]=="no_wave_support").all())
        self.assertEqual(len(out["v4"]),0)
        np.testing.assert_allclose(out["fixed_count"],anchors,atol=5.1e-7)
        only_sts=model.make_bank(x,500,[],anchors,29)
        self.assertTrue((only_sts["source"]==4).all())
        self.assertEqual(len(model.predict_bank(only_sts)["fixed_count"]),0)
        empty=model.predict_bank(model.make_bank(x,500,[],[],29))
        self.assertTrue(all(not len(v) for v in empty.values()))

    def test_one_surviving_detector_supplies_real_templates(self):
        rng=np.random.default_rng(456)
        anchors=np.arange(.225,40,.435)
        x=rng.normal(0,.02,(20000,4))
        offsets=np.arange(-15,16)
        fetal=np.exp(-.5*(offsets/4.)**2)[:,None]*np.array([.8,-.5,1.1,.7])
        maternal=np.exp(-.5*(offsets/6.)**2)[:,None]*np.array([2.,1.7,1.3,-1.8])
        for t in anchors:
            ix=round(500*t)+offsets
            if ix[-1]<len(x):
                x[ix]+=fetal
        for t in np.arange(.17,39.9,.835):
            x[round(500*t)+offsets]+=maternal
        for p,s in [(anchors,np.empty(0)),(np.empty(0),anchors)]:
            b=model.make_bank(x,500,p,s,40)
            self.assertTrue(b["support"].any())
            self.assertTrue(np.all(np.diff(b["time"])>0))
            out=model.predict_bank(b)
            self.assertTrue(len(out["v4"])>0)
            self.assertEqual(len(out["fixed_count"]),len(p))
            self.assertEqual(len(out["candidate_time"]),len(out["candidate_reason"]))
            np.testing.assert_array_equal(out["v4"],out["candidate_time"][out["candidate_selected"]])
            np.testing.assert_allclose(out["candidate_p_tight"]+out["candidate_p_absent"]+out["candidate_p_loose"],1)

    def test_heldout_amplitude_uses_three_other_leads(self):
        i=np.arange(len(model.CORE)); shape=np.exp(-((i-14)/3.5)**2)[:,None]*np.array([1.,.5,-.7,.9])
        clean=np.stack([shape.copy(),shape.copy()])
        clean[1,:,2]*=-1
        corr,geometry=model._scores(clean,shape)
        self.assertGreater(geometry[0,0],.8)
        self.assertLess(geometry[1,0],geometry[0,0])
        self.assertLess(geometry[1,2],0)
        self.assertLess(corr[1],corr[0])

    def test_inclusive_reference_clock(self):
        t=np.array([1.201,1.712])
        self.assertEqual(len(match(t,t+.010,10)),2)
        self.assertEqual(len(match(t,t+.010001,10)),0)
        self.assertEqual(len(match(t,np.array([1.211,1.712,1.714]),50)),2)

    def test_graph_against_exhaustive_paths(self):
        t=np.array([.01,.19,.33,.44,.95,2.31,2.75,4.9])
        rr=np.array([.43,.45,.46,.45,.44,.49,.48,.47])
        utility=np.array([.85,.4,.85,.95,.75,.61,.45,.85])
        support=np.array([1,1,1,1,0,1,1,1],bool)
        for tau,selected in zip([.15,.35,.9],
                model.decode(t,utility,support,model.graph(t,rr),[.15,.35,.9])):
            def worth(ids):
                gain=0.
                for j,k in enumerate(ids):
                    gain+=utility[k]-tau
                    if j:
                        gap=t[k]-t[ids[j-1]]
                        if gap<.220-1e-9:
                            return -np.inf
                        gain-=.25 if gap>1.5 else model._transition(gap,rr[k])
                return gain
            good=np.flatnonzero((utility>tau)&support)
            best=max(worth(ix) for n in range(len(good)+1) for ix in itertools.combinations(good,n))
            self.assertAlmostEqual(worth(selected),best)

    def test_fixed_count_power_cells(self):
        bank=example_bank()
        choice=model._fixed(bank,np.array([.8,.9,.97,.7,.01,.9,.95]))
        self.assertEqual(len(choice),len(bank["power"]))
        self.assertTrue(np.all(np.diff(choice)>0))


if __name__=="__main__":
    unittest.main()
