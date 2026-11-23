"""Independent high-precision oracle for the internal review; optional mpmath."""
from fractions import Fraction
from pathlib import Path
import random
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from heldout_thresholds import replay
from traffic_blend import fit_weight


def main():
    mp.mp.dps=90
    rng=random.Random(23112026)
    checks=0
    activated=0
    for case in range(60):
        def panel(offset):
            return np.array([[rng.randrange(8)/4+offset for _ in range(4)] for _ in range(42)])
        local=panel(1)
        transfer=panel(1.+(case%4))
        truth=transfer.copy() if case%3==0 else (local.copy() if case%3==1 else panel(2))
        limits={'locked_two_sided':42,'one_sided_mixture':42,'fixed_budget_mean':42,'fixed_two_thirds_mean':28}
        scale=2
        losses=[]; widths=[]
        for l,t,y in zip(local,transfer,truth):
            losses.append(mp.mpf(float((abs(t-y)-abs(l-y)).sum()))/8)
            widths.append(mp.mpf(float(abs(t-l).sum()))/8)
        for rule,nmax in limits.items():
            actual=[]
            def observe(i):
                actual.append(i)
                return truth[i]
            result=replay(rule,np.arange(42),local,transfer,scale,observe,
                          dict(rho=.01,delta=.05,eta=.01,minimum=12),nmax)
            expected=None
            for n in range(12,nmax+1):
                total=mp.fsum(losses[:n])
                v=mp.fsum(x*x for x in widths[:n])
                rho,delta,eta=map(mp.mpf,('.01','.05','.01'))
                adj=v+rho
                gain=-total-n*eta
                if rule=='locked_two_sided':
                    cross=total+mp.sqrt(adj*mp.log(adj/(rho*delta**2))) < -n*eta
                elif rule=='one_sided_mixture':
                    z=gain/mp.sqrt(adj)
                    cross=2*mp.sqrt(rho/adj)*mp.exp(z*z/2)*mp.ncdf(z) > 1/delta
                else:
                    cross=n==nmax and gain>0
                if cross:
                    expected=n
                    break
            assert result['accepted']==(expected is not None) and actual==list(range(expected or nmax))
            activated+=int(result['accepted']);checks+=1
        # Independent exhaustive rational evaluation on nonintegral binary inputs.
        left,right,y=(a[:28]/3 for a in (local,transfer,truth))
        rr=[list(map(Fraction.from_float,a.flatten())) for a in (left,right,y)]
        def price(k):
            w=Fraction(k,4)
            loss=sum(abs((1-w)*l + w*t - ob) for l,t,ob in zip(*rr))
            return loss/(scale*y.size)+w/100
        assert fit_weight(left,right,y,scale)==min(range(5),key=price)/4
    print(f'{checks} decision/stopping/label-stream matches including {activated} activations; 60 exact blend-weight matches.')


if __name__=='__main__':
    main()
