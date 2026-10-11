"""Integer-order RDP bound for one hypothetical protected mechanism."""
import json
from pathlib import Path
from scipy.optimize import brentq
import numpy as np
from dp_accounting import dp_event,privacy_accountant,rdp

def epsilon(sigma):
    accountant=rdp.RdpAccountant(orders=list(range(2,129)),neighboring_relation=
        privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE)
    accountant.compose(dp_event.PoissonSampledDpEvent(1/3,dp_event.GaussianDpEvent(sigma)),count=300)
    return accountant.get_epsilon(1e-5)

if __name__=='__main__':
    sigma=float(np.ceil(brentq(lambda s:epsilon(s)-3,.5,100)*1e6)/1e6)
    budget=dict(sigma=sigma,epsilon=float(epsilon(sigma)),delta=1e-5,rounds=300,q=1/3,
        normalizer=8,clip=1,noise_sd=sigma/8,orders=list(range(2,129)),
        scope='One hypothetical secret-randomness mechanism; public-seed outputs are utility simulations')
    assert budget['epsilon']<=3
    path=Path(__file__).resolve().parents[1]/'protocols/corrected_budget.json'
    path.write_text(json.dumps(budget,indent=2));print(json.dumps(budget))
