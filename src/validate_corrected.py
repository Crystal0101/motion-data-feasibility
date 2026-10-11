"""Independent saved-metric, mask and implementation checks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from scipy.special import softmax
from src.corrected_models import load,scores

ROOT=Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT/'results/corrected'/name).open() as f:return list(csv.DictReader(f))

def validate():
    frozen=json.loads((ROOT/'protocols/corrected_hash.json').read_text())
    for name,digest in frozen.items():
        if name not in ['clarification','recorded_at']:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train,dev=load()
    assert len(set(train['subjects']))==24 and len(set(dev['subjects']))==8
    assert not set(train['subjects'])&set(dev['subjects'])
    assert np.unique(train['labels']).size==np.unique(dev['labels']).size==27
    assert train['features'].shape==(646,108) and dev['features'].shape==(215,108)
    assert (train['features'].std(0)>1e-10).all()
    references=read('references.csv');rows=read('noise_runs.csv');curves=read('noise_learning_curves.csv')
    assert len(references)==12 and len(rows)==30 and len(curves)==9030
    assert len({(r['seed'],r['method']) for r in rows})==30
    masks=json.loads((ROOT/'results/corrected/noise_masks.json').read_text())
    summary=json.loads((ROOT/'results/corrected/noise_summary.json').read_text())
    budget=json.loads((ROOT/'protocols/corrected_budget.json').read_text())
    differences=[];max_error=0.
    # Recompute every endpoint with a separate gradient formulation.
    augmented=np.c_[train['features'].astype(float),np.ones(646)]
    indicator=np.eye(27)[train['labels']]
    owners=sorted(set(train['subjects']))
    dev_augmented=np.c_[dev['features'].astype(float),np.ones(215)]
    for seed in range(42001,42011):
        mask=np.asarray(masks[str(seed)],dtype=bool)
        expected=np.random.default_rng(np.random.SeedSequence([seed,1])).random((300,24))<1/3
        np.testing.assert_array_equal(mask,expected)
        for method in ['unclipped','clipped','noise_simulation']:
            parameters=np.zeros((109,27));rng=np.random.default_rng(np.random.SeedSequence([seed,2]))
            for step in range(300):
                residual=softmax(augmented@parameters,axis=1)-indicator
                updates=[]
                for j,subject in enumerate(owners):
                    if mask[step,j]:
                        selected=train['subjects']==subject
                        update=augmented[selected].T@residual[selected]/selected.sum()
                        if method!='unclipped':update/=max(1,np.linalg.norm(update))
                        updates.append(update)
                total=np.sum(updates,axis=0) if updates else np.zeros_like(parameters)
                if method=='noise_simulation':total+=budget['sigma']*rng.normal(size=parameters.shape)
                parameters-=total/8
            metrics=scores(dev['labels'],(dev_augmented@parameters).argmax(1),dev['subjects'])
            stored=next(r for r in rows if int(r['seed'])==seed and r['method']==method)
            for key,value in metrics.items():
                error=abs(value-float(stored['development_'+key]));max_error=max(max_error,error)
                assert error<1e-12,(seed,method,key)
            endpoint=next(r for r in curves if int(r['seed'])==seed and r['method']==method and int(r['step'])==300)
            assert float(endpoint['macro_f1'])==float(stored['development_macro_f1'])
        noisy=next(r for r in rows if int(r['seed'])==seed and r['method']=='noise_simulation')
        clean=next(r for r in rows if int(r['seed'])==seed and r['method']=='clipped')
        differences.append(float(noisy['development_macro_f1'])-float(clean['development_macro_f1']))
    differences=np.array(differences);half=t.ppf(.975,9)*differences.std(ddof=1)/np.sqrt(10)
    np.testing.assert_allclose(summary['noise_minus_clipped']['ci95'],[differences.mean()-half,differences.mean()+half])
    assert budget['epsilon']<=3 and not summary['protected_release']
    for seed in [41001,41002,41003]:
        epochs=[r for r in read('reference_learning_curves.csv') if int(r['seed'])==seed]
        assert [int(r['epoch']) for r in epochs]==list(range(0,41,5))
        assert float(epochs[-1]['train_loss'])<float(epochs[0]['train_loss'])
    report=dict(passed=True,reference_runs=12,noise_runs=30,curve_rows=9030,
        independently_retrained_noise_endpoints=30,max_endpoint_error=max_error,
        source_hashes_verified=True,test_payloads_opened=0,
        scope='Same-machine development reproducibility; not independent subject confirmation')
    (ROOT/'results/corrected/validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':validate()
