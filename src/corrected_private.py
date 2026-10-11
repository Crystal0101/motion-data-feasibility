"""Public-seed noise utility simulation with a fixed, unfitted representation."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from src.corrected_models import load,scores,save_csv
from src.experiment import probabilities

ROOT=Path(__file__).resolve().parents[1]
SEEDS=list(range(42001,42011))
ROUNDS=300

def gradients(features,labels,subjects,parameters):
    residual=probabilities(features,parameters)
    residual[np.arange(len(labels)),labels]-=1
    return np.asarray([np.vstack((features[subjects==s].T@residual[subjects==s],
              residual[subjects==s].sum(0)))/(subjects==s).sum()
              for s in sorted(set(subjects))])

def clip(gradient):
    return gradient/np.maximum(1,np.linalg.norm(gradient,axis=(1,2),keepdims=True))

def run():
    output=ROOT/'results/corrected'
    assert not (output/'noise_runs.csv').exists(), 'Preserve previous runs'
    train,dev=load()
    budget=json.loads((ROOT/'protocols/corrected_budget.json').read_text())
    shape=(train['features'].shape[1]+1,27)
    rows=[];history=[];mask_log={}
    for seed in SEEDS:
        masks=np.random.default_rng(np.random.SeedSequence([seed,1])).random((ROUNDS,24))<1/3
        mask_log[str(seed)]=masks.astype(int).tolist()
        for method in ['unclipped','clipped','noise_simulation']:
            parameters=np.zeros(shape)
            rng=np.random.default_rng(np.random.SeedSequence([seed,2]))
            count=0;clipped_count=0
            for step in range(ROUNDS+1):
                pred=probabilities(dev['features'],parameters).argmax(1)
                measured=scores(dev['labels'],pred,dev['subjects'])
                history.append(dict(seed=seed,method=method,step=step,**measured))
                if step==ROUNDS:break
                all_gradients=gradients(train['features'],train['labels'],train['subjects'],parameters)
                chosen=all_gradients[masks[step]]
                count+=len(chosen);clipped_count+=int((np.linalg.norm(chosen,axis=(1,2))>1).sum())
                if method!='unclipped':chosen=clip(chosen)
                total=chosen.sum(0)
                if method=='noise_simulation':total+=budget['sigma']*rng.normal(size=shape)
                parameters-=total/8
            training=scores(train['labels'],probabilities(train['features'],parameters).argmax(1),train['subjects'])
            rows.append(dict(seed=seed,method=method,**{'development_'+k:v for k,v in measured.items()},
                **{'train_'+k:v for k,v in training.items()},clipping_fraction=clipped_count/max(count,1),
                selected_subject_updates=count,empty_batches=int((masks.sum(1)==0).sum())))
            print(json.dumps(rows[-1]),flush=True)
            save_csv(output/'noise_progress.csv',rows)
    save_csv(output/'noise_runs.csv',rows);save_csv(output/'noise_learning_curves.csv',history)
    (output/'noise_masks.json').write_text(json.dumps(mask_log))
    paired=np.array([next(r for r in rows if r['seed']==s and r['method']=='noise_simulation')['development_macro_f1']-
        next(r for r in rows if r['seed']==s and r['method']=='clipped')['development_macro_f1'] for s in SEEDS])
    half=float(t.ppf(.975,9)*paired.std(ddof=1)/np.sqrt(10))
    summary=dict(methods={m:float(np.mean([r['development_macro_f1'] for r in rows if r['method']==m]))
        for m in ['unclipped','clipped','noise_simulation']},noise_minus_clipped=dict(mean=float(paired.mean()),
        ci95=[float(paired.mean()-half),float(paired.mean()+half)],all_differences=paired.tolist()),budget=budget,
        protected_release=False,representation='fixed, no fitted encoder',task='record-level activity classification')
    (output/'noise_summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':run()
