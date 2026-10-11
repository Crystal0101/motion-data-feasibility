"""Development references, deliberately separate from private-head simulation."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import t
from src.corrected_data import pooled_features

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[41001,41002,41003]

def scores(labels, predicted, subjects):
    confusion=np.zeros((27,27))
    for subject in sorted(set(subjects)):
        mask=subjects==subject
        count=np.zeros((27,27))
        np.add.at(count,(labels[mask],predicted[mask]),1)
        confusion+=count/mask.sum()/len(set(subjects))
    den=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*confusion.diagonal(),den,out=np.zeros(27),where=den>0)
    return dict(accuracy=float(confusion.trace()),macro_f1=float(f1.mean()))

def save_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load():
    return [dict(np.load(ROOT/'data'/('corrected_'+name+'.npz'),allow_pickle=False))
            for name in ['train','development']]

class TemporalReference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=torch.nn.GRU(27,48,batch_first=True)
        self.readout=torch.nn.Linear(48,27)
    def forward(self,sequence):
        _,hidden=self.recurrent(sequence)
        return self.readout(hidden[-1])

def run():
    destination=ROOT/'results/corrected'
    assert not (destination/'references.csv').exists(), 'Do not overwrite prior runs'
    train,dev=load();rows=[];history=[]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS:
        for method in ['fixed_logistic','fixed_extra_trees','mask_only_logistic','gru']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            started=time.perf_counter()
            if method!='gru':
                xtrain=train['sequence'][:,:,-1] if method=='mask_only_logistic' else train['features']
                xdev=dev['sequence'][:,:,-1] if method=='mask_only_logistic' else dev['features']
                model=(ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,random_state=seed,n_jobs=2)
                       if method=='fixed_extra_trees' else LogisticRegression(C=1,max_iter=2000))
                model.fit(xtrain,train['labels'])
                train_score=scores(train['labels'],model.predict(xtrain),train['subjects'])
                dev_score=scores(dev['labels'],model.predict(xdev),dev['subjects'])
                loss_initial=loss_final=float('nan')
            else:
                model=TemporalReference();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
                xtrain=torch.tensor(train['sequence']);xdev=torch.tensor(dev['sequence'])
                labels=torch.tensor(train['labels'],dtype=torch.long)
                order_rng=torch.Generator().manual_seed(seed+100000)
                loss_initial=None
                for epoch in range(41):
                    if epoch%5==0:
                        model.eval()
                        with torch.no_grad():
                            logits=model(xtrain)
                            loss=float(torch.nn.functional.cross_entropy(logits,labels))
                            train_score=scores(train['labels'],logits.argmax(1).numpy(),train['subjects'])
                            dev_score=scores(dev['labels'],model(xdev).argmax(1).numpy(),dev['subjects'])
                        if loss_initial is None:loss_initial=loss
                        history.append(dict(seed=seed,epoch=epoch,train_loss=loss,
                            train_macro_f1=train_score['macro_f1'],development_macro_f1=dev_score['macro_f1']))
                        print(json.dumps(history[-1]),flush=True)
                    if epoch==40:break
                    model.train()
                    order=torch.randperm(len(labels),generator=order_rng)
                    for start in range(0,len(labels),32):
                        batch=order[start:start+32]
                        optimizer.zero_grad(set_to_none=True)
                        torch.nn.functional.cross_entropy(model(xtrain[batch]),labels[batch]).backward()
                        optimizer.step()
                loss_final=loss
            rows.append(dict(seed=seed,method=method,
                **{'train_'+k:v for k,v in train_score.items()},
                **{'development_'+k:v for k,v in dev_score.items()},
                train_loss_initial=loss_initial,train_loss_final=loss_final,
                seconds=time.perf_counter()-started))
            save_csv(destination/'references_progress.csv',rows)
    save_csv(destination/'references.csv',rows)
    save_csv(destination/'reference_learning_curves.csv',history)
    print('Reference runs complete',flush=True)

if __name__=='__main__':run()
