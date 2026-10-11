"""Small saved-result interface used by the review notebook."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/corrected'

def read(name):
    with (RESULTS/name).open() as f:return list(csv.DictReader(f))

def show_audit():
    data=json.loads((RESULTS/'data_audit.json').read_text())
    print('Recordings: 646 training / 215 development; subjects: 24 / 8')
    print('Held-out payloads opened:',data['heldout_payloads_opened'])
    print('Absent index slots:',data['absent_index_slots'],'; empty payloads:',data['empty_payloads'])
    print('Constant pooled training features:',data['splits']['train']['constant_pooled_dimensions'])

def show_references():
    rows=read('references.csv')
    print('Method                       Dev macro-F1, %      Train macro-F1, %')
    for method in ['fixed_logistic','fixed_extra_trees','gru','mask_only_logistic']:
        subset=[r for r in rows if r['method']==method]
        dev=[100*float(r['development_macro_f1']) for r in subset]
        train=[100*float(r['train_macro_f1']) for r in subset]
        print(f'{method:28s} {np.mean(dev):6.2f} [{min(dev):.2f}, {max(dev):.2f}]     {np.mean(train):6.2f}')
    print('Brackets: seed range, not confidence intervals. Logistic repeats are deterministic.')

def show_noise():
    rows=read('noise_runs.csv')
    print('Seed      Unclipped   Clipped   Noise simulation   Difference, pp')
    for seed in range(42001,42011):
        values={r['method']:100*float(r['development_macro_f1']) for r in rows if int(r['seed'])==seed}
        print(f"{seed}       {values['unclipped']:6.2f}     {values['clipped']:6.2f}        {values['noise_simulation']:6.2f}           {values['noise_simulation']-values['clipped']:+6.2f}")
    summary=json.loads((RESULTS/'noise_summary.json').read_text())
    diff=summary['noise_minus_clipped'];lo,hi=np.array(diff['ci95'])*100
    print(f"Mean difference: {100*diff['mean']:+.2f} pp; paired t interval: [{lo:+.2f}, {hi:+.2f}] pp")
    print('Interval describes optimization-seed variation, not new-subject uncertainty.')

def plot_curves():
    rows=read('reference_learning_curves.csv')
    figure,axes=plt.subplots(1,2,figsize=(9,3.2))
    for seed in [41001,41002,41003]:
        subset=[r for r in rows if int(r['seed'])==seed]
        epochs=[int(r['epoch']) for r in subset]
        axes[0].plot(epochs,[float(r['train_loss']) for r in subset],label=str(seed))
        axes[1].plot(epochs,[100*float(r['development_macro_f1']) for r in subset])
    axes[0].set(xlabel='Epoch',ylabel='Training cross-entropy')
    axes[1].set(xlabel='Epoch',ylabel='Development macro-F1 (%)')
    axes[0].legend(title='Seed',fontsize=8)
    figure.tight_layout();return figure

def show_validation():
    for name in ['validation.json','reference_reproduction.json']:
        report=json.loads((RESULTS/name).read_text())
        print(name,':',report)
