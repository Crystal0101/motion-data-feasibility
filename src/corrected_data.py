"""Fixed anonymous-column features with explicit missing index slots."""
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def transform(values):
    return 2 * np.arctan(values) / np.pi

def frame_features(points):
    if len(points) == 0:
        return np.r_[np.zeros(26), 1.]
    values = transform(points)
    return np.r_[values.mean(0), values.std(0),
                 np.quantile(values, [.1, .5, .9], axis=0).ravel(),
                 len(points)/(len(points)+32), 1.]

def block_features(frames):
    blocks = []
    for start in range(0, 297, 10):
        chunk = frames[start:start+10]
        observed = chunk[:, -1] > 0
        signal = chunk[observed, :-1].mean(0) if observed.any() else np.zeros(26)
        blocks.append(np.r_[signal, observed.mean()])
    return np.asarray(blocks, dtype=np.float32)

def pooled_features(sequence):
    index = np.linspace(-1, 1, sequence.shape[1])
    return np.concatenate([sequence.mean(1), sequence.std(1),
            sequence[:, 15:].mean(1)-sequence[:, :15].mean(1),
            (sequence*index[None, :, None]).mean(1)], axis=1)

def build():
    split = json.loads((ROOT/'protocols/split.json').read_text())
    train, dev, test = [set(split[k]) for k in
        ['planned_train','planned_development','planned_test']]
    assert not (train & dev or train & test or dev & test)
    outputs = {name:dict(sequence=[], labels=[], subjects=[], recordings=[], masks=[])
               for name in ['train','development']}
    groups = defaultdict(list)
    pattern = re.compile(r'filtered_mmwave/(E\d\d)/(S\d\d)/(A\d\d)/frame(\d+)\.bin$')
    read_subjects, absent, empty = set(), 0, 0
    with zipfile.ZipFile(ROOT/'data/filtered_mmwave.zip') as archive:
        for item in archive.infolist():
            match = pattern.fullmatch(item.filename)
            if match and match[2] in train | dev:
                groups[(match[1],match[2],match[3])].append((int(match[4]),item))
        for (_, subject, action), items in sorted(groups.items()):
            frames = np.zeros((297,27))
            indices = [i for i,_ in items]
            assert len(indices) == len(set(indices)) and min(indices)>=1 and max(indices)<=297
            read_subjects.add(subject)
            for index,item in items:
                points = np.frombuffer(archive.read(item),dtype='<f8').reshape(-1,5)
                assert np.isfinite(points).all()
                frames[index-1] = frame_features(points)
                empty += len(points)==0
            absent += 297-len(items)
            name = 'train' if subject in train else 'development'
            out = outputs[name]
            out['sequence'].append(block_features(frames))
            out['labels'].append(int(action[1:])-1)
            out['subjects'].append(subject)
            out['recordings'].append(action)
            out['masks'].append(frames[:,-1])
    assert len(groups)==861 and read_subjects==train|dev and not read_subjects&test
    report=dict(recordings_read=len(groups),subjects_read=sorted(read_subjects),
                heldout_payloads_opened=0,absent_index_slots=absent,empty_payloads=empty,
                task='one prediction per recording; exploratory development',splits={})
    for name,out in outputs.items():
        arrays={k:np.asarray(v) for k,v in out.items()}
        arrays['features']=pooled_features(arrays['sequence']).astype(np.float32)
        assert np.isfinite(arrays['features']).all()
        path=ROOT/'data'/('corrected_'+name+'.npz')
        np.savez_compressed(path,**arrays)
        report['splits'][name]=dict(recordings=len(arrays['labels']),
            shape=list(arrays['sequence'].shape),
            constant_pooled_dimensions=np.flatnonzero(arrays['features'].std(0)<1e-10).tolist(),
            column_mean_variation=arrays['sequence'][:,:,:5].std(axis=(0,1)).tolist(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/'results/corrected/data_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    build()
