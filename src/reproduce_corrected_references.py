"""Rerun every reference in a temporary output tree without replacing results."""
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
from src import corrected_models

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    original=corrected_models.ROOT
    with tempfile.TemporaryDirectory(prefix='motion_reference_recheck_') as temporary:
        root=Path(temporary)
        (root/'results/corrected').mkdir(parents=True)
        (root/'data').symlink_to(ROOT/'data',target_is_directory=True)
        try:
            corrected_models.ROOT=root
            corrected_models.run()
        finally:
            corrected_models.ROOT=original
        first=read(ROOT/'results/corrected/references.csv')
        repeated=read(root/'results/corrected/references.csv')
        assert len(first)==len(repeated)==12
        errors=[]
        for expected,actual in zip(first,repeated):
            assert (expected['seed'],expected['method'])==(actual['seed'],actual['method'])
            for key in expected:
                if key in ['seed','method','seconds']:continue
                a,b=float(expected[key]),float(actual[key])
                if np.isnan(a):assert np.isnan(b)
                else:
                    errors.append(abs(a-b))
                    assert abs(a-b)<1e-10,(expected['seed'],key,a,b)
        report=dict(passed=True,runs_repeated=12,max_numeric_difference=max(errors),
            scope='Same machine, same installed environment; elapsed times excluded')
        (ROOT/'results/corrected/reference_reproduction.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))

if __name__=='__main__':main()
