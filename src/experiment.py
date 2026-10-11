"""Record-balanced subject updates; public fixed features and private head only."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(31001, 31011))
ROUNDS, Q, NORMALIZER, CLIP, LR = 30, 1 / 3, 8, 1.0, 1.0


def probabilities(features, parameters):
    logits = features @ parameters[:-1] + parameters[-1]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def record_gradient(features, labels, parameters):
    residual = probabilities(features, parameters)
    residual[np.arange(len(labels)), labels] -= 1
    return np.vstack((features.T @ residual, residual.sum(axis=0))) / len(labels)


def subject_gradient(records, parameters):
    # Equal recording weight prevents long recordings dominating a person.
    return np.mean([record_gradient(x, y, parameters) for x, y in records], axis=0)


def clip_gradient(gradient):
    return gradient * min(1.0, CLIP / max(np.linalg.norm(gradient), 1e-30))


def aggregate_step(gradients, shape, clipped, sigma, noise_rng):
    total = np.zeros(shape)
    for gradient in gradients:
        total += clip_gradient(gradient) if clipped else gradient
    # Do not skip empty batches or divide by the realized batch size.
    noise = noise_rng.normal(size=shape) * (sigma * CLIP)
    return (total + noise) / NORMALIZER


def metrics(subjects, parameters, classes):
    confusion = np.zeros((classes, classes))
    for records in subjects:
        subject_confusion = np.zeros_like(confusion)
        for x, y in records:
            predicted = probabilities(x, parameters).argmax(axis=1)
            np.add.at(subject_confusion, (y, predicted), 1)
        confusion += subject_confusion / subject_confusion.sum()
    confusion /= len(subjects)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * confusion.diagonal(), denominator,
                   out=np.zeros(classes), where=denominator > 0)
    return dict(accuracy=float(confusion.trace()), macro_f1=float(f1.mean()))


def synthetic_subjects(count, seed):
    rng = np.random.default_rng(seed)
    subjects = []
    for _ in range(count):
        offset = rng.normal(0, 0.08, 3)
        records = []
        for label in range(3):
            # Different lengths make record weighting observable in tests.
            length = 4 + 3 * label
            x = np.clip(np.eye(3)[label] * 0.7 + offset +
                        rng.normal(0, 0.15, (length, 3)), -1, 1)
            records.append((x, np.full(length, label)))
        subjects.append(records)
    return subjects


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_features(path):
    content = np.load(path, allow_pickle=False)
    subjects = []
    for subject in sorted(set(content['subjects'].tolist())):
        records = []
        subject_mask = content['subjects'] == subject
        for recording in sorted(set(content['recordings'][subject_mask].tolist())):
            mask = subject_mask & (content['recordings'] == recording)
            records.append((content['features'][mask], content['labels'][mask]))
        subjects.append(records)
    return subjects


def run(mode):
    budget = json.loads((ROOT / 'protocols/budget.json').read_text())
    if mode == 'synthetic':
        train, development = synthetic_subjects(24, 19001), synthetic_subjects(8, 19002)
        classes = 3
    else:
        train = load_features(ROOT / 'data/train_features.npz')
        development = load_features(ROOT / 'data/development_features.npz')
        classes = 27
        assert len(train) == 24 and len(development) == 8
    shape = (train[0][0][0].shape[1] + 1, classes)
    directory = ROOT / 'results' / mode
    directory.mkdir(exist_ok=True)
    assert not (directory / 'final.csv').exists(), 'Preserve existing results; no overwrite'
    rows, history = [], []
    masks_log = {}
    for seed in SEEDS:
        masks = np.random.default_rng(np.random.SeedSequence([seed, 1])).random(
            (ROUNDS, len(train))) < Q
        masks_log[str(seed)] = masks.astype(int).tolist()
        for method, clipped, sigma in [('unclipped', False, 0), ('clipped', True, 0),
                                       ('private', True, budget['sigma'])]:
            parameters = np.zeros(shape)
            noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            started = time.perf_counter()
            clipping_count, contribution_count = 0, 0
            for step in range(ROUNDS + 1):
                measured = metrics(development, parameters, classes)
                history.append(dict(seed=seed, method=method, step=step, **measured))
                if step == ROUNDS:
                    break
                gradients = [subject_gradient(train[index], parameters)
                             for index in np.flatnonzero(masks[step])]
                clipping_count += sum(np.linalg.norm(g) > CLIP for g in gradients)
                contribution_count += len(gradients)
                parameters -= LR * aggregate_step(gradients, shape, clipped, sigma, noise_rng)
            rows.append(dict(seed=seed, method=method,
                             **{'development_' + k: v for k, v in measured.items()},
                             **{'train_' + k: v for k, v in metrics(train, parameters, classes).items()},
                             clipping_fraction=clipping_count / max(1, contribution_count),
                             selected_subject_updates=contribution_count,
                             empty_batches=int((masks.sum(axis=1) == 0).sum()),
                             seconds=time.perf_counter() - started))
    save_csv(directory / 'final.csv', rows)
    save_csv(directory / 'learning_curves.csv', history)
    (directory / 'selection_masks.json').write_text(json.dumps(masks_log))
    summary = {'setting':mode, 'train_subjects':len(train), 'development_subjects':len(development),
               'head_only':True, 'budget':budget, 'seeds':SEEDS, 'methods':{}, 'paired_differences':{}}
    for method in ['unclipped', 'clipped', 'private']:
        scores = np.array([r['development_macro_f1'] for r in rows if r['method'] == method])
        summary['methods'][method] = dict(mean_macro_f1=float(scores.mean()),
                                          min_macro_f1=float(scores.min()),
                                          max_macro_f1=float(scores.max()))
    for first, second in [('private', 'clipped'), ('clipped', 'unclipped')]:
        differences = np.array([next(r for r in rows if r['seed'] == seed and r['method'] == first)['development_macro_f1'] -
                                next(r for r in rows if r['seed'] == seed and r['method'] == second)['development_macro_f1']
                                for seed in SEEDS])
        half = t.ppf(.975, len(SEEDS)-1) * differences.std(ddof=1) / np.sqrt(len(SEEDS))
        summary['paired_differences'][first + '_minus_' + second] = {
            'mean':float(differences.mean()), 'ci95':[float(differences.mean()-half),float(differences.mean()+half)],
            'all_seed_differences':differences.tolist()}
    baseline = metrics(development, np.zeros(shape), classes)
    summary['constant_class_baseline'] = baseline
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'motion'], required=True)
    run(parser.parse_args().mode)
