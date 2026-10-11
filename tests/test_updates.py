import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment import (record_gradient, subject_gradient, clip_gradient,
                        aggregate_step, probabilities, NORMALIZER, metrics, synthetic_subjects)


class Updates(unittest.TestCase):
    def test_finite_difference(self):
        rng = np.random.default_rng(1)
        x, y, p = rng.normal(size=(5, 3)), np.array([0, 1, 0, 1, 1]), rng.normal(size=(4, 2))
        analytical = record_gradient(x, y, p)
        numerical = np.zeros_like(p)
        for index in np.ndindex(p.shape):
            plus, minus = p.copy(), p.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            loss_plus = -np.log(probabilities(x, plus)[np.arange(5), y]).mean()
            loss_minus = -np.log(probabilities(x, minus)[np.arange(5), y]).mean()
            numerical[index] = (loss_plus-loss_minus) / 2e-6
        np.testing.assert_allclose(analytical, numerical, atol=1e-8)

    def test_record_balance(self):
        p = np.zeros((2, 2))
        short = (np.ones((1, 1)), np.array([0]))
        long = (np.zeros((9, 1)), np.ones(9, dtype=int))
        expected = (record_gradient(*short, p)+record_gradient(*long, p))/2
        np.testing.assert_allclose(subject_gradient([short, long], p), expected)
        self.assertFalse(np.allclose(expected, record_gradient(
            np.concatenate([short[0], long[0]]), np.r_[short[1], long[1]], p)))

    def test_joint_clip_and_remove_one(self):
        gradient = np.array([[3., 4.], [5., 6.]])
        self.assertLessEqual(np.linalg.norm(clip_gradient(gradient)), 1+1e-12)
        small = gradient / 100
        np.testing.assert_array_equal(clip_gradient(small), small)
        common = np.ones((2, 2))
        full = aggregate_step([common, gradient], (2, 2), True, 2, np.random.default_rng(4))
        removed = aggregate_step([common], (2, 2), True, 2, np.random.default_rng(4))
        self.assertLessEqual(np.linalg.norm(full-removed), 1/NORMALIZER+1e-12)

    def test_empty_batch_noise(self):
        actual = aggregate_step([], (2, 2), True, 2, np.random.default_rng(8))
        expected = np.random.default_rng(8).normal(size=(2, 2))*2/NORMALIZER
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(np.linalg.norm(actual), 0)

    def test_rng_isolation(self):
        a = np.random.default_rng(np.random.SeedSequence([31, 1]))
        b = np.random.default_rng(np.random.SeedSequence([31, 1]))
        noise = np.random.default_rng(np.random.SeedSequence([31, 2]))
        for _ in range(30):
            np.testing.assert_array_equal(a.random(24)<1/3, b.random(24)<1/3)
            noise.normal(size=(100, 100))

    def test_subject_normalized_metrics(self):
        parameters = np.zeros((2, 2))
        subjects = [[(np.zeros((1, 1)), np.array([0]))],
                    [(np.zeros((99, 1)), np.ones(99, dtype=int))]]
        result = metrics(subjects, parameters, 2)
        self.assertAlmostEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 1/3)

    def test_deterministic_training_rerun(self):
        subjects = synthetic_subjects(24, 111)
        masks = np.random.default_rng(222).random((30, 24)) < 1/3
        finals = []
        for _ in range(2):
            parameters = np.zeros((4, 3))
            noise = np.random.default_rng(333)
            for mask in masks:
                gradients = [subject_gradient(subjects[i], parameters) for i in np.flatnonzero(mask)]
                parameters -= aggregate_step(gradients, parameters.shape, True, 3.007841, noise)
            finals.append(parameters)
        np.testing.assert_array_equal(*finals)


if __name__ == '__main__':
    unittest.main()
