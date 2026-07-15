import unittest

import numpy as np

from scripts.classical_ml_from_scratch import KMeans, KNNClassifier, LinearRegressionGD, LogisticRegressionGD
from scripts.math_gradient_demo import analytical_gradient, numerical_gradient
from scripts.numpy_mlp import BinaryMLP
from scripts.python_numpy_fundamentals import standardize_numpy, standardize_python


class NumpyTutorialTest(unittest.TestCase):
    def test_standardization(self) -> None:
        python_values = standardize_python([1.0, 2.0, 3.0])
        numpy_values = standardize_numpy(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(python_values, numpy_values)
        self.assertAlmostEqual(float(numpy_values.mean()), 0.0)

    def test_gradient_check(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(analytical_gradient(0.5, x, y), numerical_gradient(0.5, x, y), places=6)

    def test_linear_regression(self) -> None:
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        model = LinearRegressionGD().fit(x, np.array([1.0, 3.0, 5.0, 7.0]))
        self.assertAlmostEqual(float(model.predict(np.array([[4.0]]))[0]), 9.0, places=2)
        self.assertLess(model.losses[-1], model.losses[0])

    def test_logistic_regression(self) -> None:
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        model = LogisticRegressionGD().fit(x, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(model.predict(x), np.array([0, 0, 1, 1]))

    def test_knn(self) -> None:
        model = KNNClassifier(neighbors=3).fit(np.array([[0], [1], [9], [10]]), np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(model.predict(np.array([[0.5], [9.5]])), np.array([0, 1]))

    def test_kmeans(self) -> None:
        x = np.array([[0, 0], [0, 1], [9, 9], [9, 10]], dtype=float)
        model = KMeans(clusters=2).fit(x)
        labels = model.predict(x)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_mlp_learns_xor(self) -> None:
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        model = BinaryMLP().fit(x, y, epochs=8_000)
        np.testing.assert_array_equal(model.predict(x), y)
        self.assertLess(model.losses[-1], 0.05)


if __name__ == "__main__":
    unittest.main()
