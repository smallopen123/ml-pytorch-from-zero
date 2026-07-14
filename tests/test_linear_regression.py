import unittest

from scripts.linear_regression_from_scratch import fit_line


class FitLineTest(unittest.TestCase):
    def test_learns_simple_line(self) -> None:
        weight, bias, losses = fit_line([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual(weight, 2.0, places=2)
        self.assertAlmostEqual(bias, 1.0, places=2)
        self.assertLess(losses[-1], losses[0])

    def test_rejects_mismatched_data(self) -> None:
        with self.assertRaises(ValueError):
            fit_line([1.0], [])


if __name__ == "__main__":
    unittest.main()
