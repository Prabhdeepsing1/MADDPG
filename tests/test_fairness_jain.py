import unittest


def jain_index(counts):
    positive = [float(value) for value in counts if value > 0]
    if not positive:
        return 1.0

    sum_values = sum(positive)
    sum_squares = sum(value * value for value in positive)
    if sum_squares == 0:
        return 1.0
    return (sum_values * sum_values) / (len(counts) * sum_squares)


class JainIndexTest(unittest.TestCase):
    def test_equal_counts(self):
        self.assertAlmostEqual(jain_index([10, 10, 10, 10]), 1.0)

    def test_one_station_receives_all(self):
        self.assertAlmostEqual(jain_index([100, 0, 0, 0]), 0.25)

    def test_known_three_value_example(self):
        self.assertAlmostEqual(jain_index([1, 2, 3]), 0.8571428571428571)

    def test_all_zero_counts(self):
        self.assertEqual(jain_index([0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
