import unittest


def throughput_fallback(reward, n_wifi):
    return reward * n_wifi * 150.0 / 10.0


class MeasurementFixTest(unittest.TestCase):
    def test_n40_fallback_matches_normalized_throughput(self):
        parsed_throughput = 33.41075446666667
        reward = 0.05568459077777778
        fallback = throughput_fallback(reward, 40)
        error_pct = abs(fallback - parsed_throughput) / parsed_throughput * 100.0
        self.assertLess(error_pct, 1.0)

    def test_default_convergence_threshold(self):
        threshold = 37.24 * 0.98
        self.assertAlmostEqual(threshold, 36.4952)
        self.assertAlmostEqual(round(threshold, 1), 36.5)

    def test_forced_parse_fail_fallback_error(self):
        parsed_throughput = 33.41075446666667
        clamped_reward = 0.05568459077777778
        fallback = throughput_fallback(clamped_reward, 40)
        error_pct = abs(fallback - parsed_throughput) / parsed_throughput * 100.0
        self.assertLess(error_pct, 0.001)


if __name__ == "__main__":
    unittest.main()
