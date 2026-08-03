import math
import unittest

from app.core.data_loader import DataLoader


class AllanStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_sequence_statistics_are_computed_from_finite_values(self):
        points = [
            {"signal": 1.0},
            {"signal": 2.0},
            {"signal": float("nan")},
            {"signal": 3.0},
            {"signal": 4.0},
        ]

        channel = self.loader._build_allan_channel(points, ("signal",), [1, 2])
        statistics = channel["sequence_statistics"]

        self.assertEqual(statistics["sample_count"], 4)
        self.assertAlmostEqual(statistics["mean"], 2.5)
        self.assertAlmostEqual(statistics["rms"], math.sqrt(7.5))
        self.assertAlmostEqual(statistics["standard_deviation"], math.sqrt(5.0 / 3.0))

    def test_total_channel_statistics_use_the_combined_value(self):
        points = [
            {"up": 2.0, "down": 3.0},
            {"up": 4.0, "down": 5.0},
        ]

        channel = self.loader._build_allan_channel(points, ("up", "down"), [1])
        statistics = channel["sequence_statistics"]

        self.assertEqual(statistics["sample_count"], 2)
        self.assertAlmostEqual(statistics["mean"], 7.0)
        self.assertAlmostEqual(statistics["rms"], math.sqrt(53.0))
        self.assertAlmostEqual(statistics["standard_deviation"], math.sqrt(8.0))

    def test_empty_channel_returns_empty_statistics(self):
        channel = self.loader._build_allan_channel([], ("signal",), [1])

        self.assertEqual(channel["sequence_statistics"]["sample_count"], 0)
        self.assertIsNone(channel["sequence_statistics"]["mean"])
        self.assertIsNone(channel["sequence_statistics"]["rms"])
        self.assertIsNone(channel["sequence_statistics"]["standard_deviation"])


if __name__ == "__main__":
    unittest.main()
