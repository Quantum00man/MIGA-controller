import asyncio
import math
import unittest

from app.api.routes import fit_archive_scan
from app.models.schemas import ArchiveScanFitRequest, FitModelDefinition


class ArchiveScanFittingTests(unittest.TestCase):
    def test_scan_fit_returns_baseline_removed_curve_area(self):
        request = ArchiveScanFitRequest(
            x_values=[0.0, 1.0, 2.0],
            y_values=[3.0, 3.0, 3.0],
            fit_min=0.0,
            fit_max=2.0,
            model=FitModelDefinition(
                key="fixed_line",
                label="Fixed line",
                formula="signal + offset",
                parameters=[
                    {"name": "signal", "guess": "2.0", "fixed": True},
                    {"name": "offset", "guess": "1.0", "fixed": True},
                ],
                roles={"amplitude": "signal", "offset": "offset"},
                area_mode="window_integral",
            ),
        )

        response = asyncio.run(fit_archive_scan(request))

        self.assertIn("area", response)
        self.assertTrue(math.isfinite(response["area"]))
        self.assertAlmostEqual(response["area"], 4.0)


if __name__ == "__main__":
    unittest.main()
