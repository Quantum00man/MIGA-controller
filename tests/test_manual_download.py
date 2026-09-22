import unittest
from pathlib import Path

from app.api.routes import MANUAL_PDF_PATH, router


class ManualDownloadTests(unittest.TestCase):
    def test_manual_pdf_exists(self):
        self.assertTrue(MANUAL_PDF_PATH.is_file())
        self.assertEqual(MANUAL_PDF_PATH.suffix.lower(), ".pdf")

    def test_download_route_is_registered(self):
        paths = {route.path for route in router.routes}
        self.assertIn("/manual/download", paths)

    def test_index_displays_manual_download_beside_brand(self):
        index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        brand_position = index_html.index("MIGA Controller")
        manual_position = index_html.index('href="/manual/download"')
        self.assertLess(brand_position, manual_position)
        self.assertLess(manual_position - brand_position, 500)
        self.assertIn('download="MIGA_Controller_Manual.pdf"', index_html)


if __name__ == "__main__":
    unittest.main()
