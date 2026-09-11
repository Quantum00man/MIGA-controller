import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_STYLE = ROOT / "static" / "plot-publication.js"


class PlotPublicationTests(unittest.TestCase):
    def test_index_and_archive_load_shared_plot_style(self):
        for page_name in ("index.html", "archive.html"):
            html = (ROOT / "static" / page_name).read_text(encoding="utf-8")
            self.assertIn('<script src="/plot-publication.js"></script>', html)
            self.assertLess(html.index('/plot-publication.js'), html.index('const { createApp } = Vue'))
            self.assertIn('aria-label="Publication export width"', html)
            self.assertIn('<option value="double">Double column (183 mm)</option>', html)
            self.assertIn('<option value="single">Single column (89 mm)</option>', html)

    def test_camera_exports_svg_and_600_ppi_png(self):
        source = PLOT_STYLE.read_text(encoding="utf-8")
        self.assertIn("const PNG_DPI = 600", source)
        self.assertIn("single: 89, double: 183", source)
        self.assertIn("format: 'svg'", source)
        self.assertIn("format: 'png'", source)
        self.assertIn("Plotly.toImage(holder", source)
        self.assertIn("downloadDataUrl(svgUrl", source)
        self.assertIn("downloadDataUrl(pngUrl", source)
        self.assertIn("'toImage'", source)
        self.assertIn("Download publication SVG + PNG (600 ppi)", source)

    def test_shared_theme_uses_publication_typography_and_axes(self):
        source = PLOT_STYLE.read_text(encoding="utf-8")
        self.assertIn("Arial, Helvetica, sans-serif", source)
        self.assertIn("const SCREEN = Object.freeze({ base: 14, axisTitle: 16, plotTitle: 16", source)
        self.assertIn("const PRINT = Object.freeze({ basePt: 8, axisTitlePt: 9, plotTitlePt: 10", source)
        self.assertIn("const SINGLE_PRINT = Object.freeze({ basePt: 7.5, axisTitlePt: 8.5, plotTitlePt: 9.5", source)
        self.assertIn("ticks: axis.ticks === '' ? '' : 'outside'", source)
        self.assertIn("mirror: false", source)
        self.assertIn("orientation === 'y'", source)

    def test_single_column_has_an_independent_compact_layout(self):
        source = PLOT_STYLE.read_text(encoding="utf-8")
        self.assertIn("Math.max(0.9, sourceRatio)", source)
        self.assertIn("orientation: 'h'", source)
        self.assertIn("borderwidth: singleColumn ? 0 : 0.7", source)
        self.assertIn("nticks: axis.nticks || (singleColumn ? 5 : undefined)", source)
        self.assertIn("clampSingleColumnMarkerSize", source)
        self.assertIn("graphDiv.layout || {}, dimensions.width, dimensions.height, column, xAxisTitle", source)

    def test_publication_export_restores_hidden_stacked_x_axis(self):
        source = PLOT_STYLE.read_text(encoding="utf-8")
        self.assertIn("function publicationXAxisTitle(graphDiv)", source)
        self.assertIn("graphId.replace(/Up$/, 'Dw')", source)
        self.assertIn("visible: orientation === 'x' ? true : axis.visible", source)
        self.assertIn("showticklabels: orientation === 'x' ? true : axis.showticklabels", source)
        self.assertIn("const xAxisTitle = publicationXAxisTitle(graphDiv)", source)

    def test_archive_bulk_export_uses_publication_exporter(self):
        html = (ROOT / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertNotIn("Plotly.downloadImage", html)
        self.assertIn("MigaPlotStyle.exportPublicationPlot", html)


if __name__ == "__main__":
    unittest.main()
