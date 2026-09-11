import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_HTML = ROOT / "static" / "archive.html"


class ArchiveRunOrderingTests(unittest.TestCase):
    def test_archive_uses_numeric_run_id_ordering(self):
        source = ARCHIVE_HTML.read_text(encoding="utf-8")
        comparator = "String(a.id).localeCompare(String(b.id), undefined, { numeric: true })"
        self.assertIn(comparator, source)

    def test_numeric_order_keeps_three_digit_runs_after_run99(self):
        script = """
const ids = ['run98_20260911', 'run100_20260911', 'run99_20260911', 'run101_20260911'];
ids.sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
process.stdout.write(ids.join(','));
"""
        result = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout,
            "run98_20260911,run99_20260911,run100_20260911,run101_20260911",
        )


if __name__ == "__main__":
    unittest.main()
