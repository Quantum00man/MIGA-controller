import csv
import io
import lzma
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from app.core.phase_noise_export import collect, csv_zip, labplot


class Loader:
    def __init__(self, root):
        self.root = Path(root)
        for node in ('master', 'slave1'):
            (self.root / node).mkdir()
            (self.root / node / 'results.csv').touch()

    def _get_run_dir(self, *ref): return self.root
    def _load_sync_manifest_payload(self, root):
        return {'archive_nodes': {'master': {}, 'slave1': {'name': 'Slave One'}},
                'runtime': {'slaves': [{'node_id': 'slave1'}, {'node_id': 'missing'}]},
                'node_results': {'master': [{'phase_noise_science_shot': 1, 'sync_shot_index': 12}]}}
    def _resolve_archive_node_identity(self, root, node):
        path = root / node
        if not path.exists(): raise FileNotFoundError(node)
        return path, {}
    def _load_config_data(self, path): return {'mode': 'phase_noise'}
    def _read_results_csv(self, path, max_points=None):
        assert max_points is None
        return [{'step': 1, 'phase_noise_repeat': 1, 'phase_noise_science_shot': 1, 'phase_noise_t2_us2': 100},
                {'step': 2, 'phase_noise_repeat': 2, 'phase_noise_science_shot': 2, 'phase_noise_t2_us2': 100},
                {'step': 3, 'intf_alpha_calibration_block_id': 1}]
    def recalculate_run(self, *ref, max_points, node_id, tolerate_missing_waveforms):
        assert max_points is None
        assert tolerate_missing_waveforms
        return {'data': [{'step': 1, 'atom_number_up': 20, 'temperature_up': 3,
                          'phase_noise_repeat': 1, 'interferometer_phase': .1}]}


def test_full_export_keeps_rejected_shots_and_warns_for_missing_nodes():
    with TemporaryDirectory() as root:
        ref = ('2026', '11', '03', 'run1')
        rows, warnings = collect(Loader(root), ref, {}, None)
        assert len(rows) == 6
        assert rows[0]['sync_shot_index'] == 12
        assert rows[1]['reanalysis_status'] == 'rejected'
        assert rows[1]['atom_number_up'] is None
        assert rows[2]['shot_type'] == 'calibration'
        assert any('missing' in warning for warning in warnings)
        with zipfile.ZipFile(io.BytesIO(csv_zip(rows, warnings, ref))) as archive:
            assert set(archive.namelist()) == {'shots.csv', 'statistics.csv', 'README.txt'}
            shots = list(csv.DictReader(io.StringIO(archive.read('shots.csv').decode('utf-8-sig'))))
            assert len(shots) == 6 and shots[0]['atom_number_up'] == '20'
            stats = list(csv.DictReader(io.StringIO(archive.read('statistics.csv').decode('utf-8-sig'))))
            atom = next(row for row in stats if row['node_id'] == 'master' and row['metric'] == 'atom_number_up')
            assert atom['valid_count'] == '1' and atom['sample_std'] == ''
        project = ElementTree.fromstring(lzma.decompress(labplot(rows, warnings, ref)))
        assert any('Atom number UP' in item.attrib.get('name', '') for item in project.iter('worksheet'))
