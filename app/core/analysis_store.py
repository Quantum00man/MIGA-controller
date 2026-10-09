"""Independent, immutable analysis results. Acquisition directories are inputs only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid


def input_files(directory):
    # Follow the immutable waveform references used by UI versions.
    for root, directories, filenames in os.walk(directory, followlinks=True):
        directories.sort()
        for filename in sorted(filenames):
            yield Path(root) / filename


def input_stamp(directory):
    return [(path.relative_to(directory).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
            for path in input_files(directory)]


def stamp_digest(stamp):
    return hashlib.sha256(json.dumps(stamp, separators=(',', ':')).encode()).hexdigest()


def fingerprint(directory):
    digest = hashlib.sha256()
    for path in input_files(directory):
        digest.update(path.relative_to(directory).as_posix().encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def software_commit():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


class AnalysisStore:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, group, identifier):
        if not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('Invalid analysis identifier')
        return self.root / group / (identifier + '.json')

    def _write(self, group, record):
        path = self._path(group, record['id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return record

    def read(self, identifier, group='saved'):
        record = json.loads(self._path(group, identifier).read_text(encoding='utf-8'))
        if record.get('id') != identifier:
            raise ValueError('Invalid analysis record')
        return record

    def candidate(self, kind, result, parameters, source):
        record = {'schema_version': 1, 'id': uuid.uuid4().hex, 'kind': kind,
                  'computed_at': datetime.now(timezone.utc).isoformat(), 'parameters': parameters,
                  'source': source, 'software_commit': software_commit(), 'algorithm_version': 1,
                  'result': result}
        # Unsaved calculations are temporary; named snapshots are retained.
        for path in (self.root / 'drafts').glob('*.json'):
            if path.stat().st_mtime < time.time() - 7 * 86400:
                path.unlink(missing_ok=True)
        self._write('drafts', record)
        return record['id']

    def save(self, candidate_id, name, note='', view=None, input_sha256=None):
        if not name.strip():
            raise ValueError('Analysis name is required')
        try:
            record = self.read(candidate_id, 'drafts')
        except FileNotFoundError:
            record = self.read(candidate_id)
        view = dict(view or {})
        plot = view.pop('plot_snapshot', None)
        if plot is not None:
            if record['kind'] != 'phase_noise_allan' or not isinstance(plot, dict):
                raise ValueError('Plot snapshot is only supported for Phase Noise Allan')
            if input_sha256 != record['source']['input_sha256'] and plot != record['result'].get('plot_snapshot'):
                raise ValueError('Input changed since calculation; recompute before saving a new plot')
            record['result']['plot_snapshot'] = plot
        record.update(id=uuid.uuid4().hex, candidate_id=candidate_id, name=name.strip(), note=note,
                      created_at=datetime.now(timezone.utc).isoformat(), view=view)
        return self._write('saved', record)

    def list(self):
        records = []
        for path in (self.root / 'saved').glob('*.json'):
            record = self.read(path.stem)
            records.append({key: value for key, value in record.items() if key != 'result'})
        return sorted(records, key=lambda record: record['created_at'], reverse=True)


def store_for(loader, reference):
    provider = getattr(loader, 'analysis_directory', None)
    if provider:
        return AnalysisStore(provider(reference) / 'results')
    from app.archive.controller_context import ControllerRepository
    return AnalysisStore(ControllerRepository(loader.base_dir).analysis_dir('controller', reference) / 'results')


def capture_inputs(loader, req):
    reference = [req.year, req.month, req.day, req.run_id]
    selected = loader.get_run_dir(*reference)
    original_provider = getattr(loader, 'original_directory', None)
    original = original_provider(reference) if original_provider else Path(loader.base_dir).joinpath(*reference)
    stamp = input_stamp(selected)
    raw_hash = fingerprint(original)
    selected_hash = raw_hash if selected == original else fingerprint(selected)
    if stamp != input_stamp(selected):
        raise ValueError('Input changed while reading; wait for acquisition/backup to finish and retry')
    return {'reference': reference, 'device_id': getattr(loader, 'analysis_device_id', 'controller'), 'node_id': req.node_id,
            'version': selected.parent.name if selected != original else 'original',
            'raw_sha256': raw_hash, 'input_sha256': selected_hash,
            '_stamp': stamp, '_directory': str(selected)}


def capture_result(loader, req, result, settings, kind, source):
    source = dict(source)
    stamp = source.pop('_stamp')
    directory = source.pop('_directory')
    if stamp != input_stamp(Path(directory)):
        raise ValueError('Input changed during calculation; wait for acquisition/backup to finish and retry')
    parameters = req.model_dump(exclude={'result', 'name', 'note', 'view', 'updated_data'})
    parameters['new_settings'] = settings
    candidate_id = store_for(loader, source['reference']).candidate(kind, result, parameters, source)
    return {**result, 'analysis_candidate_id': candidate_id}
