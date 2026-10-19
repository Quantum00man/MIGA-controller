"""Read-only cross-device SYNC inputs, pinned into server analysis versions on write."""
from copy import deepcopy
from pathlib import Path
import json
from urllib.parse import urlparse
import re

from app.archive.backup import atomic_json


def find_links(repository, root, manifest):
    """Match only an exact recorded run, SYNC identity and slave identity."""
    runtime = manifest.get('runtime') or {}
    master = json.loads((root / 'config.json').read_text())
    sync_id = master.get('sync_run_id') or runtime.get('sync_run_id')
    if not sync_id:
        return {}, []
    links, warnings = {}, []
    devices = repository.configuration.load().get('devices') or {}
    for slave in runtime.get('slaves') or []:
        node, run_id = str(slave.get('node_id') or ''), str(slave.get('run_id') or '')
        if not node or not re.fullmatch(r'[A-Za-z0-9_.-]+', node) or node in {'.', '..'}:
            continue
        existing = (manifest.get('archive_nodes') or {}).get(node)
        if existing and (root / str(existing.get('path') or '.')).is_dir():
            continue
        if not re.fullmatch(r'run[A-Za-z0-9_.-]+_\d{8}', run_id) or '..' in run_id:
            continue
        date = run_id[-8:]
        ref = [date[:4], date[4:6], date[6:], run_id]
        candidates = []
        for device, settings in devices.items():
            try:
                source = repository.reference(device, *ref)
                source.resolve().relative_to((repository.device_root(device) / 'runs').resolve())
                raw = json.loads((source / 'config.json').read_text())
            except (FileNotFoundError, ValueError, OSError):
                continue
            snapshot = raw.get('_system_settings_snapshot') or {}
            identity = str(raw.get('sync_node_id') or snapshot.get('sync_node_id') or '')
            host_matches = bool(settings.get('host')) and settings['host'] == urlparse(slave.get('base_url') or '').hostname
            identity_matches = identity in {node, str(slave.get('name') or '')} if identity else False
            if raw.get('sync_run_id') != sync_id or not (identity_matches or host_matches):
                continue
            if str(raw.get('sync_role') or snapshot.get('sync_role') or '').lower() != 'slave':
                continue
            if not (source / 'results.csv').is_file():
                continue
            candidates.append({'device_id': device, 'reference': ref, 'source': source})
        if len(candidates) == 1:
            links[node] = candidates[0]
        elif len(candidates) > 1:
            warnings.append(f'Multiple matching archives for {node}; no automatic association was made')
    return links, warnings


def attach_sync_links(loader, repository):
    """Install server-only resolution; controller path checks remain unchanged."""
    read_manifest = loader._load_sync_manifest_payload
    resolve_node = loader._resolve_archive_node_identity
    cache = {}

    def linked(root):
        root = Path(root)
        if root not in cache:
            manifest = read_manifest(root)
            links, warnings = find_links(repository, root, manifest) if manifest else ({}, [])
            cache[root] = (manifest, links, warnings)
        return cache[root]

    def manifest(root):
        original, links, warnings = linked(root)
        if not original:
            return original
        result = deepcopy(original)
        for node, link in links.items():
            result.setdefault('archive_nodes', {})[node] = {
                'node_id': node, 'role': 'slave', 'local': False,
                'source': 'server_archive', 'device_id': link['device_id'],
                'reference': link['reference'],
            }
        if warnings:
            result['server_archive_link_warnings'] = warnings
        return result

    def resolve(root, node=None):
        try:
            return resolve_node(root, node)
        except FileNotFoundError:
            link = linked(root)[1].get(str(node or ''))
            if not link:
                raise
            source = link['source']
            # Read source data directly; mutable requests pin a private copy first.
            return source, {'requested_node': node, 'resolved_node': node,
                'relative_path': '', 'role': 'slave', 'local': False,
                'legacy_layout': False, 'warnings': [], 'source': 'server_archive',
                'device_id': link['device_id'], 'reference': link['reference']}

    loader._load_sync_manifest_payload = manifest
    loader._resolve_archive_node_identity = resolve


def pin_sync_links(loader, repository, root, clone):
    """Persist independent inputs inside the new Master analysis version only."""
    manifest = loader._load_sync_manifest_payload(root)
    if not manifest:
        return
    links, warnings = find_links(repository, root, manifest)
    if warnings:
        manifest['server_archive_link_warnings'] = warnings
    for node, link in links.items():
        relative = f'sync_nodes/{node}'
        target = root / relative
        if target.exists():
            raise ValueError(f'Archive association destination already exists for {node}')
        from app.core.analysis_store import input_stamp, stamp_digest
        source_stamp = input_stamp(link['source'])
        clone(link['source'], target, link['source'], share_waveforms=False)
        if source_stamp != input_stamp(link['source']):
            raise ValueError(f'Slave archive changed while copying {node}; retry after backup completes')
        manifest.setdefault('archive_nodes', {})[node] = {
            'node_id': node, 'role': 'slave', 'local': False, 'path': relative,
            'source': 'server_archive_snapshot', 'device_id': link['device_id'],
            'reference': link['reference'], 'source_version': 'original',
            'source_stamp_sha256': stamp_digest(source_stamp),
        }
    if links or warnings:
        atomic_json(root / 'sync_manifest.json', manifest)
