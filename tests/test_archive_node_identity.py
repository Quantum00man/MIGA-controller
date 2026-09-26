import csv
import json
import tempfile
from pathlib import Path

from app.core.data_loader import DataLoader


def _write_run(path: Path, role: str, node_id: str, value: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "mode": "transfer_function",
        "sync_role": role,
        "sync_node_id": node_id,
        "scan_dimensions": 1,
    }), encoding="utf-8")
    with (path / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Step", "Parameter_P0", "Atom_UP"])
        writer.writeheader()
        writer.writerow({"Step": 0, "Parameter_P0": 1, "Atom_UP": value})
    (path / "sequence.mot").write_text(f"{role}:{node_id}\n", encoding="utf-8")


def test_master_layout_returns_resolved_node_identity_and_selected_data():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "2026" / "09" / "23" / "run01"
        _write_run(root, "master", "MIGA22", 22)
        _write_run(root / "sync_nodes" / "slave-a", "slave", "slave-a", 21)
        (root / "sync_manifest.json").write_text(json.dumps({
            "runtime": {"sync_role": "master", "slaves": [{"node_id": "slave-a", "name": "MIGA21"}]},
            "archive_nodes": {
                "master": {"role": "master", "local": True, "path": "."},
                "slave-a": {"role": "slave", "local": False, "path": "sync_nodes/slave-a"},
            },
            "pairs": [],
        }), encoding="utf-8")
        loader = DataLoader()
        loader.base_dir = Path(tmp)

        master = loader.load_run("2026", "09", "23", "run01", node_id="master")
        slave = loader.load_run("2026", "09", "23", "run01", node_id="slave-a")

        assert master["archive_node_identity"]["resolved_node"] == "master"
        assert master["archive_node_identity"]["relative_path"] == "."
        assert master["selected_sync_node"] == "master"
        assert master["data"][0]["atom_number_up"] == 22
        assert slave["archive_node_identity"]["resolved_node"] == "slave-a"
        assert slave["archive_node_identity"]["relative_path"] == "sync_nodes/slave-a"
        assert slave["data"][0]["atom_number_up"] == 21
        master_mot, _ = loader.get_archived_sequence_file("2026", "09", "23", "run01", node_id="master")
        slave_mot, _ = loader.get_archived_sequence_file("2026", "09", "23", "run01", node_id="slave-a")
        assert master_mot.read_text(encoding="utf-8") == "master:MIGA22\n"
        assert slave_mot.read_text(encoding="utf-8") == "slave:slave-a\n"


def test_slave_layout_defaults_to_local_slave_and_can_resolve_replicated_master():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "2026" / "09" / "23" / "run02"
        _write_run(root, "slave", "slave-a", 21)
        _write_run(root / "sync_nodes" / "master", "master", "MIGA22", 22)
        (root / "sync_manifest.json").write_text(json.dumps({
            "runtime": {"sync_role": "master", "slaves": [{"node_id": "slave-a"}]},
            "archive_nodes": {
                "master": {"role": "master", "local": False, "path": "sync_nodes/master"},
                "slave-a": {"role": "slave", "local": True, "path": "."},
            },
            "pairs": [],
        }), encoding="utf-8")
        loader = DataLoader()
        loader.base_dir = Path(tmp)

        local = loader.load_run("2026", "09", "23", "run02")
        master = loader.load_run("2026", "09", "23", "run02", node_id="master")

        assert local["selected_sync_node"] == "slave-a"
        assert local["archive_node_identity"]["local"] is True
        assert local["data"][0]["atom_number_up"] == 21
        assert master["selected_sync_node"] == "master"
        assert master["data"][0]["atom_number_up"] == 22


def test_slave_bragg_archive_recovers_mode_from_replicated_master_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "2026" / "09" / "23" / "run05"
        _write_run(root, "slave", "slave-a", 21)
        _write_run(root / "sync_nodes" / "master", "master", "MIGA22", 22)
        slave_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        slave_config["mode"] = "standard"
        (root / "config.json").write_text(json.dumps(slave_config), encoding="utf-8")
        master_config = json.loads((root / "sync_nodes" / "master" / "config.json").read_text(encoding="utf-8"))
        master_config.update({
            "mode": "bragg_fringe_calibration",
            "_bragg_calibration_coarse_shots": 8,
            "bragg_calibration_target_fringe": 1,
        })
        (root / "sync_nodes" / "master" / "config.json").write_text(
            json.dumps(master_config), encoding="utf-8"
        )
        (root / "sync_manifest.json").write_text(json.dumps({
            "runtime": {"sync_role": "master", "slaves": [{"node_id": "slave-a"}]},
            "archive_nodes": {
                "master": {"role": "master", "local": False, "path": "sync_nodes/master"},
                "slave-a": {"role": "slave", "local": True, "path": "."},
            },
            "pairs": [],
        }), encoding="utf-8")
        loader = DataLoader()
        loader.base_dir = Path(tmp)

        loaded = loader.load_run("2026", "09", "23", "run05")

        assert loaded["config"]["mode"] == "bragg_fringe_calibration"
        assert loaded["config"]["_archive_mode_recovered_from_master"] is True
        assert loaded["bragg_fringe_calibration"]["status"] == "reanalysis_failed"


def test_legacy_sync_run_keeps_root_as_master():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "2026" / "09" / "23" / "run03"
        _write_run(root, "master", "MIGA22", 22)
        (root / "sync_manifest.json").write_text(json.dumps({
            "runtime": {"sync_run_id": "legacy", "sync_role": "master", "slaves": [{"node_id": "slave-a"}]},
            "pairs": [],
        }), encoding="utf-8")
        loader = DataLoader()
        loader.base_dir = Path(tmp)

        loaded = loader.load_run("2026", "09", "23", "run03")

        assert loaded["selected_sync_node"] == "master"
        assert loaded["archive_node_identity"]["legacy_layout"] is True
        assert loaded["archive_node_identity"]["relative_path"] == "."


def test_legacy_slave_run_infers_local_slave_and_replicated_master():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "2026" / "09" / "23" / "run04"
        _write_run(root, "slave", "MIGA21", 21)
        _write_run(root / "sync_nodes" / "master", "master", "MIGA22", 22)
        (root / "sync_manifest.json").write_text(json.dumps({
            "runtime": {"sync_run_id": "legacy", "slaves": [{"node_id": "slave-a", "name": "MIGA21"}]},
            "pairs": [],
        }), encoding="utf-8")
        loader = DataLoader()
        loader.base_dir = Path(tmp)

        local = loader.load_run("2026", "09", "23", "run04")
        master = loader.load_run("2026", "09", "23", "run04", node_id="master")

        assert local["selected_sync_node"] == "slave-a"
        assert local["archive_node_identity"]["local"] is True
        assert local["data"][0]["atom_number_up"] == 21
        assert master["archive_node_identity"]["relative_path"] == "sync_nodes/master"
        assert master["data"][0]["atom_number_up"] == 22


def test_standalone_run_on_slave_role_loads_from_run_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "2026" / "09" / "26" / "run09_20260926"
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({
            "mode": "standard",
            "scan_dimensions": 1,
            "sync_role": "slave",
            "_system_settings_snapshot": {
                "sync_role": "slave",
                "sync_node_name": "MIGA21",
            },
        }), encoding="utf-8")
        (root / "results.csv").write_text(
            "Step,Parameter,Atom_Up\n1,0,21\n", encoding="utf-8"
        )
        (root / "sequence.mot").write_text("standalone slave run\n", encoding="utf-8")
        loader = DataLoader()
        loader.base_dir = Path(tmp)

        loaded = loader.load_run("2026", "09", "26", "run09_20260926")
        sequence, _ = loader.get_archived_sequence_file(
            "2026", "09", "26", "run09_20260926"
        )

        assert loaded["config"]["mode"] == "standard"
        assert loaded["sync_manifest"] is None
        assert sequence.read_text(encoding="utf-8") == "standalone slave run\n"


def test_archive_ui_uses_loaded_points_and_rolls_back_failed_node_switch():
    html = (Path(__file__).parents[1] / "static" / "archive.html").read_text(encoding="utf-8")

    assert "res.data.archive_node_identity?.resolved_node" in html
    assert "String(source || '') === String(this.loadedSyncNode || '')" in html
    assert "return this.loadedSyncNodeData;" in html
    assert "const loaded = await this.loadRun(source);" in html
    assert "await this.loadRun(previous);" in html
    assert "exportLoadedSequence(node.id)" in html
    assert "syncArchiveSequenceNodes()" in html
    assert "requestedNode ? `?node_id=${encodeURIComponent(requestedNode)}`" in html
