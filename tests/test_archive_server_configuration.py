import tempfile
from pathlib import Path

from app.archive.configuration import ArchiveServerConfiguration


def test_archive_server_initialization_and_device_registration_are_idempotent():
    with tempfile.TemporaryDirectory() as archive, tempfile.TemporaryDirectory() as state:
        config = ArchiveServerConfiguration(Path(state) / "config.json")
        initialized = config.initialize(archive)
        repeated = config.initialize(archive)
        assert initialized["archive_uuid"] == repeated["archive_uuid"]
        assert (Path(archive) / "incoming").is_dir()
        assert (Path(archive) / "derived").is_dir()

        device = config.register_device({
            "device_id": "master", "name": "MIGA Master", "role": "master",
            "transport": "ssh", "host": "192.168.1.10", "ssh_user": "miga",
            "source_path": "/home/miga/MIGA-controller/Data_log",
        })
        assert device["device_id"] == "master"
        assert config.load()["devices"]["master"]["role"] == "master"
        assert (Path(archive) / "devices" / "master" / "runs").is_dir()
        assert (Path(archive) / "devices" / "master" / "device.json").is_file()


def test_archive_server_rejects_unsafe_or_incomplete_device_identity():
    with tempfile.TemporaryDirectory() as archive, tempfile.TemporaryDirectory() as state:
        config = ArchiveServerConfiguration(Path(state) / "config.json")
        config.initialize(archive)
        try:
            config.register_device({"device_id": "../master", "source_path": "/tmp", "transport": "local"})
        except ValueError as exc:
            assert "Device ID" in str(exc)
        else:
            raise AssertionError("Unsafe device ID was accepted")

        try:
            config.register_device({"device_id": "slave-01", "source_path": "relative", "transport": "ssh"})
        except ValueError as exc:
            assert "absolute" in str(exc)
        else:
            raise AssertionError("Relative source path was accepted")
