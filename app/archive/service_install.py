"""Install a systemd user service using the current user's NAS/key permissions."""
import os
from pathlib import Path
import subprocess
import sys


def install():
    project = Path(__file__).resolve().parents[2]
    config = Path(os.environ.get('MIGA_ARCHIVE_CONFIG', '~/.config/miga-archive/config.json')).expanduser()
    host = os.environ.get('MIGA_ARCHIVE_HOST', '0.0.0.0')
    port = int(os.environ.get('MIGA_ARCHIVE_PORT', '8765'))
    # systemd has its own quoting and %-specifier syntax.
    def quote(value):
        return '"' + str(value).replace('%', '%%').replace('\\', '\\\\').replace('"', '\\"') + '"'
    units = Path.home() / '.config/systemd/user'
    units.mkdir(parents=True, exist_ok=True)
    unit = units / 'miga-archive.service'
    contents = '\n'.join([
        '[Unit]', 'Description=MIGA Archive Server', 'After=network.target', '',
        '[Service]', 'Type=simple', 'WorkingDirectory=' + quote(project),
        'Environment=' + quote('MIGA_ARCHIVE_CONFIG=' + str(config)),
        'ExecStart=' + quote(sys.executable) + ' -m uvicorn archive_main:app --host ' + host + ' --port ' + str(port),
        'Restart=on-failure', 'RestartSec=10', 'UMask=0077', '', '[Install]', 'WantedBy=default.target', '',
    ])
    if unit.exists():
        unit.with_suffix('.service.previous').write_bytes(unit.read_bytes())
    temp = unit.with_suffix('.tmp')
    temp.write_text(contents)
    temp.replace(unit)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', 'miga-archive'], check=True)
    print('Installed. Stop any foreground server, then run: systemctl --user start miga-archive')
    print('For startup before login: sudo loginctl enable-linger ' + os.environ.get('USER', 'miga'))


if __name__ == '__main__':
    install()
