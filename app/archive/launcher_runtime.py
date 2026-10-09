"""Local-only Archive launcher and explicitly approved SSH bootstrap.

This module uses the standard library so a fresh Linux desktop can repair its venv.
Passwords are handled by sudo/ssh-copy-id in a native terminal, never by the web API.
"""
import argparse
import base64
import fcntl
import json
import os
import pwd
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get('MIGA_ARCHIVE_CONFIG', '~/.config/miga-archive/config.json')).expanduser()
_spawned = {}


def login_user():
    return pwd.getpwuid(os.getuid()).pw_name


def read_json(path):
    value = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object in ' + str(path))
    return value


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def python():
    candidate = ROOT / '.venv/bin/python'
    return str(candidate) if candidate.is_file() else sys.executable


def managed_process(state):
    pid = state.get('pid')
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        start = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')',1)[1].split()[19]
        return b'archive_main:app' in args and b'uvicorn' in args and (Path('/proc') / str(pid) / 'cwd').resolve() == ROOT and (not state.get('start_ticks') or state['start_ticks'] == start)
    except OSError:
        return False


def service_info():
    if not shutil.which('systemctl'):
        return {}
    try:
        result = subprocess.run(['systemctl', '--user', 'show', 'miga-archive.service', '--property=ActiveState', '--property=WorkingDirectory', '--property=Environment', '--property=LoadState', '--property=ExecStart'], capture_output=True, text=True, timeout=3)
        fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        environment = shlex.split(fields.get('Environment', ''))
        configured = next((value.split('=',1)[1] for value in environment if value.startswith('MIGA_ARCHIVE_CONFIG=')), str(Path.home()/'.config/miga-archive/config.json'))
        if fields.get('LoadState') != 'loaded' or fields.get('WorkingDirectory') != str(ROOT) or Path(configured).expanduser().resolve() != CONFIG.resolve() or 'archive_main:app' not in fields.get('ExecStart', ''):
            return {}
        port = re.search(r'--port\s+(\d+)', fields.get('ExecStart', ''))
        return {'state':fields.get('ActiveState'), 'port':int(port.group(1)) if port else 8765}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def service_active():
    return service_info().get('state') in {'active', 'activating', 'deactivating'}


def external_process():
    """Recognize a foreground server with the same project/config, even on another port."""
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid() or (entry / 'cwd').resolve() != ROOT:
                continue
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if b'archive_main:app' not in args or b'uvicorn' not in args:
                continue
            # Only the configuration path is inspected/returned; no environment is logged.
            config = next((value.split(b'=',1)[1].decode() for value in (entry / 'environ').read_bytes().split(b'\0') if value.startswith(b'MIGA_ARCHIVE_CONFIG=')), str(Path.home()/'.config/miga-archive/config.json'))
            if Path(config).expanduser().resolve() != CONFIG.resolve():
                continue
            port = int(args[args.index(b'--port')+1]) if b'--port' in args else 8000
            return {'pid':int(entry.name), 'port':port}
        except (OSError, ValueError, IndexError):
            continue
    return None


def status(port=8765):
    for pid, child in list(_spawned.items()):
        if child.poll() is not None:
            _spawned.pop(pid)
    state = read_json(CONFIG.parent / 'launcher/runtime.json')
    service = service_info()
    result = {'running': False, 'responding': False, 'managed': False, 'service': service.get('state') in {'active','activating','deactivating'}, 'service_state':service.get('state'), 'port':service.get('port', port),
              'log': str(CONFIG.parent / 'launcher/server.log'), 'configured': bool(read_json(CONFIG).get('archive_root'))}
    if managed_process(state):
        result.update(running=True, managed=True, port=state['port'], pid=state['pid'], stopping=state.get('stop_requested', False))
    else:
        existing = external_process()
        if existing:
            result.update(running=True, **existing)
    if service.get('state') == 'deactivating': result['stopping'] = True
    try:
        with urllib.request.urlopen('http://127.0.0.1:' + str(result['port']) + '/archive-server/status', timeout=2) as response:
            info = json.load(response)
            if info.get('mode') == 'archive':
                result.update(running=True, responding=True, configured=info['configured'])
    except (OSError, ValueError):
        pass
    return result


def repair():
    candidate = ROOT / '.venv/bin/python'
    if not candidate.exists() or subprocess.run([str(candidate), '-m', 'pip', '--version'], capture_output=True).returncode != 0:
        subprocess.run([getattr(sys, '_base_executable', sys.executable), '-m', 'venv', str(ROOT / '.venv')], check=True)
    subprocess.run([str(candidate), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-archive.txt')], check=True)
    check()


def check():
    subprocess.run([python(), '-c', 'import fastapi,uvicorn,pydantic,numpy,scipy,requests; from app.core.data_loader import DataLoader; print("Archive environment OK (no Ax/hardware required)")'], cwd=ROOT, check=True)
    print('SSH tools:', ', '.join(name + (' OK' if shutil.which(name) else ' missing') for name in ['ssh', 'ssh-keygen', 'ssh-keyscan', 'ssh-copy-id', 'rsync']))


def server_tools():
    print('Install Python venv support, the OpenSSH CLIENT and rsync. No SSH daemon or firewall change is made.')
    if input('Type INSTALL TOOLS to approve: ').strip() != 'INSTALL TOOLS':
        raise ValueError('Cancelled')
    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'python3-venv', 'openssh-client', 'rsync'], check=True)


def gui_tools():
    print('LaunchUI needs Ubuntu Python Tk and venv support. No server, SSH or NAS settings are changed.')
    if input('Type INSTALL GUI to approve installation: ').strip() != 'INSTALL GUI':
        raise ValueError('GUI dependency installation cancelled')
    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'python3-tk', 'python3-venv'], check=True)


def start(port=8765):
    if os.geteuid() == 0:
        raise ValueError('Run Archive Server as your normal Linux user, not root')
    directory = CONFIG.parent / 'launcher'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = status(port)
        if current.get('stopping'):
            raise ValueError('Server is still stopping; wait for its current run to finish before starting again')
        if current['running'] or current['service']:
            print('An Archive Server is already running; existing process/configuration retained')
            return
        check()
        service = service_info()
        if service:
            subprocess.run(['systemctl', '--user', 'start', '--no-block', 'miga-archive.service'], check=True)
            for _ in range(60):
                if status(service['port'])['responding']:
                    print(f"Archive user service ready on port {service['port']}")
                    return
                time.sleep(.2)
            raise ValueError('User service is not responding yet. Check its journal in LaunchUI; do not start a second process.')
        with socket.socket() as sock:
            sock.bind(('0.0.0.0', port))
        with (directory / 'server.log').open('ab') as output:
            process = subprocess.Popen([python(), '-m', 'uvicorn', 'archive_main:app', '--host', '0.0.0.0', '--port', str(port)], cwd=ROOT,
                env={**os.environ, 'MIGA_ARCHIVE_CONFIG': str(CONFIG)}, stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True)
        _spawned[process.pid] = process
        ticks = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')',1)[1].split()[19]
        save_json(directory / 'runtime.json', {'pid': process.pid, 'port': port, 'start_ticks':ticks})
        for _ in range(60):
            if process.poll() is not None:
                raise ValueError('Server exited during startup; see server.log')
            if status(port)['running']:
                # PID existence alone does not imply HTTP readiness.
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/archive-server/status', timeout=1):
                        print(f'Archive Server ready: http://127.0.0.1:{port}/')
                        return
                except OSError:
                    pass
            time.sleep(.2)
        raise ValueError('Server is starting but not yet responding; inspect its log before retrying')


def stop(port=8765):
    directory = CONFIG.parent / 'launcher'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _stop_unlocked(port)


def _stop_unlocked(port=8765):
    if service_active():
        subprocess.run(['systemctl', '--user', 'stop', '--no-block', 'miga-archive.service'], check=True)
        print('Service shutdown requested; current run may take time to finish')
        return
    state = read_json(CONFIG.parent / 'launcher/runtime.json')
    if managed_process(state):
        if state.get('stop_requested'):
            print('Shutdown already requested; no second signal sent. Wait for the current run.')
            return
        os.kill(state['pid'], signal.SIGINT)
        state['stop_requested'] = True
        save_json(CONFIG.parent / 'launcher/runtime.json', state)
        print('Graceful stop requested once. Wait for the current run to finish; no forced kill.')
    elif status(port)['running']:
        raise ValueError('This server was started outside LaunchUI. Stop it in its original terminal; LaunchUI will not kill an unknown process.')
    else:
        print('Archive Server is not running')


def ensure_key(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    public = path.with_suffix(path.suffix + '.pub')
    if not path.exists():
        if public.exists():
            raise ValueError('Public key exists without its private key; preserve it and choose another identity')
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(path), '-C', 'miga-archive'], check=True)
    derived = subprocess.run(['ssh-keygen', '-y', '-f', str(path)], capture_output=True, text=True, check=True).stdout.strip()
    if public.exists():
        if ' '.join(public.read_text().split()[:2]) != ' '.join(derived.split()[:2]):
            raise ValueError('Existing private/public key do not match; no files replaced')
    else:
        public.write_text(derived + '\n')
    return public


def pair(device_id):
    device = read_json(CONFIG).get('devices', {}).get(device_id)
    if not device or device.get('transport', 'ssh') != 'ssh':
        raise ValueError('Register an SSH device in the web Guide first')
    host, user = device['host'], device['ssh_user']
    if host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9._:-]+', host) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', user):
        raise ValueError('Invalid host/user')
    identity = Path(device.get('identity_file') or '~/.ssh/miga_archive_ed25519').expanduser()
    ssh = ['ssh', '-i', str(identity), '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
    target = user + '@' + host
    probe = subprocess.run(ssh + ['-o', 'BatchMode=yes', target, 'true'], capture_output=True)
    if probe.returncode == 0:
        print('SSH already works. Existing key and host trust retained; no changes needed.')
        return
    public = ensure_key(identity)
    known = Path.home() / '.ssh/known_hosts'
    lookup = subprocess.run(['ssh-keygen', '-F', host, '-f', str(known)], capture_output=True, text=True)
    if lookup.returncode != 0:
        scan = subprocess.run(['ssh-keyscan', '-T', '10', '-t', 'ed25519', host], capture_output=True, text=True, timeout=20, check=True)
        lines = [line for line in scan.stdout.splitlines() if not line.startswith('#') and len(line.split()) == 3]
        if not lines or any(line.split()[0] != host or line.split()[1] != 'ssh-ed25519' for line in lines):
            raise ValueError('Cannot obtain an unambiguous host key')
        with tempfile.NamedTemporaryFile(mode='w+') as temp:
            temp.write('\n'.join(lines) + '\n'); temp.flush()
            fingerprint = subprocess.run(['ssh-keygen', '-lf', temp.name], capture_output=True, text=True, check=True).stdout
        fingerprints = {line.split()[1] for line in fingerprint.splitlines()}
        if len(fingerprints) != 1:
            raise ValueError('Multiple different host keys found; verify the host manually')
        expected = next(iter(fingerprints))
        print('Compare this host fingerprint with the SOURCE LaunchUI. Do not trust a scan alone.\n' + fingerprint)
        if input('Type the matching SHA256 fingerprint to trust this host (blank cancels): ').strip() != expected:
            raise ValueError('Host trust not confirmed; known_hosts unchanged')
        known.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with known.open('a') as stream:
            stream.write('\n' + '\n'.join(lines) + '\n')
        known.chmod(0o600)
    print('Authorize this dedicated key on the source. Your password is handled only by ssh-copy-id and is not saved.')
    subprocess.run(['ssh-copy-id', '-i', str(public), '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', target], check=True)
    subprocess.run(ssh + ['-o', 'BatchMode=yes', target, 'true'], check=True)
    print('Pairing complete. Return to web Guide and click Test / Discover.')


def validate_source(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or path in {Path('/'), Path.home().resolve(), ROOT}:
        raise ValueError('Choose the specific Data_log directory, not a home or project root')
    if path.name.lower() != 'data_log' and not any(child.is_dir() and re.fullmatch(r'\d{4}', child.name) for child in path.iterdir()):
        raise ValueError('Choose Data_log or an archive folder containing year directories, not a general system folder')
    return path


def source_prepare(path, grant_read=False):
    path = validate_source(path)
    if os.geteuid() == 0:
        raise ValueError('Use your normal Linux login; the helper asks sudo only for specific actions')
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'ssh'], capture_output=True).returncode == 0
    ready = active and shutil.which('rsync') and shutil.which('setfacl') and Path('/etc/ssh/ssh_host_ed25519_key.pub').is_file()
    if ready:
        print('Existing SSH, rsync and ACL tools detected. No installation or service changes needed.')
    else:
        print('Will install OpenSSH server, rsync and ACL tools, and enable SSH. No archive data is changed.')
        if input('Type ENABLE SSH to approve: ').strip() != 'ENABLE SSH':
            raise ValueError('Cancelled')
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'openssh-server', 'rsync', 'acl'], check=True)
        subprocess.run(['sudo', 'systemctl', 'enable', '--now', 'ssh'], check=True)
    if grant_read:
        user = login_user()
        print('Grant recursive read/traverse ACL to', user, 'only within', path)
        if input('Type GRANT READ to approve: ').strip() != 'GRANT READ':
            raise ValueError('ACL changes cancelled; SSH remains enabled')
        acl = subprocess.run(['sudo', 'getfacl', '-R', '-P', '--omit-header', str(path)], capture_output=True, text=True, check=True).stdout
        if any(line.startswith('user:' + user + ':') and 'w' in line.split(':', 2)[2].split()[0] for line in acl.splitlines()):
            raise ValueError('Existing writable named ACLs detected. Read-only ACL repair skipped to avoid removing write access; request targeted permission repair from your administrator.')
        subprocess.run(['sudo', 'setfacl', '-R', '-P', '-m', 'u:' + user + ':rX', str(path)], check=True)
    print('Source ready. Host fingerprint:')
    subprocess.run(['ssh-keygen', '-lf', '/etc/ssh/ssh_host_ed25519_key.pub'], check=True)


def validate_public_key(public_key):
    fields = public_key.strip().split()
    if len(fields) < 2 or fields[0] != 'ssh-ed25519' or '\n' in public_key.strip() or '\r' in public_key:
        raise ValueError('Paste a single ssh-ed25519 PUBLIC key, not a private key')
    decoded = base64.b64decode(fields[1], validate=True)
    if len(decoded) != 51 or decoded[:19] != b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20':
        raise ValueError('Invalid Ed25519 public key')
    return fields[:2]


def authorize(public_key):
    fields = validate_public_key(public_key)
    if os.geteuid() == 0:
        raise ValueError('Authorize the key as the source data owner, not root')
    directory = Path.home() / '.ssh'
    if directory.is_symlink() or (directory / 'authorized_keys').is_symlink():
        raise ValueError('Linked SSH directories/files need manual review; no authorization file changed')
    directory.mkdir(mode=0o700, exist_ok=True)
    file = directory / 'authorized_keys'
    existing = file.read_text() if file.exists() else ''
    if any(fields[1] in line.split() for line in existing.splitlines() if not line.lstrip().startswith('#')):
        print('Key already authorized; existing restrictions retained')
        return
    print('This key permits remote commands as', login_user(), 'with the account’s existing filesystem permissions. Forwarding/PTY are disabled; this is not a read-only account.')
    if input('Type AUTHORIZE to approve: ').strip() != 'AUTHORIZE':
        raise ValueError('Cancelled')
    with file.open('a') as stream:
        stream.write('\nrestrict ' + ' '.join(fields[:2]) + ' miga-archive\n')
    directory.chmod(0o700); file.chmod(0o600)
    print('Public key authorized; all existing authorized_keys entries preserved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status', 'check', 'repair', 'pair', 'source-prepare', 'server-tools', 'gui-tools', 'authorize', 'public-key', 'service-install', 'service-log', 'linger'])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device-id'); parser.add_argument('--path'); parser.add_argument('--public-key'); parser.add_argument('--grant-read', action='store_true')
    parser.add_argument('--pause', action='store_true', help='Keep native terminal open until Enter')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if os.geteuid() == 0 and args.action not in {'status', 'check', 'stop', 'service-log'}:
        raise ValueError('Open LaunchUI as your normal Linux user; approve individual sudo actions in its terminal')
    if args.action in {'start', 'stop', 'status'}:
        result = globals()[args.action](args.port)
        if result: print(json.dumps(result))
    elif args.action in {'check', 'repair'}: globals()[args.action]()
    elif args.action == 'pair': pair(args.device_id)
    elif args.action == 'source-prepare': source_prepare(args.path or str(ROOT / 'Data_log'), args.grant_read)
    elif args.action == 'server-tools': server_tools()
    elif args.action == 'gui-tools': gui_tools()
    elif args.action == 'authorize': authorize(args.public_key or '')
    elif args.action == 'public-key': print(ensure_key(Path.home() / '.ssh/miga_archive_ed25519').read_text())
    elif args.action == 'service-install':
        subprocess.run([python(), '-m', 'app.archive.service_install'], cwd=ROOT,
            env={**os.environ, 'MIGA_ARCHIVE_CONFIG': str(CONFIG), 'MIGA_ARCHIVE_HOST': '0.0.0.0', 'MIGA_ARCHIVE_PORT': str(args.port)}, check=True)
    elif args.action == 'service-log':
        if not service_info(): raise ValueError('No Archive user service matches this project/configuration')
        subprocess.run(['journalctl', '--user', '-u', 'miga-archive.service', '-n', '150', '--no-pager'], check=True)
    elif args.action == 'linger': subprocess.run(['sudo', 'loginctl', 'enable-linger', login_user()], check=True)


if __name__ == '__main__':
    code = 0
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('Setup failed:', exc, file=sys.stderr)
        code = 1
    finally:
        if '--pause' in sys.argv:
            try: input('\nPress Enter to close this terminal and return to LaunchUI…')
            except EOFError: pass
    raise SystemExit(code)
