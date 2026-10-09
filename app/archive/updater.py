"""Hardware-free, conservative code updates for the archive host only."""
import os
import subprocess
import threading
from pathlib import Path


class ArchiveUpdater:
    def __init__(self, configuration, backup, root=None):
        self.configuration, self.backup = configuration, backup
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]
        self.lock = threading.Lock()

    def git(self, *args):
        env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_SSH_COMMAND': 'ssh -o BatchMode=yes -o ConnectTimeout=10'}
        try:
            result = subprocess.run(['git', *args], cwd=self.root, env=env,
                                    capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError('Git update failed or timed out; check repository access and retry') from exc
        if result.returncode:
            raise ValueError(result.stderr.strip() or result.stdout.strip() or 'Git command failed')
        return result.stdout.strip()

    def branch(self, value):
        value = str(value or '').strip()
        if not value or value.startswith('-'):
            raise ValueError('Enter a valid branch name')
        self.git('check-ref-format', '--branch', value)
        return value

    def status(self):
        current = self.git('branch', '--show-current')
        branch = self.configuration.load().get('update_branch') or current
        refs = self.git('for-each-ref', '--format=%(refname:strip=3)', 'refs/remotes/origin').splitlines()
        dirty = self.git('status', '--porcelain')
        return {'current_branch': current, 'configured_branch': branch,
                'current_commit': self.git('rev-parse', '--short', 'HEAD'),
                'branches': sorted(set([current, branch, *[r for r in refs if r != 'HEAD']]) - {''}),
                'dirty': bool(dirty), 'dirty_entries': dirty.splitlines(),
                'restart_required': False}

    def save_branch(self, branch):
        with self.configuration.lock:
            config = self.configuration.load()
            config['update_branch'] = branch
            self.configuration._save(config)

    def run(self, branch, apply=False):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Another update operation is already running')
        try:
            branch = self.branch(branch)
            # Prevent a queued/manual/scheduled backup from starting during checkout.
            with self.backup.lock:
                if any(job['status'] in {'running', 'queued'} for job in self.backup.jobs()):
                    raise ValueError('Wait for all queued/running backups before updating')
                if apply and self.git('status', '--porcelain'):
                    raise ValueError('Local changes detected. Commit or preserve them before updating; nothing was overwritten')
                self.git('fetch', '--prune', 'origin')
                remote = 'refs/remotes/origin/' + branch
                try:
                    target = self.git('rev-parse', '--verify', remote + '^{commit}')
                except ValueError as exc:
                    raise ValueError('Selected branch was not found on origin; refresh branches or choose another') from exc
                if not apply:
                    self.save_branch(branch)
                    result = self.status()
                    result['remote_commit'] = target[:12]
                    result['message'] = 'Remote branches refreshed; selected update branch saved.'
                    return result
                current = self.git('branch', '--show-current')
                if not current:
                    raise ValueError('Detached HEAD: select a local branch in the terminal before updating')
                local = self.git('for-each-ref', '--format=%(refname:strip=2)', 'refs/heads').splitlines()
                if branch in local:
                    # Validate target branch before switching; never reset divergent history.
                    try:
                        self.git('merge-base', '--is-ancestor', 'refs/heads/' + branch, target)
                    except ValueError as exc:
                        raise ValueError('Target branch has local commits or diverged history; resolve in the terminal. No checkout was changed') from exc
                    if current != branch:
                        self.git('switch', branch)
                else:
                    self.git('switch', '--create', branch, '--track', 'origin/' + branch)
                output = self.git('merge', '--ff-only', target)
                self.save_branch(branch)
                result = self.status()
                result.update(restart_required=True, pull_output=output,
                              message='Code updated. Restart via LaunchUI to load it. NAS data and SSH configuration were not changed.')
                return result
        finally:
            self.lock.release()
