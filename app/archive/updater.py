"""Hardware-free, conservative code updates for the archive host only."""
import os
import subprocess
import threading
import re
from pathlib import Path
from app.archive.backup import now


class ArchiveUpdater:
    def __init__(self, configuration, backup, root=None):
        self.configuration, self.backup = configuration, backup
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]
        self.lock = threading.Lock()
        self.reload_controller = None

    def git(self, *args):
        env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_SSH_COMMAND': 'ssh -o BatchMode=yes -o ConnectTimeout=10'}
        try:
            result = subprocess.run(['git', *args], cwd=self.root, env=env,
                                    capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError('Git update failed or timed out; check repository access and retry') from exc
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip() or 'Git command failed'
            if 'insufficient permission' in message or 'Permission denied' in message:
                message += '\nThe server user cannot write repository metadata/files. Inspect ownership in the server terminal; do not use sudo git or chmod 777.'
            raise ValueError(message)
        return result.stdout.strip()

    def branch(self, value):
        value = str(value or '').strip()
        if not value or value.startswith('-'):
            raise ValueError('Enter a valid branch name')
        self.git('check-ref-format', '--branch', value)
        return value

    def optional_git(self, *args):
        try: return self.git(*args)
        except ValueError: return ''

    def repository_url(self, value):
        value = str(value or '').strip()
        if not re.fullmatch(r'(?:https://github\.com/|git@github\.com:)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?',value):
            raise ValueError('Repository must be a GitHub HTTPS or SSH URL without credentials, query parameters or extra arguments')
        return value

    def status(self, selected_branch=None):
        current = self.git('branch', '--show-current')
        config = self.configuration.load()
        configured_branch = config.get('update_branch') or current or 'main'
        branch = self.branch(selected_branch or configured_branch)
        refs = self.git('for-each-ref', '--format=%(refname:strip=3)', 'refs/remotes/origin').splitlines()
        local_branches = self.git('for-each-ref', '--format=%(refname:strip=2)', 'refs/heads').splitlines()
        dirty = self.git('status', '--porcelain')
        origin = self.optional_git('remote','get-url','origin')
        remote = 'refs/remotes/origin/'+branch
        remote_commit = self.optional_git('rev-parse','--verify',remote+'^{commit}')
        local = 'HEAD' if current == branch else 'refs/heads/'+branch
        local_commit = self.optional_git('rev-parse','--verify',local+'^{commit}')
        counts = self.optional_git('rev-list','--left-right','--count',local+'...'+remote) if remote_commit and local_commit else ''
        ahead, behind = map(int,counts.split()) if counts else (None,None)
        if not remote_commit:
            state, message = 'unknown','Remote comparison unavailable. Refresh Remote to discover branches and commits.'
        elif current != branch:
            state, message = 'branch_mismatch',f'Selected {branch}; current checkout is {current or "detached HEAD"}. Update Now will switch branches.'
        elif not counts:
            state, message = 'unknown','Cannot compare commit history. Refresh Remote and inspect repository access before updating.'
        elif ahead and behind:
            state, message = 'diverged',f'Local history diverged: {ahead} ahead, {behind} behind. Resolve in the terminal; updates will not overwrite it.'
        elif behind:
            state, message = 'behind',f'Update available: {behind} commit(s) behind origin/{branch}.'
        elif ahead:
            state, message = 'ahead',f'Local checkout is {ahead} commit(s) ahead. Local commits are preserved; automatic updates are blocked.'
        else:
            state, message = 'latest',f'Checkout matches the cached origin/{branch} commit.'
        return {'current_branch': current, 'configured_branch': configured_branch,
                'current_commit': self.git('rev-parse', '--short', 'HEAD'),
                'current_commit_short':self.git('rev-parse','--short','HEAD'),
                'last_commit_subject':self.git('log','-1','--pretty=%s'),
                'repo_root':str(self.root), 'origin_url':origin,
                'configured_repo_url':config.get('update_repo_url') or origin,
                'branches': sorted(set([current, branch, configured_branch,*local_branches, *[r for r in refs if r != 'HEAD']]) - {''}),
                'dirty': bool(dirty), 'dirty_entries': dirty.splitlines(),
                'auto_reload_after_update':config.get('auto_reload_after_update',False),
                'last_remote_fetch':config.get('update_last_fetch'),
                'comparison_branch':branch, 'comparison_ahead':ahead, 'comparison_behind':behind,
                'comparison_remote_commit_short':remote_commit[:12],
                'comparison_remote_subject':self.optional_git('log','-1','--pretty=%s',remote) if remote_commit else '',
                'version_status':state,'version_message':message,'is_latest':state=='latest',
                'restart_required': False}

    def save_branch(self, branch, repo_url=None):
        with self.configuration.lock:
            config = self.configuration.load()
            config['update_branch'] = branch
            if repo_url is not None:
                config['update_repo_url'] = repo_url
                config['update_last_fetch'] = now()
            self.configuration._save(config)

    def run(self, branch, apply=False, repo_url=None):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Another update operation is already running')
        try:
            branch = self.branch(branch)
            repo_url = self.repository_url(repo_url) if repo_url else None
            # Prevent a queued/manual/scheduled backup from starting during checkout.
            with self.backup.lock:
                if getattr(self.backup,'maintenance',False):
                    raise ValueError('Reload pending; wait for the new server')
                if any(job['status'] in {'running', 'queued'} for job in self.backup.jobs()):
                    raise ValueError('Wait for all queued/running backups before updating')
                if apply and self.git('status', '--porcelain'):
                    raise ValueError('Local changes detected. Commit or preserve them before updating; nothing was overwritten')
                auto_reload = apply and self.configuration.load().get('auto_reload_after_update',False)
                if auto_reload:
                    if not self.reload_controller:
                        raise ValueError('Automatic reload is unavailable')
                    self.reload_controller.available()
                original_origin = self.optional_git('remote','get-url','origin')
                if repo_url and repo_url != original_origin:
                    if not original_origin:
                        raise ValueError('Origin is missing; configure it in the server terminal first')
                    self.git('remote','set-url','origin',repo_url)
                try:
                    self.git('fetch', '--prune', 'origin')
                except ValueError:
                    if repo_url and repo_url != original_origin:
                        self.git('remote','set-url','origin',original_origin)
                    raise
                remote = 'refs/remotes/origin/' + branch
                try:
                    target = self.git('rev-parse', '--verify', remote + '^{commit}')
                except ValueError as exc:
                    if repo_url and repo_url != original_origin:
                        self.git('remote','set-url','origin',original_origin)
                    raise ValueError('Selected branch was not found on origin; refresh branches or choose another') from exc
                if not apply:
                    self.save_branch(branch,repo_url or original_origin)
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
                self.save_branch(branch,repo_url or original_origin)
                result = self.status()
                result.update(restart_required=True, pull_output=output,
                              message='Code updated. Restart via LaunchUI to load it. NAS data and SSH configuration were not changed.')
                if auto_reload:
                    try:
                        result['reload'] = self.reload_controller.schedule()
                        result.update(restart_required=False,message='Code updated. Safe reload scheduled; this page will reconnect automatically.')
                    except ValueError as exc:
                        result['message'] = 'Code updated but reload could not start: '+str(exc)+' Restart manually via LaunchUI.'
                return result
        finally:
            self.lock.release()
