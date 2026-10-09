"""Desktop archive launcher. Privileged/password actions stay in a native terminal."""
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from app.archive.launcher_runtime import ROOT, CONFIG, read_json, python, login_user


def terminal_command(arguments):
    command = [python(), '-m', 'app.archive.launcher_runtime', *arguments, '--pause']
    if shutil.which('gnome-terminal'):
        return ['gnome-terminal', '--wait', '--', *command]
    if shutil.which('xterm'):
        return ['xterm', '-e', *command]
    raise ValueError('Install gnome-terminal or xterm to show secure sudo/SSH prompts')


class ArchiveLauncherFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=18)
        self.queue = queue.Queue(); self.busy = False; self.polling = False; self.open_when_ready = False
        try: preferences = read_json(CONFIG.parent / 'launcher/preferences.json')
        except (OSError, ValueError): preferences = {}
        self.port = tk.StringVar(value=str(preferences.get('port', 8765)))
        self.device = tk.StringVar(); self.source_path = tk.StringVar(value=str(ROOT / 'Data_log'))
        self.grant_read = tk.BooleanVar(value=False)
        try: auto_reload = read_json(CONFIG).get('auto_reload_after_update',False)
        except (OSError,ValueError): auto_reload = False
        self.auto_reload = tk.BooleanVar(value=auto_reload)
        self.status_text = tk.StringVar(value='Ready. Existing configuration will be reused.')
        ttk.Label(self, text='Archive Server — no hardware or root startup', font=('', 16, 'bold')).pack(anchor='w')
        ttk.Label(self, text='Reuse existing NAS/devices/SSH keys. The backup engine is unchanged.').pack(anchor='w', pady=(4, 12))
        tabs = ttk.Notebook(self); tabs.pack(fill='both', expand=True)
        server = ttk.Frame(tabs, padding=14); source = ttk.Frame(tabs, padding=14)
        tabs.add(server, text='Archive Server'); tabs.add(source, text='Prepare this source computer')
        row = ttk.Frame(server); row.pack(fill='x')
        ttk.Label(row, text='Port').pack(side='left'); ttk.Entry(row, textvariable=self.port, width=8).pack(side='left', padx=8)
        ttk.Label(row, text='Lab LAN bind: 0.0.0.0 • normal Linux user').pack(side='left')
        ttk.Checkbutton(server,text='Auto reload after successful web update',variable=self.auto_reload,
            command=lambda:self.run('reload-preference','--enabled','1' if self.auto_reload.get() else '0')).pack(anchor='w',pady=(10,0))
        ttk.Label(server,text='Explicit updates only; no file watching. Backups must be idle. The browser reconnects after a short interruption.').pack(anchor='w')
        ttk.Label(server, textvariable=self.status_text, wraplength=1050).pack(anchor='w', pady=12)
        actions = ttk.Frame(server); actions.pack(fill='x')
        for index, (label, command) in enumerate([
            ('Check environment', lambda: self.run('check')),
            ('Repair environment', self.repair), ('Start server', lambda: self.run('start', open_guide=True)),
            ('Stop gracefully', self.stop), ('Reload server safely', self.reload_server), ('Open setup Guide', lambda: self.open_page('setup')),
            ('Open management', lambda: self.open_page('')), ('Refresh status', self.refresh),
            ('Enable startup service', self.service), ('Enable before-login startup', self.linger),
            ('Install Linux prerequisites', lambda: self.run('server-tools', native=True)),
        ]):
            actions.columnconfigure(index % 3, weight=1)
            ttk.Button(actions, text=label, command=command).grid(row=index // 3, column=index % 3, sticky='ew', padx=4, pady=4)
        ttk.Separator(server).pack(fill='x', pady=12)
        ttk.Label(server, text='SSH pairing — register devices in the web Guide, then select one here', font=('', 11, 'bold')).pack(anchor='w')
        pairing = ttk.Frame(server); pairing.pack(fill='x', pady=6)
        self.device_select = ttk.Combobox(pairing, textvariable=self.device, state='readonly', width=24)
        self.device_select.pack(side='left')
        ttk.Button(pairing, text='Reload devices', command=self.reload_devices).pack(side='left', padx=6)
        ttk.Button(pairing, text='Pair / reuse existing SSH', command=self.pair).pack(side='left')
        ttk.Button(pairing, text='Show dedicated public key', command=self.public_key).pack(side='left', padx=6)
        ttk.Label(server, text='First-time host trust must be compared with the source fingerprint. SSH passwords remain in the terminal; they are never logged here.', wraplength=1050).pack(anchor='w')
        self.output = tk.Text(server, height=14, wrap='word'); self.output.pack(fill='both', expand=True, pady=12)
        ttk.Button(server, text='Show latest server log', command=self.show_log).pack(anchor='w')
        ttk.Label(source, text='Run this tab ON the Master/Slave computer, logged in as its data owner.', font=('', 12, 'bold')).pack(anchor='w')
        ttk.Label(source, text=f'This computer: {socket.gethostname()} • Linux user: {login_user()}').pack(anchor='w', pady=6)
        ttk.Button(source, text='Show local IP addresses', command=self.source_addresses).pack(anchor='w')
        ttk.Label(source, text='1. Prepare SSH locally (one sudo confirmation). 2. Compare host fingerprints. 3. Authorize the server key or use server-side pairing.', wraplength=1050).pack(anchor='w', pady=8)
        path_row = ttk.Frame(source); path_row.pack(fill='x')
        ttk.Label(path_row, text='Data_log').pack(side='left')
        ttk.Entry(path_row, textvariable=self.source_path).pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(path_row, text='Choose folder', command=self.choose_source).pack(side='left')
        ttk.Checkbutton(source, text='Also grant my user read/traverse access within this Data_log (optional; separate confirmation)', variable=self.grant_read).pack(anchor='w', pady=8)
        ttk.Button(source, text='Prepare SSH / rsync', command=self.prepare_source).pack(anchor='w')
        ttk.Button(source, text='Show this source host fingerprint', command=self.fingerprint).pack(anchor='w', pady=8)
        ttk.Label(source, text='Optional: paste the Archive Server PUBLIC key to authorize locally. Existing keys are kept.').pack(anchor='w')
        self.public_input = tk.Text(source, height=3, wrap='word'); self.public_input.pack(fill='x', pady=8)
        ttk.Button(source, text='Authorize public key (confirm in terminal)', command=self.authorize).pack(anchor='w')
        ttk.Label(source, text='Authorization permits remote commands as your Linux account. This is not a filesystem-enforced read-only account; the backup application itself only reads source data.', wraplength=1050).pack(anchor='w', pady=12)
        self.reload_devices(); self.refresh(); self.after(200, self.drain); self.after(2500, self.poll_status)

    def poll_status(self):
        if self.winfo_ismapped() and not self.busy: self.refresh()
        self.after(2500, self.poll_status)

    def arguments(self, action, *extra):
        port = int(self.port.get())
        if not 1 <= port <= 65535: raise ValueError('Port must be between 1 and 65535')
        return [action, '--port', str(port), *extra]

    def run(self, action, *extra, native=False, open_guide=False):
        if self.busy: messagebox.showinfo('Busy', 'Wait for the current launcher action.'); return
        try:
            args = self.arguments(action, *extra)
            command = terminal_command(args) if native else [python(), '-m', 'app.archive.launcher_runtime', *args]
            from app.archive.launcher_runtime import save_json
            save_json(CONFIG.parent / 'launcher/preferences.json', {'port': int(self.port.get())})
        except (ValueError, OSError) as exc:
            messagebox.showerror('Archive launcher', str(exc)); return
        self.busy = True
        self.output.insert('end', '\n' + action + (' — follow the secure terminal prompts' if native else '') + '\n')
        def worker():
            try:
                if native:
                    result = subprocess.run(command, cwd=ROOT)
                else:
                    result = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in result.stdout: self.queue.put(('output', line.rstrip()))
                    result.wait()
                self.queue.put(('done', result.returncode, open_guide, action))
            except OSError as exc:
                self.queue.put(('output', str(exc))); self.queue.put(('done', 1, False, action))
        threading.Thread(target=worker, daemon=True).start()

    def drain(self):
        while not self.queue.empty():
            item = self.queue.get()
            if item[0] == 'output': self.output.insert('end', item[1] + '\n'); self.output.see('end')
            elif item[0] == 'status':
                self.polling = False
                state = item[1]
                self.status_text.set(('Stopping — waiting for current run' if state.get('stopping') else 'Running' if state.get('responding') else 'Starting / not yet responding' if state.get('running') else 'Stopped') +
                    f" • port {state.get('port', self.port.get())} • " + ('existing setup found' if state.get('configured') else 'setup needed') +
                    (' • systemd service' if state.get('service') else ''))
                if state.get('running'): self.port.set(str(state['port']))
                if state.get('responding') and not self.busy:
                    self.auto_reload.set(state.get('auto_reload_after_update',False))
                if state.get('reload_state',{}).get('status') in {'scheduled','restarting'}:
                    self.status_text.set('Reloading safely — wait for the server to reconnect')
                elif state.get('reload_state',{}).get('status') == 'failed':
                    self.status_text.set(self.status_text.get()+' • Reload failed; inspect launcher/reload.log or service journal')
                if self.open_when_ready and state.get('responding'):
                    self.open_when_ready = False; self.open_page('')
            elif item[0] == 'done':
                self.busy = False
                if item[1] == 0:
                    if item[2]: self.open_when_ready = True
                    if item[3] == 'public-key': self.show_public_key()
                else: self.output.insert('end', 'Action failed. Review the output/terminal, fix the issue and retry.\n')
                self.refresh()
        self.after(200, self.drain)

    def refresh(self):
        if self.polling: return
        try: port = int(self.port.get())
        except ValueError: return
        self.polling = True
        def worker():
            from app.archive.launcher_runtime import status
            try: self.queue.put(('status', status(port)))
            except (OSError, ValueError): self.queue.put(('status', {}))
        threading.Thread(target=worker, daemon=True).start()

    def open_page(self, page):
        try: port = int(self.port.get())
        except ValueError: return
        webbrowser.open(f'http://127.0.0.1:{port}/{page}')

    def repair(self):
        if messagebox.askyesno('Repair Archive environment', 'Install Archive-only Python dependencies into this project’s .venv? No Ax/hardware setup is needed.'):
            self.run('repair')

    def stop(self):
        if messagebox.askyesno('Stop Archive Server', 'Request graceful shutdown? The current run may take time to finish. Do not start another server until it exits.'):
            self.run('stop')

    def reload_server(self):
        if messagebox.askyesno('Reload Archive Server','Reload code through a graceful restart? Queued/running backups block this action. No forced kill or duplicate server will be used.'):
            self.run('reload')

    def service(self):
        if messagebox.askyesno('Enable startup service', 'Install/enable a systemd USER service for this project and port? This preserves current backups and replaces an existing unit with a .previous copy. It does not stop/start the currently running server.'):
            self.run('service-install')

    def linger(self):
        if messagebox.askyesno('Before-login startup', 'Enable linger for your Linux user? Its user services can then run before login and after logout. sudo authorization will appear in a terminal.'):
            self.run('linger', native=True)

    def reload_devices(self):
        try: records = read_json(CONFIG).get('devices', {})
        except (OSError, ValueError): records = {}
        self.device_select['values'] = [key for key, row in records.items() if row.get('transport', 'ssh') == 'ssh']
        if not self.device.get() and self.device_select['values']: self.device.set(self.device_select['values'][0])

    def pair(self):
        if not self.device.get(): messagebox.showinfo('Register first', 'Register the source in the web Guide, then reload devices here.'); return
        if messagebox.askyesno('SSH pairing', 'Reuse working SSH, or install a dedicated key into the selected source account? First-time trust requires fingerprint comparison; the account’s existing permissions are retained. Passwords are only entered in the terminal.'):
            self.run('pair', '--device-id', self.device.get(), native=True)

    def public_key(self):
        if messagebox.askyesno('Dedicated public key', 'Reuse the existing dedicated identity or generate it if missing? No existing key is replaced.'):
            self.run('public-key')

    def show_public_key(self):
        file = Path.home() / '.ssh/miga_archive_ed25519.pub'
        if file.exists():
            window = tk.Toplevel(self); window.title('Archive Server PUBLIC key — copy to source')
            text = tk.Text(window, width=100, height=4); text.pack(padx=12, pady=12); text.insert('1.0', file.read_text())

    def choose_source(self):
        folder = filedialog.askdirectory(initialdir=ROOT)
        if folder: self.source_path.set(folder)

    def prepare_source(self):
        self.run('source-prepare', '--path', self.source_path.get(), *(['--grant-read'] if self.grant_read.get() else []), native=True)

    def fingerprint(self):
        try:
            result = subprocess.run(['ssh-keygen', '-lf', '/etc/ssh/ssh_host_ed25519_key.pub'], capture_output=True, text=True, timeout=5)
            messagebox.showinfo('Source host fingerprint', result.stdout or 'SSH host key missing. Prepare SSH first.\n' + result.stderr)
        except OSError as exc: messagebox.showerror('Fingerprint', str(exc))

    def source_addresses(self):
        try:
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=5)
            messagebox.showinfo('Source connection details', f'Linux user: {login_user()}\nHost: {socket.gethostname()}\nAddresses: {result.stdout.strip() or "Unavailable"}\nData_log: {self.source_path.get()}\nUse the lab LAN address in server registration.')
        except OSError as exc: messagebox.showerror('Addresses', str(exc))

    def authorize(self):
        from app.archive.launcher_runtime import validate_public_key
        try:
            public = ' '.join(validate_public_key(self.public_input.get('1.0', 'end').strip()))
        except ValueError as exc:
            messagebox.showerror('Public key required', str(exc)); return
        self.run('authorize', '--public-key', public, native=True)

    def show_log(self):
        from app.archive.launcher_runtime import service_info
        if service_info():
            self.run('service-log'); return
        log = CONFIG.parent / 'launcher/server.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 24000)); self.output.insert('end', stream.read().decode(errors='replace'))
            self.output.see('end')
