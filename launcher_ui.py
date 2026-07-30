#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

ROOT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT_DIR / 'start_controller.sh'
SETTINGS_PATH = ROOT_DIR / 'user_settings.json'
STATE_DIR = ROOT_DIR / '.launcher_state'
LOG_FILE = STATE_DIR / 'server.log'
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = '8000'
DEFAULT_RELOAD = '1'
TAIL_BYTES = 24000


class LauncherUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MIGA Controller Launcher')
        self.geometry('1260x860')
        self.minsize(1120, 760)
        self.configure(bg='#dfe6ec')

        self.command_queue = queue.Queue()
        self.task_running = False
        self.last_log_snapshot = ''
        self.launch_options_initialized = False
        self.launch_vars = {
            'port': tk.StringVar(value=DEFAULT_PORT),
            'reload': tk.BooleanVar(value=DEFAULT_RELOAD != '0'),
        }
        self.status_vars = {
            'server': tk.StringVar(value='Unknown'),
            'pid': tk.StringVar(value='-'),
            'venv': tk.StringVar(value='Pending'),
            'env_check': tk.StringVar(value='Not checked'),
            'auth': tk.StringVar(value=self._detect_auth_mode()),
            'host': tk.StringVar(value=DEFAULT_HOST),
            'port': tk.StringVar(value=DEFAULT_PORT),
            'reload': tk.StringVar(value='enabled'),
            'tmot': tk.StringVar(value='Not configured'),
            'cmot': tk.StringVar(value='Not configured'),
            'settings': tk.StringVar(value=str(SETTINGS_PATH)),
        }

        self._build_style()
        self._build_layout()
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.after(200, self._drain_queue)
        self.after(300, self.refresh_everything)
        self.after(1500, self._poll_runtime)

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Root.TFrame', background='#dfe6ec')
        style.configure('Card.TFrame', background='#f8fbfd', relief='flat')
        style.configure('Header.TLabel', background='#dfe6ec', foreground='#15314b', font=('Noto Sans', 19, 'bold'))
        style.configure('SubHeader.TLabel', background='#dfe6ec', foreground='#48657d', font=('Noto Sans', 10))
        style.configure('Section.TLabel', background='#f8fbfd', foreground='#204564', font=('Noto Sans', 11, 'bold'))
        style.configure('Key.TLabel', background='#f8fbfd', foreground='#506575', font=('Noto Sans', 10))
        style.configure('Value.TLabel', background='#f8fbfd', foreground='#14293e', font=('Noto Sans', 10, 'bold'))
        style.configure('Hint.TLabel', background='#f8fbfd', foreground='#6a7f91', font=('Noto Sans', 9))
        style.configure('Primary.TButton', font=('Noto Sans', 10, 'bold'))
        style.configure('Secondary.TButton', font=('Noto Sans', 10))
        style.configure('Card.TCheckbutton', background='#f8fbfd', foreground='#14293e', font=('Noto Sans', 10))

    def _build_layout(self):
        root = ttk.Frame(self, style='Root.TFrame', padding=16)
        root.pack(fill='both', expand=True)

        header = ttk.Frame(root, style='Root.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text='MIGA Controller Launcher', style='Header.TLabel').pack(anchor='w')
        ttk.Label(
            header,
            text='UI launcher for venv checks, Ax environment repair, root startup, and live server log tracking.',
            style='SubHeader.TLabel',
        ).pack(anchor='w', pady=(2, 10))

        body = ttk.Panedwindow(root, orient='horizontal')
        body.pack(fill='both', expand=True)

        left = ttk.Frame(body, style='Root.TFrame', padding=(0, 0, 12, 0))
        right = ttk.Frame(body, style='Root.TFrame')
        body.add(left, weight=1)
        body.add(right, weight=2)

        self._build_status_panel(left)
        self._build_log_panel(right)

    def _build_status_panel(self, parent: ttk.Frame):
        runtime = self._card(parent)
        runtime.pack(fill='x', pady=(0, 12))
        ttk.Label(runtime, text='Runtime', style='Section.TLabel').pack(anchor='w')
        self._kv(runtime, 'Server', self.status_vars['server'])
        self._kv(runtime, 'PID', self.status_vars['pid'])
        self._kv(runtime, 'Virtual Env', self.status_vars['venv'])
        self._kv(runtime, 'Ax Check', self.status_vars['env_check'])
        self._kv(runtime, 'Privilege Mode', self.status_vars['auth'])
        self._kv(runtime, 'Host', self.status_vars['host'])
        self._kv(runtime, 'Port', self.status_vars['port'])
        self._kv(runtime, 'Reload', self.status_vars['reload'])

        hardware = self._card(parent)
        hardware.pack(fill='x', pady=(0, 12))
        ttk.Label(hardware, text='Hardware Paths', style='Section.TLabel').pack(anchor='w')
        self._kv(hardware, 'tmot', self.status_vars['tmot'])
        self._kv(hardware, 'cmot', self.status_vars['cmot'])
        self._kv(hardware, 'Settings File', self.status_vars['settings'])

        launch = self._card(parent)
        launch.pack(fill='x', pady=(0, 12))
        ttk.Label(launch, text='Launch Options', style='Section.TLabel').pack(anchor='w')

        host_row = ttk.Frame(launch, style='Card.TFrame')
        host_row.pack(fill='x', pady=(10, 4))
        ttk.Label(host_row, text='Host:', style='Key.TLabel').pack(side='left')
        ttk.Label(host_row, text=DEFAULT_HOST, style='Value.TLabel').pack(side='left', padx=(8, 0))

        port_row = ttk.Frame(launch, style='Card.TFrame')
        port_row.pack(fill='x', pady=4)
        ttk.Label(port_row, text='Port:', style='Key.TLabel').pack(side='left')
        self.port_entry = ttk.Entry(port_row, textvariable=self.launch_vars['port'], width=10)
        self.port_entry.pack(side='left', padx=(8, 0))
        ttk.Label(port_row, text='Default 8000', style='Hint.TLabel').pack(side='left', padx=(10, 0))

        reload_row = ttk.Frame(launch, style='Card.TFrame')
        reload_row.pack(fill='x', pady=(6, 0))
        ttk.Checkbutton(
            reload_row,
            text='Enable uvicorn --reload',
            variable=self.launch_vars['reload'],
            style='Card.TCheckbutton',
        ).pack(anchor='w')
        ttk.Label(reload_row, text='Default is enabled. Disable it for a more stable hardware run.', style='Hint.TLabel').pack(anchor='w', pady=(4, 0))

        actions = self._card(parent)
        actions.pack(fill='x')
        ttk.Label(actions, text='Actions', style='Section.TLabel').pack(anchor='w')

        button_grid = ttk.Frame(actions, style='Card.TFrame')
        button_grid.pack(fill='x', pady=(10, 0))
        button_grid.columnconfigure(0, weight=1)
        button_grid.columnconfigure(1, weight=1)

        buttons = [
            ('Refresh', self.refresh_everything, 'Secondary.TButton'),
            ('Check / Repair Env', lambda: self.run_launcher(['--check-only']), 'Primary.TButton'),
            ('Start Controller', self.start_controller, 'Primary.TButton'),
            ('Stop Controller', lambda: self.run_launcher(['--stop'], elevated=True), 'Secondary.TButton'),
            ('Open Main UI', lambda: self._open_browser_page('index.html'), 'Secondary.TButton'),
            ('Open Optimize UI', lambda: self._open_browser_page('optimize.html'), 'Secondary.TButton'),
        ]

        for index, (label, command, style) in enumerate(buttons):
            row = index // 2
            column = index % 2
            padx = (0, 6) if column == 0 else (6, 0)
            pady = (0, 6) if row < 2 else (0, 0)
            ttk.Button(
                button_grid,
                text=label,
                command=command,
                style=style,
            ).grid(row=row, column=column, sticky='ew', padx=padx, pady=pady)

    def _build_log_panel(self, parent: ttk.Frame):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)

        output_tab = ttk.Frame(notebook, style='Root.TFrame', padding=6)
        log_tab = ttk.Frame(notebook, style='Root.TFrame', padding=6)
        notebook.add(output_tab, text='Action Output')
        notebook.add(log_tab, text='Server Log')

        self.output_text = tk.Text(
            output_tab,
            wrap='word',
            bg='#101820',
            fg='#d8f0ff',
            insertbackground='#d8f0ff',
            font=('DejaVu Sans Mono', 10),
            relief='flat',
        )
        self.output_text.pack(fill='both', expand=True)
        self.output_text.configure(state='disabled')

        self.server_log_text = tk.Text(
            log_tab,
            wrap='word',
            bg='#141b24',
            fg='#e8edf2',
            insertbackground='#e8edf2',
            font=('DejaVu Sans Mono', 10),
            relief='flat',
        )
        self.server_log_text.pack(fill='both', expand=True)
        self.server_log_text.configure(state='disabled')

    def _card(self, parent):
        return ttk.Frame(parent, style='Card.TFrame', padding=14)

    def _kv(self, parent, key, variable):
        row = ttk.Frame(parent, style='Card.TFrame')
        row.pack(fill='x', pady=3)
        ttk.Label(row, text=f'{key}:', style='Key.TLabel').pack(side='left')
        ttk.Label(row, textvariable=variable, style='Value.TLabel').pack(side='left', padx=(8, 0))

    def _detect_auth_mode(self):
        if os.geteuid() == 0:
            return 'Running as root'
        if shutil.which('pkexec'):
            return 'pkexec available'
        if shutil.which('sudo'):
            return 'sudo (non-interactive only)'
        return 'No elevation helper found'

    def _append_output(self, text: str):
        self.output_text.configure(state='normal')
        self.output_text.insert('end', text + '\n')
        self.output_text.see('end')
        self.output_text.configure(state='disabled')

    def _set_server_log(self, content: str):
        self.server_log_text.configure(state='normal')
        self.server_log_text.delete('1.0', 'end')
        self.server_log_text.insert('end', content)
        self.server_log_text.see('end')
        self.server_log_text.configure(state='disabled')

    def _tail_text(self, path: Path) -> str:
        if not path.exists():
            return 'No server log yet. Start the controller to populate this panel.\n'
        data = path.read_bytes()
        if len(data) > TAIL_BYTES:
            data = data[-TAIL_BYTES:]
        return data.decode('utf-8', errors='replace')

    def _read_settings(self):
        if not SETTINGS_PATH.exists():
            self.status_vars['settings'].set(f'Missing: {SETTINGS_PATH}')
            self.status_vars['tmot'].set('Not configured')
            self.status_vars['cmot'].set('Not configured')
            return

        self.status_vars['settings'].set(str(SETTINGS_PATH))
        try:
            payload = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
        except Exception as exc:
            self.status_vars['tmot'].set(f'Invalid settings: {exc}')
            self.status_vars['cmot'].set('Invalid settings file')
            return

        self.status_vars['tmot'].set(self._format_path_status(payload.get('tmot_path')))
        self.status_vars['cmot'].set(self._format_path_status(payload.get('cmot_path')))

    def _format_path_status(self, raw_value):
        value = str(raw_value or '').strip()
        if not value:
            return 'Not configured'
        path = Path(value)
        return f'OK | {value}' if path.exists() else f'Missing | {value}'

    def _initialize_launch_options(self, parsed):
        if self.launch_options_initialized:
            return
        port = str(parsed.get('PORT', DEFAULT_PORT) or DEFAULT_PORT).strip()
        reload_value = str(parsed.get('RELOAD', DEFAULT_RELOAD) or DEFAULT_RELOAD).strip()
        if port.isdigit():
            self.launch_vars['port'].set(port)
        else:
            self.launch_vars['port'].set(DEFAULT_PORT)
        self.launch_vars['reload'].set(reload_value != '0')
        self.launch_options_initialized = True

    def _read_launcher_status(self):
        self.status_vars['venv'].set('Present' if (ROOT_DIR / '.venv' / 'bin' / 'python').exists() else 'Missing')
        try:
            completed = subprocess.run(
                ['/bin/bash', str(SCRIPT_PATH), '--status'],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as exc:
            self.status_vars['server'].set(f'Status error: {exc}')
            self.status_vars['pid'].set('-')
            return

        parsed = {}
        for line in completed.stdout.splitlines():
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            parsed[key.strip()] = value.strip()

        self._initialize_launch_options(parsed)
        server_state = parsed.get('STATUS', 'unknown')
        pid = parsed.get('PID') or '-'
        self.status_vars['server'].set(server_state.title())
        self.status_vars['pid'].set(pid)
        self.status_vars['host'].set(parsed.get('HOST', DEFAULT_HOST))
        self.status_vars['port'].set(parsed.get('PORT', DEFAULT_PORT))
        self.status_vars['reload'].set('enabled' if parsed.get('RELOAD', DEFAULT_RELOAD) != '0' else 'disabled')

    def refresh_everything(self):
        self.status_vars['auth'].set(self._detect_auth_mode())
        self._read_settings()
        self._read_launcher_status()
        self._refresh_server_log()

    def _refresh_server_log(self):
        content = self._tail_text(LOG_FILE)
        if content != self.last_log_snapshot:
            self.last_log_snapshot = content
            self._set_server_log(content)

    def _get_launch_config(self):
        port_raw = self.launch_vars['port'].get().strip()
        if not port_raw:
            messagebox.showerror('Invalid Port', 'Port cannot be empty.')
            return None
        if not port_raw.isdigit():
            messagebox.showerror('Invalid Port', f'Port must be an integer, got: {port_raw}')
            return None
        port_value = int(port_raw)
        if port_value < 1 or port_value > 65535:
            messagebox.showerror('Invalid Port', 'Port must be between 1 and 65535.')
            return None
        normalized_port = str(port_value)
        self.launch_vars['port'].set(normalized_port)
        return {
            'host': DEFAULT_HOST,
            'port': normalized_port,
            'reload': '1' if self.launch_vars['reload'].get() else '0',
        }

    def _build_command(self, args, elevated=False, launch_config=None):
        if launch_config is None:
            launch_config = {
                'host': DEFAULT_HOST,
                'port': DEFAULT_PORT,
                'reload': DEFAULT_RELOAD,
            }
        host = launch_config['host']
        port = launch_config['port']
        reload_value = launch_config['reload']

        if not elevated:
            return ['/bin/bash', str(SCRIPT_PATH), *args]
        if os.geteuid() == 0:
            return ['env', f'MIGA_HOST={host}', f'MIGA_PORT={port}', f'MIGA_RELOAD={reload_value}', '/bin/bash', str(SCRIPT_PATH), *args]
        if shutil.which('pkexec'):
            return [
                'pkexec',
                'env',
                f'MIGA_HOST={host}',
                f'MIGA_PORT={port}',
                f'MIGA_RELOAD={reload_value}',
                '/bin/bash',
                str(SCRIPT_PATH),
                *args,
            ]
        if shutil.which('sudo'):
            return [
                'sudo',
                '-n',
                'env',
                f'MIGA_HOST={host}',
                f'MIGA_PORT={port}',
                f'MIGA_RELOAD={reload_value}',
                '/bin/bash',
                str(SCRIPT_PATH),
                *args,
            ]
        return None

    def start_controller(self):
        launch_config = self._get_launch_config()
        if launch_config is None:
            return
        self.run_launcher(['--start-detached'], elevated=True, launch_config=launch_config)

    def run_launcher(self, args, elevated=False, launch_config=None):
        if self.task_running:
            messagebox.showinfo('Launcher Busy', 'Another launcher action is still running.')
            return

        command = self._build_command(args, elevated=elevated, launch_config=launch_config)
        if command is None:
            messagebox.showerror(
                'Root Permission Required',
                'No non-interactive privilege helper is available. Install pkexec or run this GUI itself with sudo.',
            )
            return

        label = ' '.join(args) if args else 'start'
        if launch_config is not None and '--start-detached' in args:
            reload_label = 'on' if launch_config['reload'] != '0' else 'off'
            label = f'{label} [port={launch_config["port"]}, reload={reload_label}]'
        self.task_running = True
        self._append_output(f'$ {" ".join(command)}')
        threading.Thread(target=self._worker, args=(command, label, elevated), daemon=True).start()

    def _worker(self, command, label, elevated):
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.command_queue.put(('line', f'Failed to start command: {exc}'))
            self.command_queue.put(('done', 1, label, elevated))
            return

        assert process.stdout is not None
        for line in process.stdout:
            self.command_queue.put(('line', line.rstrip('\n')))
        self.command_queue.put(('done', process.wait(), label, elevated))

    def _drain_queue(self):
        while True:
            try:
                item = self.command_queue.get_nowait()
            except queue.Empty:
                break

            kind = item[0]
            if kind == 'line':
                self._append_output(item[1])
                if 'Environment check passed' in item[1]:
                    self.status_vars['env_check'].set(f'Passed at {time.strftime("%H:%M:%S")}')
                elif 'Missing packages:' in item[1] or 'Version mismatches:' in item[1] or 'Ax runtime check failed:' in item[1]:
                    self.status_vars['env_check'].set('Repairing...')
            elif kind == 'done':
                code, label, elevated = item[1], item[2], item[3]
                self.task_running = False
                if code == 0:
                    if label.startswith('--check-only'):
                        self.status_vars['env_check'].set(f'Passed at {time.strftime("%H:%M:%S")}')
                    self._append_output(f'[launcher-ui] Completed: {label}')
                else:
                    if label.startswith('--check-only'):
                        self.status_vars['env_check'].set('Failed')
                    self._append_output(f'[launcher-ui] Command failed ({code}): {label}')
                    if elevated:
                        messagebox.showerror(
                            'Privilege Error',
                            'Root launcher command failed. If sudo credentials are not cached, use pkexec or run the GUI with sudo.',
                        )
                self.refresh_everything()

        self.after(200, self._drain_queue)

    def _browser_port(self):
        if self.status_vars['server'].get().strip().lower() == 'running':
            runtime_port = self.status_vars['port'].get().strip()
            if runtime_port.isdigit():
                return runtime_port
        launch_port = self.launch_vars['port'].get().strip()
        return launch_port if launch_port.isdigit() else DEFAULT_PORT

    def _open_browser_page(self, page_name):
        webbrowser.open(f'http://127.0.0.1:{self._browser_port()}/{page_name}')

    def _poll_runtime(self):
        self.refresh_everything()
        self.after(1500, self._poll_runtime)

    def _on_close(self):
        if self.task_running:
            if not messagebox.askyesno('Launcher Busy', 'A launcher command is still running. Close the GUI anyway?'):
                return
        self.destroy()


def main():
    app = LauncherUI()
    app.mainloop()


if __name__ == '__main__':
    main()
