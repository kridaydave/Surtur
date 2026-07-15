import os
import sys
import time
import re
import glob
import json
from datetime import datetime

import db

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.align import Align
    from rich.box import DOUBLE
except ImportError:
    print("[Surtur] Error: 'rich' library is required to run the CLI dashboard. Please install it with 'pip install rich'.")
    sys.exit(1)

console = Console()

def get_runs_data():
    conn = db.get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 8")
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()

def get_evals_data():
    conn = db.get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.*, r.arm, r.model_id 
            FROM evals e 
            LEFT JOIN runs r ON e.run_id = r.run_id 
            ORDER BY e.created_at DESC LIMIT 5
        """)
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()

def get_latest_verdict():
    # Attempt to load verdict from experiments/p0/M0_verdict.json
    paths = ["experiments/p0/M0_verdict.json", "surtur_out/M0_verdict.json", "M0_verdict.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def parse_last_log_line(run_id):
    # Locate log file
    log_dirs = ["surtur_out/logs", "experiments/p0/logs", "logs"]
    log_path = None
    for d in log_dirs:
        p = os.path.join(d, f"{run_id}.log")
        if os.path.exists(p):
            log_path = p
            break
            
    if not log_path:
        # Fallback to search log files matching run_id
        for d in log_dirs:
            matches = glob.glob(os.path.join(d, f"*{run_id}*.log"))
            if matches:
                log_path = matches[0]
                break
                
    if not log_path or not os.path.exists(log_path):
        return None

    try:
        # Read last 30 lines to find progress
        with open(log_path, "r", errors="ignore") as f:
            lines = f.readlines()[-30:]
        
        # Look for HF/trl progress patterns or custom output prints
        # Pattern: {'loss': 1.23, 'learning_rate': 4.5e-5, 'epoch': 0.1, 'step': 10}
        step, max_steps = 0, 100
        loss, lr = None, None
        
        for line in reversed(lines):
            # Parse HF dict
            if "'loss':" in line or '"loss":' in line:
                # Use regex to extract numbers
                step_match = re.search(r"'step':\s*(\d+)", line) or re.search(r'"step":\s*(\d+)', line)
                loss_match = re.search(r"'loss':\s*([0-9.]+)", line) or re.search(r'"loss":\s*([0-9.]+)', line)
                lr_match = re.search(r"'learning_rate':\s*([0-9.e-]+)", line) or re.search(r'"learning_rate":\s*([0-9.e-]+)', line)
                
                if step_match:
                    step = int(step_match.group(1))
                if loss_match:
                    loss = float(loss_match.group(1))
                if lr_match:
                    lr = float(lr_match.group(1))
                break
                
        # Also try to extract max_steps from the log file
        for line in lines:
            if "max_steps" in line:
                max_match = re.search(r"max_steps['\"]?:\s*(\d+)", line)
                if max_match:
                    max_steps = int(max_match.group(1))
                    break
        
        return {
            "step": step,
            "max_steps": max_steps,
            "loss": loss,
            "lr": lr,
            "last_line": lines[-1].strip() if lines else ""
        }
    except Exception:
        return None

def render_inferno(runs, evals, verdict):
    active_run = None
    if runs:
        running_runs = []
        for r in runs:
            status = ''
            if hasattr(r, 'keys'):
                try:
                    status = r['status']
                except Exception:
                    pass
            elif isinstance(r, dict):
                status = r.get('status', '')
            if status == 'running':
                running_runs.append(r)
        if running_runs:
            active_run = running_runs[-1]
        else:
            active_run = runs[0]
            
    steps = '-'
    loss = '-'
    lr = '-'
    log_tail = 'No log file found.'
    run_id = 'N/A'
    
    if active_run:
        def get_val(item, key, default='-'):
            if hasattr(item, 'keys'):
                try:
                    return item[key]
                except Exception:
                    pass
                try:
                    return item.get(key, default)
                except Exception:
                    pass
            elif isinstance(item, dict):
                return item.get(key, default)
            try:
                return item[key]
            except Exception:
                return default
                
        run_id = get_val(active_run, 'run_id')
        steps = get_val(active_run, 'step', get_val(active_run, 'active_steps', '-'))
        loss = get_val(active_run, 'loss', '-')
        if isinstance(loss, float):
            loss = f'{loss:.4f}'
        lr = get_val(active_run, 'lr', get_val(active_run, 'learning_rate', '-'))
        if isinstance(lr, float):
            lr = f'{lr:.2e}'
            
        log_path = f'surtur_out/logs/{run_id}.log'
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    tail_lines = [line.strip() for line in lines[-5:] if line.strip()]
                    if tail_lines:
                        log_tail = '\n'.join(tail_lines)
                    else:
                        log_tail = 'Log file is empty.'
            except Exception as e:
                log_tail = f'Error reading log file: {e}'
        else:
            log_tail = f'[red]Not found:[/] [yellow]logs/{run_id}.log[/]'

    telemetry_text = (
        f'[bold #ff6f00]ACTIVE RUN ID:[/] {run_id}\n'
        f'[bold #ff6f00]ACTIVE STEPS:[/] {steps}\n'
        f'[bold #ff6f00]LOSS:[/] {loss}\n'
        f'[bold #ff6f00]LEARNING RATE:[/] {lr}\n\n'
        f'[bold #ffaa00]LOG TAIL (LAST 5 LINES):[/]\n'
        f'{log_tail}'
    )
    
    table = Table(
        box=DOUBLE,
        border_style='bold #ff6f00',
        title='[bold #ffaa00]INFERNO TELEMETRY - ACTIVE RUNS[/]',
        title_style='bold #ffaa00',
        expand=True
    )
    table.add_column('Run ID', style='bold #ffaa00')
    table.add_column('Arm', style='bold #ff6f00')
    table.add_column('Seed', justify='right')
    table.add_column('Epoch', justify='right')
    table.add_column('Step', justify='right')
    table.add_column('Loss', justify='right')
    table.add_column('Status')
    
    for r in runs:
        def get_val(item, key, default='-'):
            if hasattr(item, 'keys'):
                try:
                    return item[key]
                except Exception:
                    pass
                try:
                    return item.get(key, default)
                except Exception:
                    pass
            elif isinstance(item, dict):
                return item.get(key, default)
            try:
                return item[key]
            except Exception:
                return default
                
        r_id = get_val(r, 'run_id')
        arm = get_val(r, 'arm')
        seed = str(get_val(r, 'seed'))
        epoch = str(get_val(r, 'epoch', '-'))
        step = str(get_val(r, 'step', '-'))
        r_loss = get_val(r, 'loss', '-')
        if isinstance(r_loss, float):
            r_loss = f'{r_loss:.4f}'
        else:
            r_loss = str(r_loss)
        status = get_val(r, 'status')
            
        table.add_row(
            r_id, arm, seed, epoch, step, r_loss, status
        )
        
    layout = Layout()
    layout.split_column(
        Layout(name='header', size=3),
        Layout(name='body', ratio=1),
        Layout(name='footer', size=3)
    )
    
    verdict_str = 'N/A'
    verdict_style = 'bold white'
    if verdict:
        status_val = ''
        if isinstance(verdict, dict):
            if 'verdict' in verdict:
                status_val = verdict.get('verdict', '')
            elif 'pass' in verdict:
                status_val = 'PASS' if verdict.get('pass') else 'FAIL'
        if status_val == 'PASS' or status_val is True:
            verdict_str = 'PASS'
            verdict_style = 'bold green'
        elif status_val == 'FAIL' or status_val is False:
            verdict_str = 'FAIL'
            verdict_style = 'bold red'
            
    header_panel = Panel(
        Align.center(f'[bold #ffaa00]🔥 SURTUR INFERNO DASHBOARD 🔥[/]  |  [bold #ff6f00]Verdict:[/] [{verdict_style}]{verdict_str}[/]  |  {get_animation_flair()}'),
        box=DOUBLE,
        border_style='bold #ffaa00'
    )
    layout['header'].update(header_panel)
    
    layout['body'].split_row(
        Layout(name='telemetry', ratio=1),
        Layout(name='runs', ratio=3)
    )
    
    telemetry_panel = Panel(
        telemetry_text,
        box=DOUBLE,
        border_style='bold #ffaa00',
        title=f'{get_animation_flair()} [bold #ffaa00]TELEMETRY MONITOR[/]'
    )
    layout['body']['telemetry'].update(telemetry_panel)
    layout['body']['runs'].update(table)

    footer_panel = Panel(
        Align.center(f'[bold #ff6f00]{get_marquee()}[/]'),
        box=DOUBLE,
        border_style='bold #ffaa00'
    )
    layout['footer'].update(footer_panel)
    
    return layout

def get_animation_flair():
    frames = ["🔥", "💥", "⚡", "✨", "☄️"]
    idx = int(time.time() * 4) % len(frames)
    colors = ["#ff3300", "#ff6600", "#ffaa00", "#ffcc00", "#ffff00", "#ffcc00", "#ffaa00", "#ff6600"]
    c_idx = int(time.time() * 6) % len(colors)
    return f"[bold {colors[c_idx]}]{frames[idx]}[/]"

def get_marquee():
    text = "   🔥 SURTUR INFERNO ACTIVE FINETUNING HUB  |  DATABASE ONLINE  |  UNSLOTH READY  |  READY TO IGNITE  |  "
    shift = int(time.time() * 3) % len(text)
    return text[shift:] + text[:shift]

def animate_welcome_banner():
    console.clear()
    boot_lines = [
        "[bold red][Surtur TUI][/] Initializing Surtur Inferno Engine...",
        "[bold red][Surtur TUI][/] Connecting to sqlite database... [green]OK[/]",
        "[bold red][Surtur TUI][/] Checking active training checkpoints... [green]READY[/]",
        "[bold red][Surtur TUI][/] Ignition sequence... [bold #ff6f00]🔥 ONLINE[/]\n"
    ]
    for line in boot_lines:
        console.print(line)
        time.sleep(0.08)
        
    banner_lines = welcome_banner.strip().split("\n")
    for line in banner_lines:
        console.print(line)
        time.sleep(0.03)

welcome_banner = """
[bold #ffaa00]🔥 WELCOME TO SURTUR INFERNO INTERACTIVE TUI 🔥[/]
[#ff6f00]-----------------------------------------------[/]
Type [bold #ffaa00]/help[/] or [bold #ffaa00]help[/] for a list of available commands.
Type [bold #ffaa00]/status[/] or [bold #ffaa00]status[/] to view the real-time telemetry.
Type [bold #ffaa00]/exit[/] or [bold #ffaa00]exit[/] to quit.
[#ff6f00]-----------------------------------------------[/]
"""

import threading
import contextlib
import io

class CommandArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class ExitIntercepted(BaseException):
    def __init__(self, code):
        self.code = code

@contextlib.contextmanager
def intercept_exit():
    original_exit = sys.exit
    def mock_exit(code=0):
        raise ExitIntercepted(code)
    sys.exit = mock_exit
    try:
        yield
    except SystemExit as e:
        raise ExitIntercepted(e.code)
    finally:
        sys.exit = original_exit

_local = threading.local()

class ThreadLocalStreamWrapper:
    def __init__(self, original_stream):
        self._original = original_stream

    def write(self, text):
        target = getattr(_local, "redirect_target", None)
        if target is not None:
            target.write(text)
        else:
            self._original.write(text)

    def flush(self):
        target = getattr(_local, "redirect_target", None)
        if target is not None:
            target.flush()
        else:
            self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)

if not isinstance(sys.stdout, ThreadLocalStreamWrapper):
    sys.stdout = ThreadLocalStreamWrapper(sys.stdout)
if not isinstance(sys.stderr, ThreadLocalStreamWrapper):
    sys.stderr = ThreadLocalStreamWrapper(sys.stderr)

class OutputRedirector:
    def __init__(self, buffer_list):
        self.buffer_list = buffer_list

    def write(self, text):
        if text:
            for line in text.splitlines(keepends=True):
                self.buffer_list.append(line)
            
    def flush(self):
        pass

@contextlib.contextmanager
def redirect_output(buffer_list):
    redirector = OutputRedirector(buffer_list)
    old_target = getattr(_local, "redirect_target", None)
    _local.redirect_target = redirector
    try:
        yield
    finally:
        _local.redirect_target = old_target

def get_db_stats():
    conn = db.get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM runs")
        total_runs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM runs WHERE status='completed'")
        completed_runs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM runs WHERE status='running'")
        running_runs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM runs WHERE status='failed'")
        failed_runs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM evals")
        total_evals = cursor.fetchone()[0]
        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "running_runs": running_runs,
            "failed_runs": failed_runs,
            "total_evals": total_evals
        }
    except Exception:
        return {
            "total_runs": 0,
            "completed_runs": 0,
            "running_runs": 0,
            "failed_runs": 0,
            "total_evals": 0
        }
    finally:
        conn.close()

def render_help():
    table = Table(title="Surtur TUI Help", box=DOUBLE, border_style="bold #ffaa00")
    table.add_column("Command", style="bold #ff6f00")
    table.add_column("Description", style="white")
    table.add_row("/help, help", "Show this help table")
    table.add_row("/status, status", "Draw Surtur Inferno dashboard layout (updates in real-time; Ctrl+C to exit)")
    table.add_row("/run [config], run [config]", "Trigger training run (defaults to src/config.yaml)")
    table.add_row("/eval [run_id], eval [run_id]", "Evaluate completed runs (all or specific run)")
    table.add_row("/reproduce [run_id], reproduce [run_id]", "Reproduce a run")
    table.add_row("/view [run_id], view [run_id]", "Tails log file for a specific run ID")
    table.add_row("/config, config", "Display the contents of the config YAML file")
    table.add_row("/style [status|dual|split], style [status|dual|split]", "Change layout style dynamically")
    table.add_row("/clear, clear", "Clear terminal screen")
    table.add_row("/exit, exit", "Quit the TUI dashboard")
    return table

def view_log(run_id=None):
    if not run_id:
        runs = get_runs_data()
        if runs:
            run_id = runs[0].get('run_id')
    if not run_id:
        console.print("[bold red]Error: No runs available to view logs.[/]")
        return
    log_dirs = ["surtur_out/logs", "experiments/p0/logs", "logs"]
    log_path = None
    for d in log_dirs:
        p = os.path.join(d, f"{run_id}.log")
        if os.path.exists(p):
            log_path = p
            break
    if not log_path:
        for d in log_dirs:
            matches = glob.glob(os.path.join(d, f"*{run_id}*.log"))
            if matches:
                log_path = matches[0]
                break
    if not log_path or not os.path.exists(log_path):
        console.print(f"[bold red]Error: Log file not found for run ID {run_id}.[/]")
        return
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            tail = [line.strip() for line in lines[-25:] if line.strip()]
            console.print(Panel(
                "\n".join(tail),
                title=f"[bold #ffaa00]Log file tail: {log_path}[/]",
                border_style="bold #ff6f00"
            ))
    except Exception as e:
        console.print(f"[bold red]Error reading log: {e}[/]")

def render_split(runs, evals, verdict, log_buffer=None):
    layout = Layout()
    layout.split_column(
        Layout(name='header', size=3),
        Layout(name='body', ratio=1),
        Layout(name='footer', size=3)
    )
    
    verdict_str = 'N/A'
    verdict_style = 'bold white'
    if verdict:
        status_val = ''
        if isinstance(verdict, dict):
            if 'verdict' in verdict:
                status_val = verdict.get('verdict', '')
            elif 'pass' in verdict:
                status_val = 'PASS' if verdict.get('pass') else 'FAIL'
        if status_val == 'PASS' or status_val is True:
            verdict_str = 'PASS'
            verdict_style = 'bold green'
        elif status_val == 'FAIL' or status_val is False:
            verdict_str = 'FAIL'
            verdict_style = 'bold red'

    header_panel = Panel(
        Align.center(f'[bold #ffaa00]🔥 SURTUR INFERNO - SPLIT TUI 🔥[/]  |  [bold #ff6f00]Verdict:[/] [{verdict_style}]{verdict_str}[/]  |  {get_animation_flair()}'),
        box=DOUBLE,
        border_style='bold #ffaa00'
    )
    layout['header'].update(header_panel)
    
    layout['body'].split_row(
        Layout(name='left', ratio=1),
        Layout(name='right', ratio=1)
    )
    
    # Left: Logs and Input Logs
    active_run = None
    if runs:
        running_runs = [r for r in runs if r.get('status') == 'running']
        if running_runs:
            active_run = running_runs[-1]
        else:
            active_run = runs[0]
            
    log_content = ""
    if log_buffer:
        log_content = "".join(log_buffer[-20:])
    elif active_run:
        run_id = active_run.get('run_id')
        log_path = f'surtur_out/logs/{run_id}.log'
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    tail_lines = [line.strip() for line in lines[-20:] if line.strip()]
                    if tail_lines:
                        log_content = '\n'.join(tail_lines)
                    else:
                        log_content = 'Log file is empty.'
            except Exception as e:
                log_content = f'Error reading log file: {e}'
        else:
            log_content = f'Log file not found: {log_path}'
    else:
        log_content = "No runs or logs available."

    left_panel = Panel(
        log_content,
        box=DOUBLE,
        border_style='bold #ffaa00',
        title=f'{get_animation_flair()} [bold #ffaa00]LOGS & CONSOLE[/]'
    )
    layout['body']['left'].update(left_panel)
    
    # Right: Runs table and Database stats
    layout['body']['right'].split_column(
        Layout(name='runs_table', ratio=2),
        Layout(name='db_stats', ratio=1)
    )
    
    table = Table(
        box=DOUBLE,
        border_style='bold #ff6f00',
        expand=True
    )
    table.add_column('Run ID', style='bold #ffaa00')
    table.add_column('Arm', style='bold #ff6f00')
    table.add_column('Step', justify='right')
    table.add_column('Loss', justify='right')
    table.add_column('Status')
    
    for r in runs[:5]:
        r_id = r.get('run_id', '-')
        arm = r.get('arm', '-')
        step = str(r.get('step', '-'))
        r_loss = r.get('loss', '-')
        if isinstance(r_loss, float):
            r_loss = f'{r_loss:.4f}'
        else:
            r_loss = str(r_loss)
        status = r.get('status', '-')
        table.add_row(r_id, arm, step, r_loss, status)
        
    layout['body']['right']['runs_table'].update(table)
    
    stats = get_db_stats()
    stats_text = (
        f"[bold #ff6f00]Total Runs:[/] {stats['total_runs']}\n"
        f"[bold #ff6f00]Completed Runs:[/] {stats['completed_runs']}\n"
        f"[bold #ff6f00]Running Runs:[/] {stats['running_runs']}\n"
        f"[bold #ff6f00]Failed Runs:[/] {stats['failed_runs']}\n"
        f"[bold #ff6f00]Total Evals:[/] {stats['total_evals']}"
    )
    stats_panel = Panel(
        stats_text,
        box=DOUBLE,
        border_style='bold #ffaa00',
        title=f'{get_animation_flair()} [bold #ffaa00]DATABASE STATS[/]'
    )
    layout['body']['right']['db_stats'].update(stats_panel)

    footer_panel = Panel(
        Align.center(f'[bold #ff6f00]{get_marquee()}[/]'),
        box=DOUBLE,
        border_style='bold #ffaa00'
    )
    layout['footer'].update(footer_panel)
    
    return layout

def show_status_live(current_style, refresh_rate):
    console.clear()
    with Live(auto_refresh=False, console=console) as live:
        try:
            while True:
                runs = get_runs_data()
                evals = get_evals_data()
                verdict = get_latest_verdict()
                if current_style == 'split':
                    layout = render_split(runs, evals, verdict)
                else:
                    layout = render_inferno(runs, evals, verdict)
                live.update(layout, refresh=True)
                time.sleep(refresh_rate)
        except KeyboardInterrupt:
            pass

def run_command_in_background(task_func, args, current_style, refresh_rate):
    log_buffer = []
    thread_exception = None
    
    def worker():
        nonlocal thread_exception
        try:
            with redirect_output(log_buffer):
                with intercept_exit():
                    task_func(args)
        except ExitIntercepted as e:
            if e.code != 0 and e.code is not None:
                thread_exception = Exception(f"Command exited with code {e.code}")
        except Exception as e:
            thread_exception = e

    t = threading.Thread(target=worker)
    t.start()

    with Live(auto_refresh=False, console=console) as live:
        while t.is_alive():
            runs = get_runs_data()
            evals = get_evals_data()
            verdict = get_latest_verdict()
            if current_style == 'split':
                layout = render_split(runs, evals, verdict, log_buffer=log_buffer)
            else:
                layout = render_inferno(runs, evals, verdict)
            live.update(layout, refresh=True)
            time.sleep(refresh_rate)
            
    t.join()
    if thread_exception:
        console.print(f"[bold red]Command failed: {thread_exception}[/]")
    else:
        console.print("[bold green]Command finished successfully.[/]")

def run_dashboard(args):
    global surtur
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import surtur

    refresh_rate = getattr(args, 'refresh', 1.0)
    current_style = getattr(args, 'style', 'dual')
    db.init_db()
    
    console.clear()
    animate_welcome_banner()
    
    while True:
        try:
            runs = get_runs_data()
            evals = get_evals_data()
            verdict = get_latest_verdict()
            
            if current_style == 'split':
                console.clear()
                layout = render_split(runs, evals, verdict)
                console.print(layout)
            elif current_style == 'status':
                latest_run_id = '-'
                latest_status = '-'
                if runs:
                    latest_run_id = runs[0].get('run_id', '-')
                    latest_status = runs[0].get('status', '-')
                
                verdict_str = 'N/A'
                if verdict:
                    status_val = ''
                    if isinstance(verdict, dict):
                        if 'verdict' in verdict:
                            status_val = verdict.get('verdict', '')
                        elif 'pass' in verdict:
                            status_val = 'PASS' if verdict.get('pass') else 'FAIL'
                    if status_val == 'PASS' or status_val is True:
                        verdict_str = 'PASS'
                    elif status_val == 'FAIL' or status_val is False:
                        verdict_str = 'FAIL'
                
                console.print(f"[bold #ff6f00]🔥 STATUS | Runs: {len(runs)} | Evals: {len(evals)} | Verdict: {verdict_str} | Latest: {latest_run_id} ({latest_status})[/]")
            
            try:
                console.print(f"{get_animation_flair()} [bold #ffaa00]surtur>[/] ", end="")
                user_input = input().strip()
            except EOFError:
                break
            
            if not user_input:
                continue
                
            parts = user_input.split()
            cmd = parts[0]
            args_list = parts[1:]
            
            cmd_clean = cmd.lower()
            if cmd_clean.startswith('/'):
                cmd_clean = cmd_clean[1:]
                
            if cmd_clean == 'exit':
                break
            elif cmd_clean == 'help':
                console.print(render_help())
            elif cmd_clean == 'clear':
                console.clear()
            elif cmd_clean == 'status':
                show_status_live(current_style, refresh_rate)
            elif cmd_clean == 'run':
                console.print("\n[bold #ffaa00]Select run configuration mode:[/]")
                console.print("  [1] Default Configuration (src/config.yaml)")
                console.print("  [2] Custom Configuration Mode (interactive setup)")
                try:
                    mode_choice = input("Select [1/2] (default: 1): ").strip()
                except (KeyboardInterrupt, EOFError):
                    continue
                
                if mode_choice == "2":
                    console.print("\n[bold #ffaa00]Entering Custom Configuration Mode. Press Enter to keep defaults.[/]")
                    def get_input(prompt, default_val):
                        try:
                            val = input(f"{prompt} (default: {default_val}): ").strip()
                            return val if val else default_val
                        except (KeyboardInterrupt, EOFError):
                            return default_val

                    model_id = get_input("  Model ID", "facebook/opt-125m")
                    method = get_input("  Method (sft/grpo)", "sft")
                    layer_spec = get_input("  Layer spec", "last_4")
                    dataset_path = get_input("  Dataset path", "data/p0_train.jsonl")
                    seeds_str = get_input("  Seeds (comma-separated)", "42")
                    arms_str = get_input("  Arms (comma-separated)", "surtur")
                    max_steps_str = get_input("  Max steps", "10")
                    batch_size_str = get_input("  Batch size", "4")
                    lr_str = get_input("  Learning rate", "5e-5")
                    dry_run_choice = get_input("  Dry run (y/n)", "n")

                    try:
                        seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
                    except ValueError:
                        seeds = [42]
                    arms = [a.strip() for a in arms_str.split(",") if a.strip()]
                    try:
                        max_steps = int(max_steps_str)
                    except ValueError:
                        max_steps = 10
                    try:
                        batch_size = int(batch_size_str)
                    except ValueError:
                        batch_size = 4
                    try:
                        lr = float(lr_str)
                    except ValueError:
                        lr = 5e-5
                    
                    dry_run = dry_run_choice.lower() in ("y", "yes")

                    custom_cfg = {
                        "model_id": model_id,
                        "method": method,
                        "layer_spec": layer_spec,
                        "dataset_path": dataset_path,
                        "seeds": seeds,
                        "arms": arms,
                        "max_steps": max_steps,
                        "batch_size": batch_size,
                        "lr": lr,
                        "dtype": "bf16"
                    }

                    os.makedirs("surtur_out/temp_configs", exist_ok=True)
                    import yaml
                    config_path = f"surtur_out/temp_configs/custom_{int(time.time())}.yaml"
                    try:
                        with open(config_path, "w") as f:
                            yaml.dump(custom_cfg, f)
                        console.print(f"[bold green]Created custom config at {config_path}[/]")
                    except Exception as e:
                        console.print(f"[bold red]Failed to write custom config: {e}. Falling back to default config.[/]")
                        config_path = "src/config.yaml"
                        dry_run = False
                else:
                    config_path = "src/config.yaml"
                    dry_run = "--dry-run" in args_list
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, "r") as f:
                                content = f.read()
                            console.print(Panel(
                                content,
                                title=f"[bold #ffaa00]Running Default Configuration: {config_path}[/]",
                                border_style="bold #ffaa00"
                            ))
                        except Exception:
                            pass
                    else:
                        console.print(f"[bold red]Error: Default config not found at {config_path}[/]")
                        continue
                
                run_args = CommandArgs(config=config_path, dry_run=dry_run)
                if current_style in ('dual', 'split'):
                    run_command_in_background(surtur.run_cmd, run_args, current_style, refresh_rate)
                else:
                    try:
                        with intercept_exit():
                            surtur.run_cmd(run_args)
                    except ExitIntercepted as e:
                        if e.code != 0 and e.code is not None:
                            console.print(f"[bold red]Command exited with code {e.code}[/]")
                    except Exception as e:
                        console.print(f"[bold red]Command failed: {e}[/]")
            elif cmd_clean == 'eval':
                run_id = None
                max_examples = 1000
                non_flag_args = [a for a in args_list if not a.startswith('-')]
                if len(non_flag_args) > 0:
                    run_id = non_flag_args[0]
                if len(non_flag_args) > 1:
                    try:
                        max_examples = int(non_flag_args[1])
                    except ValueError:
                        pass
                eval_args = CommandArgs(run_id=run_id, max_examples=max_examples)
                if current_style in ('dual', 'split'):
                    run_command_in_background(surtur.eval_cmd, eval_args, current_style, refresh_rate)
                else:
                    try:
                        with intercept_exit():
                            surtur.eval_cmd(eval_args)
                    except ExitIntercepted as e:
                        if e.code != 0 and e.code is not None:
                            console.print(f"[bold red]Command exited with code {e.code}[/]")
                    except Exception as e:
                        console.print(f"[bold red]Command failed: {e}[/]")
            elif cmd_clean == 'reproduce':
                if not args_list:
                    console.print("[bold red]Error: run_id is required for reproduce.[/]")
                    continue
                run_id = args_list[0]
                reproduce_args = CommandArgs(run_id=run_id)
                if current_style in ('dual', 'split'):
                    run_command_in_background(surtur.reproduce_cmd, reproduce_args, current_style, refresh_rate)
                else:
                    try:
                        with intercept_exit():
                            surtur.reproduce_cmd(reproduce_args)
                    except ExitIntercepted as e:
                        if e.code != 0 and e.code is not None:
                            console.print(f"[bold red]Command exited with code {e.code}[/]")
                    except Exception as e:
                        console.print(f"[bold red]Command failed: {e}[/]")
            elif cmd_clean == 'view':
                run_id = args_list[0] if args_list else None
                view_log(run_id)
            elif cmd_clean == 'config':
                config_path = "src/config.yaml"
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            content = f.read()
                        console.print(Panel(
                            content,
                            title=f"[bold #ffaa00]Configuration: {config_path}[/]",
                            border_style="bold #ffaa00"
                        ))
                    except Exception as e:
                        console.print(f"[bold red]Error reading config: {e}[/]")
                else:
                    console.print(f"[bold red]Config file not found at {config_path}[/]")
            elif cmd_clean == 'style':
                if not args_list:
                    console.print(f"Current style: {current_style}")
                    continue
                new_style = args_list[0].lower()
                if new_style not in ('status', 'dual', 'split'):
                    console.print("[bold red]Error: Style must be one of 'status', 'dual', 'split'.[/]")
                else:
                    current_style = new_style
                    console.print(f"Style changed to: {current_style}")
            else:
                console.print(f"[bold red]Unknown command: {cmd}[/]")
        except KeyboardInterrupt:
            console.print("\n[bold #ff6f00]Type /exit or exit to quit Surtur TUI.[/]")

if __name__ == "__main__":
    class Args:
        refresh = 1.0
        style = 'dual'
    run_dashboard(Args())

