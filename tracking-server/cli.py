#!/usr/bin/env python3
"""Artifacta CLI - Data visualization and experiment tracking."""
# mypy: disable-error-code="untyped-decorator"

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import click

# Import configuration constants
sys.path.insert(0, str(Path(__file__).parent))
from config import DEFAULT_DB_PATH, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_UI_PORT


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="artifacta")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Artifacta - Universal experiment and artifact tracking — gain insights and optimize models with confidence."""
    if ctx.invoked_subcommand is None:
        # Default to ui command when no subcommand is provided
        ctx.invoke(ui)


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Host to bind the server to")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port for the tracking server")
@click.option("--ui-port", default=DEFAULT_UI_PORT, type=int, help="Port for the UI (dev mode only)")
@click.option("--db", default=DEFAULT_DB_PATH, help="Database file path")
@click.option("--debug-logs", is_flag=True, help="Enable console log capture to file")
@click.option("--dev", is_flag=True, help="Run in development mode with hot-reload (requires Node.js)")
def ui(host: str, port: int, ui_port: int, db: str, debug_logs: bool, dev: bool) -> None:
    """Start the full UI (tracking server + frontend).

    By default, serves pre-built UI from dist/ folder on the tracking server port.
    Use --dev flag to run in development mode with hot-reload (requires Node.js).
    """
    project_root = get_project_root()
    server_dir = project_root / "tracking-server"

    # Check if UI is built (unless in dev mode)
    # Check both installed location and dev location
    try:
        from artifacta_ui import UI_DIST_PATH
        dist_exists = UI_DIST_PATH.exists()
    except ImportError:
        dist_exists = (project_root / "dist").exists()

    if not dev and not dist_exists:
        click.echo("❌ UI not built. Please run 'npm install && npm run build' first.")
        click.echo("   Or use --dev flag to run in development mode (requires Node.js).")
        sys.exit(1)

    # Set environment variables
    os.environ["TRACKING_SERVER_HOST"] = host
    os.environ["TRACKING_SERVER_PORT"] = str(port)
    os.environ["DATABASE_PATH"] = db

    if dev:
        # Development mode - run vite dev server
        os.environ["VITE_API_URL"] = f"http://{host}:{port}"

        # Enable debug logging if requested
        if debug_logs:
            os.environ["VITE_DEBUG_LOGS"] = "true"
            click.echo("🐛 Debug logging enabled - logs will be saved to browser downloads")

        processes: List[subprocess.Popen[bytes]] = []

        try:
            # Start tracking server
            click.echo(f"📊 Starting tracking server on {host}:{port}...")
            server_process = subprocess.Popen(
                [sys.executable, "main.py"], cwd=server_dir, env=os.environ.copy()
            )
            processes.append(server_process)

            # Give server time to start
            time.sleep(1)

            # Start frontend dev server
            click.echo(f"🎨 Starting UI dev server on http://localhost:{ui_port}...")
            frontend_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "--port", str(ui_port)],
                cwd=project_root,
                env=os.environ.copy(),
            )
            processes.append(frontend_process)

            click.echo("\n✅ Artifacta is running in development mode!")
            click.echo(f"   - Tracking Server: http://{host}:{port}")
            click.echo(f"   - UI: http://localhost:{ui_port}")
            click.echo("\nPress Ctrl+C to stop...")

            # Wait for processes
            for process in processes:
                process.wait()

        except KeyboardInterrupt:
            click.echo("\n🛑 Stopping Artifacta...")
            for process in processes:
                process.terminate()
            for process in processes:
                process.wait()
            click.echo("✅ Artifacta stopped")
        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            for process in processes:
                process.terminate()
            sys.exit(1)
    else:
        # Production mode - serve built UI from FastAPI server
        click.echo(f"📊 Starting server with built-in UI on http://{host}:{port}...")

        try:
            # Check if running in development (tracking-server dir exists with main.py)
            main_py = server_dir / "main.py"
            if main_py.exists():
                # Development mode
                subprocess.run([sys.executable, "main.py"], cwd=server_dir, env=os.environ.copy())
            else:
                # Installed mode - run as module
                import uvicorn
                from tracking_server.config import get_host, get_port, SERVER_BIND_HOST
                uvicorn.run("tracking_server.main:app", host=SERVER_BIND_HOST, port=get_port(), log_level="info")
        except KeyboardInterrupt:
            click.echo("\n🛑 Server stopped")


@cli.command()
@click.option("--host", default=DEFAULT_HOST, help="Host to bind to")
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port to bind to")
@click.option("--db", default=DEFAULT_DB_PATH, help="Database file path")
def server(host: str, port: int, db: str) -> None:
    """Start the tracking server without UI."""
    click.echo(f"🚀 Starting Artifacta tracking server on {host}:{port}...")

    project_root = get_project_root()
    server_dir = project_root / "tracking-server"

    # Set environment variables
    os.environ["TRACKING_SERVER_HOST"] = host
    os.environ["TRACKING_SERVER_PORT"] = str(port)
    os.environ["DATABASE_PATH"] = db

    try:
        subprocess.run([sys.executable, "main.py"], cwd=server_dir, env=os.environ.copy())
    except KeyboardInterrupt:
        click.echo("\n🛑 Server stopped")


@cli.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command()
@click.option("--db", default=DEFAULT_DB_PATH, help="Database file path")
def init(db: str) -> None:
    """Initialize the database."""
    click.echo(f"🗄️  Initializing database: {db}")

    project_root = get_project_root()
    db_path = project_root / db

    if db_path.exists():
        click.echo(f"⚠️  Database already exists: {db_path}")
        if not click.confirm("Do you want to reinitialize it?"):
            return

    # Import database initialization
    sys.path.insert(0, str(project_root / "tracking-server"))
    from database import init_db

    os.environ["DATABASE_PATH"] = db
    init_db()
    click.echo(f"✅ Database initialized: {db_path}")


@db.command()
@click.option("--db", default=DEFAULT_DB_PATH, help="Database file path")
@click.confirmation_option(prompt="Are you sure you want to clean the database?")
def clean(db: str) -> None:
    """Clean/reset the database (removes all data)."""
    click.echo(f"🧹 Cleaning database: {db}")

    project_root = get_project_root()
    db_path = project_root / db

    if db_path.exists():
        db_path.unlink()
        click.echo(f"✅ Database removed: {db_path}")

    # Reinitialize
    sys.path.insert(0, str(project_root / "tracking-server"))
    from database import init_db

    os.environ["DATABASE_PATH"] = db
    init_db()
    click.echo(f"✅ Database reinitialized: {db_path}")


@db.command(name="reset")
@click.option("--db", default=DEFAULT_DB_PATH, help="Database file path")
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
def db_reset(db: str) -> None:
    """Reset the database (alias for clean)."""
    click.echo(f"🔄 Resetting database: {db}")

    # Call clean command
    ctx = click.get_current_context()
    ctx.invoke(clean, db=db)


@cli.command()
@click.option("--db", default=DEFAULT_DB_PATH, help="Database file path")
def reset(db: str) -> None:
    """Reset database and show instructions to restart the server."""
    click.echo("🔄 Resetting Artifacta...")

    project_root = get_project_root()
    db_path = project_root / db

    # Clean database
    if db_path.exists():
        db_path.unlink()
        click.echo(f"✅ Database removed: {db_path}")

    # Reinitialize
    sys.path.insert(0, str(project_root / "tracking-server"))
    from database import init_db

    os.environ["DATABASE_PATH"] = db
    init_db()
    click.echo(f"✅ Database reinitialized: {db_path}")
    click.echo("\n⚠️  Please restart the server:")
    click.echo("   1. Stop the current server (Ctrl+C)")
    click.echo("   2. Run: python cli.py ui")


@cli.command()
def stop() -> None:
    """Stop all Artifacta processes."""
    click.echo("🛑 Stopping Artifacta processes...")

    # Kill Python processes running cli.py or main.py
    try:
        subprocess.run(["pkill", "-f", "python.*cli.py"], stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["pkill", "-f", "python.*main.py"], stderr=subprocess.DEVNULL, check=False)
        # Kill vite dev server
        subprocess.run(["pkill", "-f", "vite"], stderr=subprocess.DEVNULL, check=False)
        time.sleep(1)
        click.echo("✅ All processes stopped")
    except Exception as e:
        click.echo(f"⚠️  Error stopping processes: {e}", err=True)
        click.echo("   You may need to manually stop processes")


if __name__ == "__main__":
    cli()
