"""CLI: launch the FiloHFT Streamlit observability dashboard.

    python -m filo_hft.scripts.run_dashboard \\
        --telemetry-root ../telemetry \\
        --config ../config/default.ini \\
        --port 8501

Thin wrapper around ``streamlit run``: we shell out with ``subprocess.run``
so Streamlit owns the event loop, and we pass configuration to the app
via environment variables (``FILOHFT_TELEMETRY_ROOT`` /
``FILOHFT_CONFIG_PATH``) which :mod:`filo_hft.gui.dashboard` reads at
startup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import click

from filo_hft.gui import dashboard as _dashboard_module


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--telemetry-root",
    "telemetry_root",
    default="../telemetry",
    type=click.Path(file_okay=False),
    help="Path to the telemetry root (contains YYYYMMDD folders).",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional config/default.ini for pip math and defaults.",
)
@click.option(
    "--port",
    default=8501,
    type=int,
    show_default=True,
    help="TCP port that Streamlit binds.",
)
@click.pass_context
def main(
    ctx: click.Context,
    telemetry_root: str,
    config_path: str | None,
    port: int,
) -> None:
    """Launch ``streamlit run dashboard.py`` as a subprocess."""

    dashboard_path = Path(_dashboard_module.__file__).resolve()
    if not dashboard_path.exists():
        raise click.ClickException(f"dashboard.py not found at {dashboard_path}")

    env = os.environ.copy()
    env["FILOHFT_TELEMETRY_ROOT"] = str(Path(telemetry_root).expanduser())
    if config_path:
        env["FILOHFT_CONFIG_PATH"] = str(Path(config_path).expanduser())

    passthrough: Tuple[str, ...] = tuple(ctx.args)
    cmd = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.headless=true",
        f"--server.port={port}",
        *passthrough,
    ]
    click.echo(f"[run_dashboard] launching: {' '.join(cmd)}")
    click.echo(
        f"[run_dashboard] FILOHFT_TELEMETRY_ROOT="
        f"{env['FILOHFT_TELEMETRY_ROOT']}"
    )
    if "FILOHFT_CONFIG_PATH" in env:
        click.echo(f"[run_dashboard] FILOHFT_CONFIG_PATH={env['FILOHFT_CONFIG_PATH']}")

    try:
        result = subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError as exc:
        raise click.ClickException(
            "streamlit executable not found on PATH — install with "
            "`pip install streamlit` or activate the project venv."
        ) from exc
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
