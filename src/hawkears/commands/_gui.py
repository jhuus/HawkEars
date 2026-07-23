"""Launch the HawkEars desktop application from the main CLI."""

import sys

import click


@click.command(name="gui", short_help="Launch the HawkEars desktop GUI.")
@click.pass_context
def _gui_cmd(context: click.Context) -> None:
    """Launch the HawkEars desktop GUI."""
    from hawkears.gui.app import main

    context.exit(main([sys.argv[0]]))
