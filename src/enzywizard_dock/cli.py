from __future__ import annotations

import argparse

from .commands.dock import add_dock_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzywizard-dock",
        description="EnzyWizard-Dock: Perform molecular docking of one or multiple substrates with a cleaned protein structure and generating a detailed JSON report."
    )
    add_dock_parser(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)