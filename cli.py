#!/usr/bin/env python3
"""
Subject Frame Extractor CLI Entrypoint.
"""

import sys

from core.cli_args import cli

if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
