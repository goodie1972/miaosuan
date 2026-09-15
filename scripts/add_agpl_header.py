#!/usr/bin/env python3
"""Add AGPL-3.0 license header to all Python source files that lack it.

Uses Python file I/O (not bash echo) to avoid corrupting escape sequences
like \\n, \\t, \\\\ in source code.
"""
from pathlib import Path
import subprocess

HEADER = """# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""

def main():
    # Get all .py files tracked by git
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        capture_output=True, text=True, check=True
    )
    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]

    updated = 0
    skipped = 0
    for filepath in files:
        path = Path(filepath)
        content = path.read_text(encoding="utf-8", errors="replace")

        # Check if file already contains the license line
        if "Licensed under the GNU Affero" in content[:500]:
            skipped += 1
            continue

        # Prepend header (preserve original content exactly)
        path.write_text(HEADER + content, encoding="utf-8")
        print(f"Updated {filepath}")
        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped")

if __name__ == "__main__":
    main()
