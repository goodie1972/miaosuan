#!/usr/bin/env bash
# Add AGPL‑3.0 header to all Python source files that lack it.
HEADER='# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.
'

# Find all .py files tracked by git
git ls-files "*.py" | while read -r file; do
  # Check if file already contains the license line
  if ! grep -q "Licensed under the GNU Affero" "$file"; then
    # Prepend header
    echo -e "$HEADER\n$(cat "$file")" > "$file"
    echo "Updated $file"
  fi
done
