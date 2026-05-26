#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def strip_user_site_packages() -> None:
    """Prevent ~/.local packages from shadowing the active conda env."""
    user_site_root = os.path.expanduser("~/.local/lib")
    sys.path[:] = [
        path
        for path in sys.path
        if not (
            path
            and os.path.abspath(path).startswith(user_site_root)
            and "site-packages" in os.path.abspath(path)
        )
    ]
