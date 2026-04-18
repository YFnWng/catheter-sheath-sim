"""Ensure SOFA plugin Python packages are importable.

When running via ``runSofa``, plugin paths are set automatically.
When running via ``python`` directly (headless mode), we must add them
to ``sys.path`` manually.

Call ``ensure_sofa_paths()`` before importing any SOFA-dependent modules.
"""
import os
import sys


def ensure_sofa_paths() -> None:
    """Add SOFA plugin Python site-packages to sys.path if missing."""
    sofa_root = os.environ.get("SOFA_ROOT", "/opt/SOFA/v25.12.00")
    plugin_dirs = [
        os.path.join(sofa_root, "plugins", "STLIB", "lib", "python3", "site-packages"),
        os.path.join(sofa_root, "plugins", "Cosserat", "lib", "python3", "site-packages"),
        os.path.join(sofa_root, "plugins", "SoftRobots", "lib", "python3", "site-packages"),
    ]
    for d in plugin_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
