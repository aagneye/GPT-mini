"""
Compatibility shim for older cloud scripts and docs.

`train_cloud.py` imports `config.py`, which now contains the actual
environment-aware defaults and all cloud tuning knobs.
"""

from config import *  # noqa: F401,F403
