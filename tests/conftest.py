"""Shared pytest configuration.

Puts `scripts/` on sys.path so benchmark tooling can be imported and tested
the same way as the package itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
