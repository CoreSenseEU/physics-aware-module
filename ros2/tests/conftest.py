"""Make the physics_amm package importable without a colcon build/ROS.

L1/L2 are pure-Python contract tests; only the L4-L6 integration tests need a
sourced ROS environment and a built workspace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "physics_amm"))
