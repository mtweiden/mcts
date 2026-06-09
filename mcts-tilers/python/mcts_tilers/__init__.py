import sys

from .mcts_tilers import *
from . import mcts_tilers as _ext

# PyO3 submodules created via `add_submodule` are not auto-registered in
# `sys.modules` — without this `from mcts_tilers import rl` works but
# `import mcts_tilers.rl` raises ModuleNotFoundError.
rl = _ext.rl
sys.modules["mcts_tilers.rl"] = _ext.rl

__doc__ = _ext.__doc__
if hasattr(_ext, "__all__"):
    __all__ = _ext.__all__  # type: ignore
