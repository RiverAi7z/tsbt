"""Time-series operators (AlphaForge semantics; ``d`` counts bars, columns independent).

All operators accept ``(T,)`` or ``(T, N)`` arrays and return the same shape.
"""

from tsbt.ops._util import min_valid  # noqa: F401
from tsbt.ops.elementwise import *  # noqa: F403
from tsbt.ops.elementwise import __all__ as _ew
from tsbt.ops.recurrence import *  # noqa: F403
from tsbt.ops.recurrence import __all__ as _rec
from tsbt.ops.rolling import *  # noqa: F403
from tsbt.ops.rolling import __all__ as _roll

__all__ = sorted({*_ew, *_rec, *_roll, "min_valid"})
