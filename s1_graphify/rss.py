from __future__ import annotations

import resource
import sys
from collections.abc import Callable


def rss_bytes(reader: Callable[[], int] | None = None) -> int:
    if reader is not None:
        return int(reader())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024
