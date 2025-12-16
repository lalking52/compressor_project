import os
from pathlib import Path


def set_numba_cache_dir():
    """
    Ensure numba can write its cache in environments where the default
    site-packages directory might be read-only.
    """
    cache_dir = Path(__file__).resolve().parent.parent / ".numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    return cache_dir


# Configure cache dir on import.
set_numba_cache_dir()
