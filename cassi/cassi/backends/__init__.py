"""Backend layer: pluggable inference backends for CASSI.

A *backend* is anything that can:
- score a candidate token sequence under a "target" distribution
- sample a "draft" token given a prefix
- maintain a small KV-cache abstraction (optional)

The default :class:`cassi.backends.toy.ToyBackend` implements all of the above
in pure Python so the algorithmic core of CASSI can be exercised without any
ML runtime installed. Heavier backends (PyTorch, llama.cpp, MLC-LLM) can be
added by subclassing :class:`BaseBackend`.
"""

from cassi.backends.base import BaseBackend
from cassi.backends.toy import ToyBackend

__all__ = ["BaseBackend", "ToyBackend"]
