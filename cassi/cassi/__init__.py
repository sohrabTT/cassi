"""
CASSI - Confidence-Adaptive Speculative Inference
===================================================

A backend-agnostic framework for speculative decoding of large language
models. CASSI implements three contributions on top of vanilla speculative
decoding:

1. Adaptive draft length (k) controlled by online acceptance-rate feedback.
2. Cross-request pipelining that overlaps draft generation of one request
   with target verification of another.
3. Smart rollback that preserves KV-cache state for tokens beyond the first
   rejection point, reducing wasted compute.

The framework ships with a fully-functional *toy* backend that requires no
machine-learning runtime, so the algorithmic core can be developed, tested,
and benchmarked on any machine. Production backends (PyTorch, llama.cpp,
MLC-LLM) can be plugged in by subclassing :class:`cassi.backends.BaseBackend`.

Example
-------
>>> from cassi import CassiEngine, ToyBackend
>>> engine = CassiEngine(backend=ToyBackend())
>>> result = engine.generate(prompt="Hello", max_new_tokens=32)
>>> print(result.text)
"""

from cassi.engine import CassiEngine
from cassi.backends.base import BaseBackend
from cassi.backends.toy import ToyBackend

__version__ = "0.1.0"
__author__ = "Sohrab"
__all__ = [
    "CassiEngine",
    "BaseBackend",
    "ToyBackend",
    "__version__",
]
