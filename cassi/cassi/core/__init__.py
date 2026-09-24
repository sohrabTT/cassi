"""Core algorithmic components of CASSI.

This sub-package contains the three contributions of CASSI:

- :mod:`cassi.core.adaptive_k`    - online controller for the draft length k.
- :mod:`cassi.core.cross_request` - cross-request pipelining scheduler.
- :mod:`cassi.core.rollback`      - smart rollback with KV-cache preservation.
- :mod:`cassi.core.verifier`      - parallel verification of draft tokens.
- :mod:`cassi.core.draft_model`   - draft-model wrapper / scheduler.
- :mod:`cassi.core.target_model`  - target-model wrapper / verifier driver.
"""
