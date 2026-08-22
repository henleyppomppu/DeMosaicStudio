"""DeMosaic Studio AI worker. prd.md §3.1.

``PROTOCOL_VERSION`` is re-exported from :mod:`demosaic_worker.protocol`, never restated here.
A second copy of a version constant drifts silently; this package has been given exactly one.
"""

from __future__ import annotations

from .protocol import PROTOCOL_VERSION

#: Worker build version, reported at handshake. Independent of the protocol version: the worker can
#: gain features without the wire format changing, and must be able to say so.
__version__ = "0.1.0"

__all__ = ["PROTOCOL_VERSION", "__version__"]
