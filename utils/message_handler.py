"""SOFA message handler for headless runs.

Intercepts SOFA's internal messages (info, warning, error) and prints
them to stdout so they're visible when running without the GUI.

Usage as a context manager::

    with SofaMessageHandler():
        Sofa.Simulation.init(root)
        for _ in range(n_steps):
            Sofa.Simulation.animate(root, dt)

Or register/unregister manually::

    handler = SofaMessageHandler()
    handler.register()
    ...
    handler.unregister()
"""
from __future__ import annotations

import sys

import Sofa.Helper


_TYPE_PREFIX = {
    "Info":       "[INFO]   ",
    "Warning":    "[WARN]   ",
    "Error":      "[ERROR]  ",
    "Fatal":      "[FATAL]  ",
    "Deprecated": "[DEPR]   ",
}


class SofaMessageHandler(Sofa.Helper.MessageHandler):
    """Prints SOFA messages to stdout during headless simulation."""

    def __init__(self, print_info: bool = False) -> None:
        super().__init__()
        self._print_info = print_info
        self.n_errors = 0
        self.n_warnings = 0
        self.n_info = 0

    def process(self, msg) -> None:
        msg_type = msg.get("type", "Info") if isinstance(msg, dict) else str(getattr(msg, "type", "Info"))
        sender = msg.get("sender", "") if isinstance(msg, dict) else str(getattr(msg, "sender", ""))
        text = msg.get("message", "") if isinstance(msg, dict) else str(getattr(msg, "message", str(msg)))

        if msg_type in ("Error", "Fatal"):
            self.n_errors += 1
            prefix = _TYPE_PREFIX.get(msg_type, "[ERROR]  ")
            print(f"{prefix}[{sender}] {text}", file=sys.stderr, flush=True)
        elif msg_type == "Warning":
            self.n_warnings += 1
            prefix = _TYPE_PREFIX["Warning"]
            print(f"{prefix}[{sender}] {text}", file=sys.stderr, flush=True)
        elif msg_type == "Deprecated":
            self.n_warnings += 1
            prefix = _TYPE_PREFIX["Deprecated"]
            print(f"{prefix}[{sender}] {text}", file=sys.stderr, flush=True)
        else:
            self.n_info += 1
            if self._print_info:
                prefix = _TYPE_PREFIX.get(msg_type, "[INFO]   ")
                print(f"{prefix}[{sender}] {text}", flush=True)

    def summary(self) -> str:
        return (f"SOFA messages: {self.n_errors} errors, "
                f"{self.n_warnings} warnings, {self.n_info} info")
