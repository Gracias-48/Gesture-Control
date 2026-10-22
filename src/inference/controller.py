"""OS action dispatcher via PyAutoGUI with master arm/disarm + fail-safe (PRD FR-05)."""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionResult:
    class_id: int
    label: str
    shortcut: str
    executed: bool
    latency_ms: float
    error: Optional[str] = None


class OSActionController:
    """Maps class ids to keyboard / scroll commands.

    * ``armed=False`` (default) -> commands are logged but never injected.
    * PyAutoGUI ``FAILSAFE``: slamming the mouse into a screen corner raises
      ``FailSafeException``; the controller then disarms itself automatically.
    * On macOS ``ctrl`` is remapped to ``command`` for zoom shortcuts.
    """

    def __init__(self, mapping: dict, failsafe: bool = True, armed: bool = False):
        self.mapping = {int(k): v for k, v in mapping.items()}
        self.failsafe = failsafe
        self._armed = armed
        self._lock = threading.Lock()
        self._pg = None
        self.available = False
        self.import_error: Optional[str] = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import pyautogui

            pyautogui.FAILSAFE = self.failsafe
            pyautogui.PAUSE = 0.0  # no implicit sleep after each call (latency budget)
            self._pg = pyautogui
            self.available = True
        except Exception as exc:  # headless / missing display server
            self.import_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    # ---- arming ----------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed and self.available

    def set_armed(self, value: bool) -> None:
        self._armed = bool(value)

    # ---- dispatch ----------------------------------------------------------------
    @staticmethod
    def _platform_keys(keys: list[str]) -> list[str]:
        if sys.platform == "darwin":
            return ["command" if k == "ctrl" else k for k in keys]
        return keys

    def describe(self, class_id: int) -> tuple[str, str]:
        spec = self.mapping.get(int(class_id), {"label": "Unknown", "shortcut": "-"})
        return spec.get("label", ""), spec.get("shortcut", "")

    def execute(self, class_id: int) -> ActionResult:
        """Execute the action bound to ``class_id`` (no-op when disarmed)."""
        spec = self.mapping.get(int(class_id), {"type": "none"})
        label, shortcut = spec.get("label", ""), spec.get("shortcut", "")
        t0 = time.perf_counter()
        if spec.get("type", "none") == "none" or not self.armed:
            return ActionResult(class_id, label, shortcut, False, 0.0,
                                None if self.available else self.import_error)
        err = None
        with self._lock:
            try:
                kind = spec["type"]
                if kind == "key":
                    for k in self._platform_keys(spec["keys"]):
                        self._pg.press(k)
                elif kind == "hotkey":
                    self._pg.hotkey(*self._platform_keys(spec["keys"]))
                elif kind == "scroll":
                    self._pg.scroll(int(spec["clicks"]))
                else:
                    err = f"unknown action type {kind}"
            except Exception as exc:
                if type(exc).__name__ == "FailSafeException":
                    self._armed = False
                    err = "PyAutoGUI fail-safe triggered — OS control disarmed"
                else:
                    err = f"{type(exc).__name__}: {exc}"
        return ActionResult(class_id, label, shortcut, err is None,
                            (time.perf_counter() - t0) * 1e3, err)
