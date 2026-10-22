"""Anti-false-trigger post-processing gate + application state machine (PRD FR-04, APP_FLOW §1–2).

Decision chain for every inference window::

    Motion Gate  ->  argmax != Background  ->  max(P) >= threshold (0.80)
                 ->  same class in 3 consecutive windows  ->  cooldown expired (0.8–1.0 s)
                 ->  (release rule) class differs from the last fired class or was released

The *release rule* addresses a property of continuous sliding windows: a
32-frame window (~2.7 s) still contains the just-executed gesture after the
cooldown elapses, so without it the same command could fire twice. A class is
re-armed as soon as any window predicts something else (including Background /
low confidence).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SystemState(str, Enum):
    STANDBY = "STANDBY"        # camera off / no hand > 1 s, buffer empty      (gray)
    BUFFERING = "BUFFERING"    # hand present, buffer filling to 32/32          (blue)
    MONITORING = "MONITORING"  # buffer full, inference running, no trigger     (emerald outline)
    TRIGGERED = "TRIGGERED"    # gesture confirmed, OS command dispatched       (green solid)
    COOLDOWN = "COOLDOWN"      # ignoring new triggers for 0.8–1.0 s            (amber)
    PAUSED = "PAUSED"          # engine stopped by the user                    (red)


@dataclass
class GateConfig:
    confidence_threshold: float = 0.80
    consecutive_windows: int = 3
    cooldown_s: float = 0.9
    motion_threshold: float = 0.006
    require_release: bool = True
    triggered_display_s: float = 0.35

    @classmethod
    def from_cfg(cls, cfg: dict) -> "GateConfig":
        pp = cfg["postprocessing"]
        return cls(float(pp["confidence_threshold"]), int(pp["consecutive_windows"]),
                   float(pp["cooldown_s"]), float(pp["motion_threshold"]),
                   bool(pp["require_release"]), float(pp["triggered_display_s"]))


@dataclass
class GateDecision:
    fire: bool
    class_id: Optional[int]
    confidence: float
    consistency: int
    motion_ok: bool
    cooldown_remaining: float
    checks: dict = field(default_factory=dict)
    reason: str = ""


class PostProcessGate:
    """Stateful decision gate; call :meth:`update` once per inference window."""

    def __init__(self, config: GateConfig, background_id: int = 7):
        self.cfg = config
        self.background_id = background_id
        self.history: deque = deque(maxlen=config.consecutive_windows)
        self.last_trigger_t = -np.inf
        self.last_fired_class: Optional[int] = None
        self.last_decision: Optional[GateDecision] = None

    # ---- live-tunable parameters --------------------------------------------
    def set_threshold(self, value: float) -> None:
        self.cfg.confidence_threshold = float(value)

    def set_cooldown(self, value: float) -> None:
        self.cfg.cooldown_s = float(value)

    def set_consecutive(self, n: int) -> None:
        self.cfg.consecutive_windows = int(n)
        self.history = deque(self.history, maxlen=int(n))

    def reset(self) -> None:
        """Clear temporal-consistency history (e.g. buffer reset or model hot-swap)."""
        self.history.clear()
        self.last_fired_class = None

    # ---- core logic ------------------------------------------------------------
    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self.cfg.cooldown_s - (now - self.last_trigger_t))

    def update(self, probs: Optional[np.ndarray], motion_ok: bool, now: float) -> GateDecision:
        cd = self.cooldown_remaining(now)
        pred, conf = None, 0.0
        if probs is not None and motion_ok:
            pred, conf = int(np.argmax(probs)), float(np.max(probs))

        not_bg = pred is not None and pred != self.background_id
        conf_ok = not_bg and conf >= self.cfg.confidence_threshold
        raw_candidate = pred if conf_ok else None

        # Release rule (evaluated on the raw candidate, independent of cooldown)
        if self.last_fired_class is not None and raw_candidate != self.last_fired_class:
            self.last_fired_class = None
        released = (not self.cfg.require_release) or self.last_fired_class is None \
            or raw_candidate != self.last_fired_class

        # During cooldown new evidence is ignored -> consistency must rebuild afterwards
        self.history.append(raw_candidate if cd == 0.0 else None)
        consistency = 0
        if raw_candidate is not None:
            for c in reversed(self.history):
                if c != raw_candidate:
                    break
                consistency += 1

        checks = {
            "motion": bool(motion_ok),
            "not_background": bool(not_bg),
            "confidence": bool(conf_ok),
            "consistency": consistency >= self.cfg.consecutive_windows,
            "cooldown": cd == 0.0,
            "released": bool(released),
        }
        fire = all(checks.values())
        reason = "fire" if fire else next(k for k, v in checks.items() if not v)
        if fire:
            self.last_trigger_t = now
            self.last_fired_class = raw_candidate
            self.history.clear()
        decision = GateDecision(fire, pred, conf, consistency, bool(motion_ok),
                                self.cooldown_remaining(now), checks, reason)
        self.last_decision = decision
        return decision

    def phase(self, now: float) -> SystemState:
        """State once the buffer is full: TRIGGERED -> COOLDOWN -> MONITORING."""
        dt = now - self.last_trigger_t
        if dt < self.cfg.triggered_display_s:
            return SystemState.TRIGGERED
        if dt < self.cfg.cooldown_s:
            return SystemState.COOLDOWN
        return SystemState.MONITORING
