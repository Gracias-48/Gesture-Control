"""Raycast Design System Token Adaptation for Streamlit HUD (DESIGN.md).

Tokens:
- Canvas: #07080a (Deep Inky Black)
- Surface Cards: #101111 / #141516
- Hairline Border: 1px solid #242728
- Primary Text: #f9f9f9, Secondary: #9c9c9d, Disabled: #6a6b6c
- Accents: Raycast Coral (#ff6363), Cyan (#56c2ff), Emerald (#30d158), Amber (#ffd60a)
- Typography: Inter for UI, Monospace for Telemetry (tabular-nums)
"""
import numpy as np

RAYCAST_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* --- Root & Global Reset --- */
:root {
    --ray-bg: #07080a;
    --ray-surface: #101111;
    --ray-surface-elevated: #161718;
    --ray-border: #242728;
    --ray-border-hover: #35383a;
    --ray-text: #f9f9f9;
    --ray-text-muted: #9c9c9d;
    --ray-text-dim: #6a6b6c;
    --ray-coral: #ff6363;
    --ray-cyan: #56c2ff;
    --ray-green: #30d158;
    --ray-amber: #ffd60a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--ray-bg) !important;
    color: var(--ray-text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.012em;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0b0c0e !important;
    border-right: 1px solid var(--ray-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--ray-text) !important;
}

/* Hide default streamlit clutter */
#MainMenu, footer {visibility: hidden;}

/* Monospace font for telemetry and code */
.mono-text, .stMetricValue, code, pre {
    font-family: 'JetBrains Mono', 'Geist Mono', Consolas, monospace !important;
    font-feature-settings: 'tnum' on, 'zero' on !important;
}

/* --- Raycast Header Bar --- */
.ray-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background-color: var(--ray-surface);
    border: 1px solid var(--ray-border);
    border-radius: 8px;
    margin-bottom: 16px;
}
.ray-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ray-text);
    display: flex;
    align-items: center;
    gap: 8px;
}
.ray-title span.dot {
    width: 8px;
    height: 8px;
    background-color: var(--ray-coral);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px rgba(255, 99, 99, 0.6);
}

/* --- Pill Badges for State Machine --- */
.ray-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-standby {
    background-color: rgba(106, 107, 108, 0.15);
    color: #9c9c9d;
    border: 1px solid #2e3032;
}
.badge-buffering {
    background-color: rgba(86, 194, 255, 0.15);
    color: var(--ray-cyan);
    border: 1px solid rgba(86, 194, 255, 0.4);
}
.badge-monitoring {
    background-color: rgba(48, 209, 88, 0.12);
    color: var(--ray-green);
    border: 1px solid rgba(48, 209, 88, 0.4);
}
.badge-triggered {
    background-color: var(--ray-coral);
    color: #ffffff;
    border: 1px solid #ff7b7b;
    box-shadow: 0 0 12px rgba(255, 99, 99, 0.5);
    animation: pulse 1s infinite;
}
.badge-cooldown {
    background-color: rgba(255, 214, 10, 0.15);
    color: var(--ray-amber);
    border: 1px solid rgba(255, 214, 10, 0.4);
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.03); }
    100% { transform: scale(1); }
}

/* --- HUD Telemetry Card Grid --- */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
.telemetry-card {
    background-color: var(--ray-surface);
    border: 1px solid var(--ray-border);
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
}
.telemetry-label {
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--ray-text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.telemetry-value {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--ray-text);
    font-family: 'JetBrains Mono', monospace;
}
.telemetry-sub {
    font-size: 0.70rem;
    color: var(--ray-text-dim);
    margin-top: 2px;
}

/* --- Confidence Bars --- */
.conf-container {
    background-color: var(--ray-surface);
    border: 1px solid var(--ray-border);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 16px;
}
.conf-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.8rem;
}
.conf-label {
    width: 110px;
    color: var(--ray-text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.conf-bar-bg {
    flex-grow: 1;
    height: 8px;
    background-color: #1a1b1d;
    border-radius: 4px;
    margin: 0 10px;
    overflow: hidden;
    position: relative;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background-color: #35383a;
    transition: width 0.15s ease-out;
}
.conf-bar-fill.active {
    background-color: var(--ray-cyan);
    box-shadow: 0 0 6px rgba(86, 194, 255, 0.4);
}
.conf-bar-fill.triggered {
    background-color: var(--ray-coral);
    box-shadow: 0 0 8px rgba(255, 99, 99, 0.6);
}
.conf-val {
    width: 44px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--ray-text);
}

/* --- Toast Pill (Trigger Overlay) --- */
.toast-pill {
    background-color: rgba(22, 23, 24, 0.95);
    border: 1px solid var(--ray-coral);
    box-shadow: 0 4px 20px rgba(255, 99, 99, 0.35);
    border-radius: 20px;
    padding: 8px 18px;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: #ffffff;
    font-weight: 600;
    font-size: 0.9rem;
    backdrop-filter: blur(8px);
}
.toast-pill span.shortcut {
    background-color: #2b1d20;
    color: var(--ray-coral);
    border: 1px solid rgba(255, 99, 99, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* --- Action Log Table --- */
.action-log-card {
    background-color: var(--ray-surface);
    border: 1px solid var(--ray-border);
    border-radius: 8px;
    padding: 12px;
}
.action-log-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1a1b1d;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}
.action-log-item:last-child {
    border-bottom: none;
}
.action-log-ts {
    color: var(--ray-text-dim);
}
.action-log-gesture {
    color: var(--ray-text);
    font-weight: 500;
}
.action-log-key {
    color: var(--ray-cyan);
}

/* --- Section Title --- */
.hud-section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ray-text-muted);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
/* --- Live Diagnostic Bar --- */
.diagnostic-card {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    background-color: var(--ray-surface);
    border: 1px solid var(--ray-border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 10px;
    margin-bottom: 12px;
}
.diag-col {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.diag-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ray-text-muted);
}
.diag-value {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}
.diag-badge-ok {
    background-color: rgba(48, 209, 88, 0.15);
    color: #30d158;
    border: 1px solid rgba(48, 209, 88, 0.4);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.diag-badge-blocked {
    background-color: rgba(255, 99, 99, 0.15);
    color: #ff6363;
    border: 1px solid rgba(255, 99, 99, 0.4);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.diag-badge-triggered {
    background-color: #ff6363;
    color: #ffffff;
    border: 1px solid #ff8888;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    box-shadow: 0 0 10px rgba(255, 99, 99, 0.6);
}
.diag-badge-monitoring {
    background-color: rgba(86, 194, 255, 0.15);
    color: #56c2ff;
    border: 1px solid rgba(86, 194, 255, 0.4);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.diag-badge-standby {
    background-color: rgba(106, 107, 108, 0.15);
    color: #9c9c9d;
    border: 1px solid #2e3032;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
"""


def render_state_badge(state: str) -> str:
    """Render HTML badge warna-warni sesuai Raycast theme."""
    st = str(state).upper()
    css_class = {
        "STANDBY": "badge-standby",
        "BUFFERING": "badge-buffering",
        "MONITORING": "badge-monitoring",
        "TRIGGERED": "badge-triggered",
        "COOLDOWN": "badge-cooldown",
    }.get(st, "badge-standby")
    icon = {
        "STANDBY": "○",
        "BUFFERING": "◐",
        "MONITORING": "●",
        "TRIGGERED": "⚡",
        "COOLDOWN": "⏳",
    }.get(st, "●")
    return f'<span class="ray-badge {css_class}">{icon} {st}</span>'


def render_confidence_bars(class_names: list, probs: list, threshold: float = 0.80) -> str:
    """Render horizontal bar chart 8 kelas gestur dengan threshold marker."""
    html = ['<div class="conf-container">', '<div class="hud-section-title">8-Class Softmax Confidence</div>']
    p_list = [float(x) for x in probs] if probs is not None and len(probs) > 0 else []
    active_idx = p_list.index(max(p_list)) if p_list else -1
    for i, name in enumerate(class_names):
        val = p_list[i] if i < len(p_list) else 0.0
        pct = max(0.0, min(100.0, val * 100.0))
        is_active = (i == active_idx and val >= 0.20)
        is_triggered = (i == active_idx and val >= threshold)

        bar_class = "conf-bar-fill"
        if is_triggered:
            bar_class += " triggered"
        elif is_active:
            bar_class += " active"

        html.append(f"""
        <div class="conf-row">
            <div class="conf-label" title="{name}">{name}</div>
            <div class="conf-bar-bg">
                <div class="{bar_class}" style="width: {pct:.1f}%;"></div>
            </div>
            <div class="conf-val">{pct:4.1f}%</div>
        </div>
        """)
    html.append('</div>')
    return "".join(html)


def render_telemetry_cards(fps: float, latency_ms: float, buffer_fill: int, buffer_max: int,
                           cooldown_s: float, os_armed: bool) -> str:
    """Render 4 kartu metrik monospasi bergaya Raycast HUD."""
    armed_str = '<span style="color: #30d158;">ARMED</span>' if os_armed else '<span style="color: #9c9c9d;">DISARMED</span>'
    cd_str = f"{cooldown_s:.2f}s" if cooldown_s > 0 else "READY"
    cd_color = "#ffd60a" if cooldown_s > 0 else "#30d158"

    return f"""
    <div class="telemetry-grid">
        <div class="telemetry-card">
            <div class="telemetry-label">Throughput</div>
            <div class="telemetry-value">{fps:4.1f} <span style="font-size: 0.75rem; color: #9c9c9d;">FPS</span></div>
            <div class="telemetry-sub">Target 12.0 FPS</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">Inference Latency</div>
            <div class="telemetry-value">{latency_ms:4.1f} <span style="font-size: 0.75rem; color: #9c9c9d;">ms</span></div>
            <div class="telemetry-sub">Target &lt; 20 ms</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">Sliding Buffer</div>
            <div class="telemetry-value">{buffer_fill}/{buffer_max}</div>
            <div class="telemetry-sub">Window ~2.7s</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">Cooldown / Gate</div>
            <div class="telemetry-value" style="color: {cd_color};">{cd_str}</div>
            <div class="telemetry-sub">OS Control: {armed_str}</div>
        </div>
    </div>
    """


def render_diagnostic_bar(
    candidate_name: str,
    candidate_conf: float,
    motion_passed: bool,
    motion_speed: float,
    motion_threshold: float,
    gate_reason: str,
    consistency_count: int,
    consistency_target: int,
) -> str:
    """Render status diagnostik real-time tepat di bawah viewport webcam."""
    # Motion Gate Pill
    if motion_passed:
        motion_badge = f'<span class="diag-badge-ok">PASSED ({motion_speed:.4f})</span>'
    else:
        motion_badge = f'<span class="diag-badge-blocked">BLOCKED ({motion_speed:.4f} &lt; {motion_threshold:.4f})</span>'

    # Gate Reason Pill
    reason_clean = str(gate_reason)
    if "TRIGGERED" in reason_clean:
        reason_badge = f'<span class="diag-badge-triggered">⚡ {reason_clean}</span>'
    elif "BLOCKED" in reason_clean:
        reason_badge = f'<span class="diag-badge-blocked">{reason_clean}</span>'
    elif "MONITORING" in reason_clean:
        reason_badge = f'<span class="diag-badge-monitoring">{reason_clean}</span>'
    else:
        reason_badge = f'<span class="diag-badge-standby">{reason_clean}</span>'

    # Confidence bar color
    conf_pct = candidate_conf * 100.0
    conf_color = "#30d158" if conf_pct >= 50.0 else "#ffd60a" if conf_pct >= 30.0 else "#9c9c9d"

    return f"""
    <div class="diagnostic-card">
        <div class="diag-col">
            <div class="diag-label">Raw Candidate & Confidence</div>
            <div class="diag-value">
                <span style="color: #f9f9f9; font-weight: 600;">{candidate_name}</span>
                <span class="mono-text" style="color: {conf_color}; font-size: 0.95rem; margin-left: 6px;">{conf_pct:.1f}%</span>
            </div>
        </div>
        <div class="diag-col">
            <div class="diag-label">Motion Gate Status</div>
            <div class="diag-value">{motion_badge}</div>
        </div>
        <div class="diag-col">
            <div class="diag-label">Gate Decision & Vote</div>
            <div class="diag-value">{reason_badge}</div>
        </div>
    </div>
    """
