"""Cyber Dark HUD Design System & HTML Component Renderers (COMP6826001).

Tokens:
- Canvas: #07080a (Deep Inky Black)
- Surface Cards: #101111 / #141516
- Hairline Border: 1px solid #242728
- Primary Text: #f9f9f9, Secondary: #9c9c9d, Disabled: #6a6b6c
- Accents: Coral Red (#ff6363), Cyan (#56c2ff), Emerald Green (#30d158), Amber Gold (#ffd60a)
- Typography: Inter for UI, JetBrains Mono for telemetry & tabular-nums
"""

HUD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* --- Root & Global Reset --- */
:root {
    --hud-bg: #07080a;
    --hud-surface: #101111;
    --hud-surface-elevated: #161718;
    --hud-border: #242728;
    --hud-border-hover: #35383a;
    --hud-text: #f9f9f9;
    --hud-text-muted: #9c9c9d;
    --hud-text-dim: #6a6b6c;
    --hud-coral: #ff6363;
    --hud-cyan: #56c2ff;
    --hud-green: #30d158;
    --hud-amber: #ffd60a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--hud-bg) !important;
    color: var(--hud-text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.012em;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0b0c0e !important;
    border-right: 1px solid var(--hud-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--hud-text) !important;
}

/* Hide default streamlit clutter */
#MainMenu, footer {visibility: hidden;}

/* Monospace font for telemetry and code */
.mono-text, .stMetricValue, code, pre {
    font-family: 'JetBrains Mono', 'Geist Mono', Consolas, monospace !important;
    font-feature-settings: 'tnum' on, 'zero' on !important;
}

/* --- HUD Header Command Bar --- */
.hud-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background-color: var(--hud-surface);
    border: 1px solid var(--hud-border);
    border-radius: 8px;
    margin-bottom: 16px;
}
.hud-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--hud-text);
    display: flex;
    align-items: center;
    gap: 8px;
}
.hud-title span.dot {
    width: 8px;
    height: 8px;
    background-color: var(--hud-coral);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px rgba(255, 99, 99, 0.6);
}

/* --- Pill Badges for State Machine --- */
.hud-badge {
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
    color: var(--hud-cyan);
    border: 1px solid rgba(86, 194, 255, 0.4);
}
.badge-monitoring {
    background-color: rgba(48, 209, 88, 0.12);
    color: var(--hud-green);
    border: 1px solid rgba(48, 209, 88, 0.4);
}
.badge-triggered {
    background-color: var(--hud-coral);
    color: #ffffff;
    border: 1px solid #ff7b7b;
    box-shadow: 0 0 12px rgba(255, 99, 99, 0.5);
    animation: pulse 0.4s ease-in-out infinite alternate;
}
.badge-cooldown {
    background-color: rgba(255, 214, 10, 0.15);
    color: var(--hud-amber);
    border: 1px solid rgba(255, 214, 10, 0.4);
}

@keyframes pulse {
    from { opacity: 0.85; }
    to { opacity: 1.0; }
}

/* --- Telemetry Grid & Cards --- */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
.telemetry-card {
    background-color: var(--hud-surface);
    border: 1px solid var(--hud-border);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.telemetry-card:hover {
    border-color: var(--hud-border-hover);
}
.telemetry-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--hud-text-muted);
    margin-bottom: 4px;
}
.telemetry-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--hud-text);
    font-family: 'JetBrains Mono', monospace;
}
.telemetry-sub {
    font-size: 0.70rem;
    color: var(--hud-text-dim);
    margin-top: 4px;
}

/* --- Confidence Progress Bars --- */
.conf-container {
    background-color: var(--hud-surface);
    border: 1px solid var(--hud-border);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 16px;
}
.conf-row {
    display: grid;
    grid-template-columns: 110px 1fr 45px;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.conf-row:last-child {
    margin-bottom: 0;
}
.conf-label {
    font-size: 0.78rem;
    color: var(--hud-text-muted);
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.conf-bar-bg {
    background-color: #1a1c1d;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background-color: #3e4246;
    transition: width 0.1s ease;
}
.conf-bar-fill.active {
    background-color: var(--hud-cyan);
}
.conf-bar-fill.triggered {
    background-color: var(--hud-coral);
    box-shadow: 0 0 8px rgba(255, 99, 99, 0.7);
}
.conf-val {
    font-size: 0.75rem;
    color: var(--hud-text);
    font-family: 'JetBrains Mono', monospace;
    text-align: right;
}

/* --- Floating Toast Notification --- */
.toast-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background-color: #1c1416;
    border: 1px solid var(--hud-coral);
    border-radius: 20px;
    padding: 6px 16px;
    color: #ffffff;
    font-weight: 600;
    font-size: 0.9rem;
    backdrop-filter: blur(8px);
}
.toast-pill span.shortcut {
    background-color: #2b1d20;
    color: var(--hud-coral);
    border: 1px solid rgba(255, 99, 99, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* --- Action Log Table --- */
.action-log-card {
    background-color: var(--hud-surface);
    border: 1px solid var(--hud-border);
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
    color: var(--hud-text-dim);
}
.action-log-gesture {
    color: var(--hud-text);
    font-weight: 500;
}
.action-log-key {
    color: var(--hud-cyan);
}

/* --- Section Title --- */
.hud-section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--hud-text-muted);
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
    background-color: var(--hud-surface);
    border: 1px solid var(--hud-border);
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
    color: var(--hud-text-muted);
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

# Alias for backwards compatibility
RAYCAST_CSS = HUD_CSS


def clean_html(raw_html: str) -> str:
    """Strip leading spaces from each line so Markdown parser treats it as pure raw HTML, never an indented code block."""
    lines = [line.strip() for line in raw_html.strip().splitlines() if line.strip()]
    return "".join(lines)


def render_state_badge(state: str) -> str:
    """Render HTML badge warna-warni sesuai state machine."""
    st = str(state).replace("SystemState.", "").upper()
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
    return clean_html(f'<span class="hud-badge {css_class}">{icon} {st}</span>')


def render_confidence_bars(class_names: list, probs: list, threshold: float = 0.80) -> str:
    """Render horizontal bar chart 8 kelas gestur dengan threshold marker (unindented HTML)."""
    p_list = [float(x) for x in probs] if probs is not None and len(probs) > 0 else []
    active_idx = p_list.index(max(p_list)) if p_list else -1

    rows = []
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

        rows.append(
            f'<div class="conf-row">'
            f'<div class="conf-label" title="{name}">{name}</div>'
            f'<div class="conf-bar-bg"><div class="{bar_class}" style="width: {pct:.1f}%;"></div></div>'
            f'<div class="conf-val">{pct:4.1f}%</div>'
            f'</div>'
        )

    content = (
        f'<div class="conf-container">'
        f'<div class="hud-section-title">8-Class Softmax Confidence</div>'
        f'{"".join(rows)}'
        f'</div>'
    )
    return clean_html(content)


def render_telemetry_cards(
    fps: float,
    latency_ms: float,
    buffer_fill: int,
    buffer_max: int,
    cooldown_s: float,
    os_armed: bool,
) -> str:
    """Render 4 kartu metrik monospasi bergaya Dark Cyber HUD."""
    armed_str = '<span style="color: #30d158;">ARMED</span>' if os_armed else '<span style="color: #9c9c9d;">DISARMED</span>'
    cd_str = f"{cooldown_s:.2f}s" if cooldown_s > 0 else "READY"
    cd_color = "#ffd60a" if cooldown_s > 0 else "#30d158"

    content = (
        f'<div class="telemetry-grid">'
        f'<div class="telemetry-card">'
        f'<div class="telemetry-label">Throughput</div>'
        f'<div class="telemetry-value">{fps:4.1f} <span style="font-size: 0.75rem; color: #9c9c9d;">FPS</span></div>'
        f'<div class="telemetry-sub">Target 12.0 FPS</div>'
        f'</div>'
        f'<div class="telemetry-card">'
        f'<div class="telemetry-label">Inference Latency</div>'
        f'<div class="telemetry-value">{latency_ms:4.1f} <span style="font-size: 0.75rem; color: #9c9c9d;">ms</span></div>'
        f'<div class="telemetry-sub">Target &lt; 20 ms</div>'
        f'</div>'
        f'<div class="telemetry-card">'
        f'<div class="telemetry-label">Sliding Buffer</div>'
        f'<div class="telemetry-value">{buffer_fill}/{buffer_max}</div>'
        f'<div class="telemetry-sub">Window ~2.7s</div>'
        f'</div>'
        f'<div class="telemetry-card">'
        f'<div class="telemetry-label">Cooldown / Gate</div>'
        f'<div class="telemetry-value" style="color: {cd_color};">{cd_str}</div>'
        f'<div class="telemetry-sub">OS Control: {armed_str}</div>'
        f'</div>'
        f'</div>'
    )
    return clean_html(content)


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
    """Render status diagnostik real-time tepat di bawah viewport webcam (unindented HTML)."""
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

    # Confidence color
    conf_pct = candidate_conf * 100.0
    conf_color = "#30d158" if conf_pct >= 50.0 else "#ffd60a" if conf_pct >= 30.0 else "#9c9c9d"

    content = (
        f'<div class="diagnostic-card">'
        f'<div class="diag-col">'
        f'<div class="diag-label">Raw Candidate & Confidence</div>'
        f'<div class="diag-value">'
        f'<span style="color: #f9f9f9; font-weight: 600;">{candidate_name}</span>'
        f'<span class="mono-text" style="color: {conf_color}; font-size: 0.95rem; margin-left: 6px;">{conf_pct:.1f}%</span>'
        f'</div>'
        f'</div>'
        f'<div class="diag-col">'
        f'<div class="diag-label">Motion Gate Status</div>'
        f'<div class="diag-value">{motion_badge}</div>'
        f'</div>'
        f'<div class="diag-col">'
        f'<div class="diag-label">Gate Decision & Vote</div>'
        f'<div class="diag-value">{reason_badge}</div>'
        f'</div>'
        f'</div>'
    )
    return clean_html(content)


def render_action_log(action_history: list) -> str:
    """Render riwayat trigger aksi OS dengan styling unindented HTML."""
    if not action_history:
        return clean_html(
            '<div class="action-log-card" style="color: #6a6b6c; font-size: 0.8rem; text-align: center; padding: 20px;">'
            'Belum ada aksi yang dipicu'
            '</div>'
        )

    rows = []
    for item in action_history[:5]:
        rows.append(
            f'<div class="action-log-item">'
            f'<span class="action-log-ts">{item["timestamp"]}</span>'
            f'<span class="action-log-gesture">{item["gesture"]}</span>'
            f'<span class="action-log-key">{item["action"]} ({item["shortcut"]})</span>'
            f'<span style="color: #56c2ff;">{item["confidence"]}</span>'
            f'<span style="color: #6a6b6c;">{item["latency_ms"]}</span>'
            f'</div>'
        )

    content = f'<div class="action-log-card">{"".join(rows)}</div>'
    return clean_html(content)
