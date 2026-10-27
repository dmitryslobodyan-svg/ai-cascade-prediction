"""
Кастомные CSS-стили для Streamlit UI учебного MVP.

Применение: в app.py вызвать apply_custom_styles() один раз в начале.
Цвета согласованы с config (зелёный / жёлтый / красный статусы).
"""

from __future__ import annotations

import streamlit as st

from config import COLOR_GREEN, COLOR_NEUTRAL, COLOR_RED, COLOR_YELLOW


def get_custom_css() -> str:
    """Возвращает CSS-строку для st.markdown(..., unsafe_allow_html=True)."""
    return f"""
<style>
    /* ---- Общий фон и типографика ---- */
    .stApp {{
        background: linear-gradient(165deg, #0f172a 0%, #1e293b 45%, #0f766e 140%);
        color: #e2e8f0;
    }}

    /* Убираем лишний верхний padding */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #134e4a 100%);
        border-right: 1px solid rgba(45, 212, 191, 0.25);
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {{
        color: #ccfbf1 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #5eead4 !important;
    }}

    /* ---- Заголовок дашборда ---- */
    .main-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }}
    .main-subtitle {{
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.25rem;
    }}
    .data-badge {{
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.15);
        color: #5eead4;
        border: 1px solid rgba(45, 212, 191, 0.35);
        margin-left: 0.5rem;
        vertical-align: middle;
    }}

    /* ---- Карточки метрик ---- */
    div[data-testid="stMetric"] {{
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}
    div[data-testid="stMetric"] label {{
        color: #94a3b8 !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: #f1f5f9 !important;
    }}

    /* ---- Вкладки ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
        background: transparent;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(13, 148, 136, 0.35) !important;
        color: #ccfbf1 !important;
        border-bottom: 2px solid #2dd4bf;
    }}

    /* ---- Алерты статусов ---- */
    .alert-critical {{
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.25), rgba(127, 29, 29, 0.35));
        border: 1px solid {COLOR_RED};
        border-left: 6px solid {COLOR_RED};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #fecaca;
        font-weight: 600;
        margin: 0.75rem 0 1rem 0;
        box-shadow: 0 0 24px rgba(239, 68, 68, 0.2);
    }}
    .alert-stable {{
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.2), rgba(20, 83, 45, 0.3));
        border: 1px solid {COLOR_GREEN};
        border-left: 6px solid {COLOR_GREEN};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #bbf7d0;
        font-weight: 600;
        margin: 0.75rem 0 1rem 0;
    }}
    .alert-warning {{
        background: linear-gradient(90deg, rgba(234, 179, 8, 0.2), rgba(113, 63, 18, 0.3));
        border: 1px solid {COLOR_YELLOW};
        border-left: 6px solid {COLOR_YELLOW};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #fef08a;
        font-weight: 600;
        margin: 0.75rem 0 1rem 0;
    }}

    /* ---- Блок ИИ-анализа ---- */
    .ai-box {{
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(45, 212, 191, 0.35);
        border-radius: 12px;
        padding: 1.1rem 1.35rem;
        color: #e2e8f0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.88rem;
        line-height: 1.55;
        white-space: pre-wrap;
        box-shadow: inset 0 0 40px rgba(13, 148, 136, 0.08);
    }}

    /* ---- Индикаторы статуса в таблице ---- */
    .status-pill {{
        display: inline-block;
        min-width: 4.5rem;
        text-align: center;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .status-green {{
        background: rgba(34, 197, 94, 0.2);
        color: {COLOR_GREEN};
        border: 1px solid {COLOR_GREEN};
    }}
    .status-yellow {{
        background: rgba(234, 179, 8, 0.2);
        color: {COLOR_YELLOW};
        border: 1px solid {COLOR_YELLOW};
    }}
    .status-red {{
        background: rgba(239, 68, 68, 0.2);
        color: {COLOR_RED};
        border: 1px solid {COLOR_RED};
    }}

    /* ---- Карточка узла (клик / детали) ---- */
    .node-card {{
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
    }}
    .node-card h4 {{
        margin: 0 0 0.35rem 0;
        color: #f1f5f9;
    }}
    .node-card p {{
        margin: 0.15rem 0;
        color: #94a3b8;
        font-size: 0.9rem;
    }}

    /* ---- Кнопки ---- */
    div.stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(45, 212, 191, 0.4);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(45, 212, 191, 0.25);
    }}
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #0d9488, #0891b2);
        color: #f0fdfa;
    }}

    /* ---- Прогресс-бар акцент ---- */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLOR_GREEN}, {COLOR_YELLOW}, {COLOR_RED});
    }}

    /* ---- Таблицы ---- */
    div[data-testid="stDataFrame"] {{
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ---- Подпись «модельные данные» ---- */
    .model-note {{
        font-size: 0.8rem;
        color: {COLOR_NEUTRAL};
        font-style: italic;
        margin-top: 0.5rem;
    }}

    /* ---- Блокировка интерфейса при критическом риске ---- */
    .lock-banner {{
        position: relative;
        background: rgba(127, 29, 29, 0.45);
        border: 2px dashed {COLOR_RED};
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        color: #fecaca;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1rem 0;
    }}

    /* Скрыть футер Streamlit */
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
</style>
"""


def apply_custom_styles() -> None:
    """Внедряет CSS в текущую Streamlit-страницу."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def status_pill_html(status: str) -> str:
    """
    HTML-бейдж статуса узла.

    Parameters
    ----------
    status :
        'green' | 'yellow' | 'red'
    """
    labels = {
        "green": "🟢 Норма",
        "yellow": "🟡 Внимание",
        "red": "🔴 Риск",
    }
    css_class = {
        "green": "status-green",
        "yellow": "status-yellow",
        "red": "status-red",
    }.get(status, "status-yellow")
    text = labels.get(status, status)
    return f'<span class="status-pill {css_class}">{text}</span>'


def alert_html(message: str, level: str = "stable") -> str:
    """
    HTML-блок алерта.

    level: 'critical' | 'stable' | 'warning'
    """
    css = {
        "critical": "alert-critical",
        "stable": "alert-stable",
        "warning": "alert-warning",
    }.get(level, "alert-warning")
    return f'<div class="{css}">{message}</div>'


def ai_box_html(text: str) -> str:
    """Оборачивает текст ИИ-анализа в стилизованный блок."""
    # Экранируем HTML-спецсимволы минимально
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<div class="ai-box">{safe}</div>'


def lock_banner_html(message: str) -> str:
    """Баннер блокировки при критическом каскадном риске."""
    return f'<div class="lock-banner">{message}</div>'


def page_header_html(title: str, subtitle: str) -> str:
    """Заголовок главной области с бейджем «модельные данные»."""
    return (
        f'<div class="main-title">{title}'
        f'<span class="data-badge">модельные данные</span></div>'
        f'<div class="main-subtitle">{subtitle}</div>'
    )


def node_card_html(
    title: str,
    temperature: float,
    load: float,
    hours: float,
    risk: float,
) -> str:
    """Карточка параметров узла (для деталей при выборе на графе)."""
    return (
        f'<div class="node-card">'
        f"<h4>{title}</h4>"
        f"<p>Температура: <b>{temperature:.1f} °C</b></p>"
        f"<p>Нагрузка: <b>{load:.1f} %</b></p>"
        f"<p>Наработка: <b>{hours:.0f} ч</b></p>"
        f"<p>Вероятность отказа: <b>{risk:.1f} %</b></p>"
        f'<p class="model-note">модельные данные</p>'
        f"</div>"
    )
