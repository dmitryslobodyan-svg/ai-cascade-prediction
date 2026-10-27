"""
Точка входа Streamlit: ИИ-прогнозирование каскадных простоев.

Учебный MVP (модельные данные). Логика вынесена в модули —
здесь только сборка UI и вызовы функций.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from pyvis.network import Network

import config
from ai_recommendations import generate_recommendations, generate_short_alert
from cascade_engine import (
    CascadeResult,
    apply_cascade_risks_to_nodes,
    prepare_pyvis_payload,
    simulate_cascade,
)
from data_generator import (
    apply_threat,
    create_default_nodes,
    list_threat_names,
    reset_nodes,
)
from ml_model import calculate_all_risks, risk_status_color
from ui_styles import (
    ai_box_html,
    alert_html,
    apply_custom_styles,
    lock_banner_html,
    node_card_html,
    page_header_html,
    status_pill_html,
)


# ---------------------------------------------------------------------------
# Инициализация состояния сессии
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Создаёт стартовые ключи session_state при первом запуске."""
    if "nodes" not in st.session_state:
        st.session_state.nodes = create_default_nodes()
    if "threat_applied" not in st.session_state:
        st.session_state.threat_applied = None
    if "selected_graph_node" not in st.session_state:
        st.session_state.selected_graph_node = "A"


def _recompute() -> tuple[Dict[str, Dict[str, Any]], CascadeResult]:
    """
    Пересчитывает риски узлов и каскад по текущему session_state.nodes.

    Returns
    -------
    tuple
        (nodes_with_risks, cascade_result)
    """
    nodes_with_risks = calculate_all_risks(st.session_state.nodes)
    cascade = simulate_cascade(nodes_with_risks)
    return nodes_with_risks, cascade


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Боковая панель: сценарий, слайдеры, угрозы, сброс."""
    with st.sidebar:
        st.markdown("## 🏭 ИИ-Прогнозирование каскадных простоев")
        st.caption("Учебный MVP · модельные данные")

        st.selectbox(
            "Сценарий",
            options=[config.SCENARIO_NAME],
            index=0,
            help="Пока доступна только учебная производственная линия из 5 узлов",
        )

        st.markdown("---")
        st.markdown("### Параметры узлов")

        # Слайдеры по каждому узлу — обновляют session_state.nodes
        updated = dict(st.session_state.nodes)
        for node_id in config.NODE_ORDER:
            node = updated[node_id]
            _, full_name, _ = config.NODE_DEFINITIONS[node_id]
            with st.expander(f"{node_id}. {full_name}", expanded=(node_id == "A")):
                temp = st.slider(
                    "Температура (°C)",
                    min_value=int(config.TEMP_MIN),
                    max_value=int(config.TEMP_MAX),
                    value=int(round(float(node["temperature"]))),
                    key=f"slider_temp_{node_id}",
                )
                load = st.slider(
                    "Нагрузка (%)",
                    min_value=int(config.LOAD_MIN),
                    max_value=int(config.LOAD_MAX),
                    value=int(round(float(node["load"]))),
                    key=f"slider_load_{node_id}",
                )
                hours = st.slider(
                    "Наработка (часы)",
                    min_value=int(config.HOURS_MIN),
                    max_value=int(config.HOURS_MAX),
                    value=int(round(float(node["hours"]))),
                    step=100,
                    key=f"slider_hours_{node_id}",
                )
                node = dict(node)
                node["temperature"] = float(temp)
                node["load"] = float(load)
                node["hours"] = float(hours)
                updated[node_id] = node

        st.session_state.nodes = updated

        st.markdown("---")
        st.markdown("### Симуляция угрозы")
        threat_name = st.selectbox(
            "Тип угрозы",
            options=list_threat_names(),
            key="threat_select",
        )

        col_t, col_r = st.columns(2)
        with col_t:
            if st.button("🚨 Симулировать угрозу", width="stretch"):
                try:
                    st.session_state.nodes = apply_threat(
                        st.session_state.nodes, threat_name
                    )
                    st.session_state.threat_applied = threat_name
                    # Синхронизация слайдеров с новыми значениями
                    for nid in config.NODE_ORDER:
                        n = st.session_state.nodes[nid]
                        st.session_state[f"slider_temp_{nid}"] = int(
                            round(float(n["temperature"]))
                        )
                        st.session_state[f"slider_load_{nid}"] = int(
                            round(float(n["load"]))
                        )
                        st.session_state[f"slider_hours_{nid}"] = int(
                            round(float(n["hours"]))
                        )
                    st.success(f"Угроза применена: {threat_name}")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось применить угрозу: {exc}")

        with col_r:
            if st.button("🔄 Сбросить параметры", width="stretch"):
                try:
                    st.session_state.nodes = reset_nodes(seed=42)
                    st.session_state.threat_applied = None
                    for nid in config.NODE_ORDER:
                        n = st.session_state.nodes[nid]
                        st.session_state[f"slider_temp_{nid}"] = int(
                            round(float(n["temperature"]))
                        )
                        st.session_state[f"slider_load_{nid}"] = int(
                            round(float(n["load"]))
                        )
                        st.session_state[f"slider_hours_{nid}"] = int(
                            round(float(n["hours"]))
                        )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ошибка сброса: {exc}")

        if st.session_state.threat_applied:
            st.info(f"Активная угроза: **{st.session_state.threat_applied}**")


# ---------------------------------------------------------------------------
# Вкладка 1 — Мониторинг
# ---------------------------------------------------------------------------

def render_monitoring_tab(
    nodes: Dict[str, Dict[str, Any]],
    cascade: CascadeResult,
) -> None:
    """Таблица состояния узлов и ключевые метрики."""
    st.markdown("#### Ключевые показатели")
    st.caption("модельные данные")

    avg_risk = sum(
        float(nodes[n]["failure_probability"]) for n in config.NODE_ORDER
    ) / len(config.NODE_ORDER)
    max_node = max(
        config.NODE_ORDER,
        key=lambda n: float(nodes[n]["failure_probability"]),
    )
    max_risk = float(nodes[max_node]["failure_probability"])
    _, max_name, _ = config.NODE_DEFINITIONS[max_node]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Общий риск каскада", f"{cascade.overall_cascade_risk:.1f}%")
    m2.metric("Средний риск узлов", f"{avg_risk:.1f}%")
    m3.metric("Макс. риск узла", f"{max_risk:.1f}%", delta=max_name)
    m4.metric(
        "Отказавших узлов",
        f"{len(cascade.failed_nodes)}",
        delta="критично" if cascade.failed_nodes else "норма",
        delta_color="inverse" if cascade.failed_nodes else "off",
    )

    st.markdown("#### Состояние узлов")

    # Таблица с HTML-статусами через pandas + column config неудобна для HTML,
    # поэтому строим компактную таблицу + отдельные бейджи.
    rows = []
    for nid in config.NODE_ORDER:
        node = nodes[nid]
        risk = float(node["failure_probability"])
        status = risk_status_color(risk)
        code, name, _ = config.NODE_DEFINITIONS[nid]
        rows.append(
            {
                "Статус": status,
                "Узел": f"{code}. {name}",
                "Температура (°C)": node["temperature"],
                "Нагрузка (%)": node["load"],
                "Наработка (ч)": int(node["hours"]),
                "Вероятность отказа (%)": risk,
            }
        )

    # Отображение статус-бейджей + dataframe
    badge_cols = st.columns(5)
    for i, nid in enumerate(config.NODE_ORDER):
        risk = float(nodes[nid]["failure_probability"])
        status = risk_status_color(risk)
        with badge_cols[i]:
            st.markdown(status_pill_html(status), unsafe_allow_html=True)
            st.caption(f"{nid}: {risk:.1f}%")

    df = pd.DataFrame(rows).drop(columns=["Статус"])
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown(
        '<p class="model-note">Все значения — модельные данные учебной демонстрации.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Вкладка 2 — Каскадный прогноз (Pyvis)
# ---------------------------------------------------------------------------

def _build_pyvis_file(
    nodes: Dict[str, Dict[str, Any]],
    cascade: CascadeResult,
) -> Path:
    """
    Строит HTML интерактивного графа Pyvis и сохраняет во временный файл.

    Файл кладётся в каталог сессии Streamlit, чтобы st.iframe мог его открыть.
    """
    payload = prepare_pyvis_payload(nodes, cascade)
    net = Network(
        height="480px",
        width="100%",
        directed=True,
        bgcolor="#0f172a",
        font_color="#e2e8f0",
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.02,
    )

    for item in payload["nodes"]:
        net.add_node(
            item["id"],
            label=item["label"],
            title=item["title"],
            color=item["color"],
            size=28 + item["risk"] * 0.25,
            borderWidth=2,
        )

    for edge in payload["edges"]:
        net.add_edge(
            edge["from"],
            edge["to"],
            arrows="to",
            color="#94a3b8",
            width=2,
        )

    net.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true
          },
          "nodes": {
            "shape": "dot",
            "font": {"size": 14, "face": "arial", "multi": "html"}
          },
          "edges": {
            "smooth": {"type": "cubicBezier"}
          },
          "physics": {"enabled": true}
        }
        """
    )

    cache_dir = Path(tempfile.gettempdir()) / "cascade_mvp_graphs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cascade_graph.html"
    net.write_html(str(path), notebook=False)
    return path


def render_cascade_tab(
    nodes: Dict[str, Dict[str, Any]],
    cascade: CascadeResult,
) -> None:
    """Интерактивный граф зависимостей и карточка выбранного узла."""
    st.markdown("#### Граф зависимостей")
    st.caption(
        "🟢 <10% · 🟡 10–25% · 🔴 >25% · наведите / выберите узел — параметры "
        "(модельные данные)"
    )

    # Показываем риски ПОСЛЕ каскада
    nodes_after = apply_cascade_risks_to_nodes(nodes, cascade)

    try:
        graph_path = _build_pyvis_file(nodes_after, cascade)
        st.iframe(graph_path, height=520, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось отрисовать граф: {exc}")

    # Клик в iframe не пробрасывается в Streamlit — дублируем выбор узла
    labels = {
        nid: f"{config.NODE_DEFINITIONS[nid][0]}. {config.NODE_DEFINITIONS[nid][1]}"
        for nid in config.NODE_ORDER
    }
    selected = st.selectbox(
        "Показать параметры узла",
        options=config.NODE_ORDER,
        format_func=lambda x: labels[x],
        key="selected_graph_node",
        help="Аналог клика по узлу графа (Streamlit не получает клики из Pyvis iframe)",
    )

    node = nodes_after[selected]
    risk = float(cascade.final_risks.get(selected, node["failure_probability"]))
    st.markdown(
        node_card_html(
            title=labels[selected],
            temperature=float(node["temperature"]),
            load=float(node["load"]),
            hours=float(node["hours"]),
            risk=risk,
        ),
        unsafe_allow_html=True,
    )

    if cascade.cascade_chain:
        with st.expander("Цепочка каскада", expanded=bool(cascade.failed_nodes)):
            for step in cascade.cascade_chain:
                st.text(step)


# ---------------------------------------------------------------------------
# Вкладка 3 — ИИ-анализ
# ---------------------------------------------------------------------------

def render_ai_tab(
    nodes: Dict[str, Dict[str, Any]],
    cascade: CascadeResult,
) -> None:
    """Прогресс общего риска, алерты, рекомендации, ущерб."""
    st.markdown("#### Общий риск каскадного простоя")
    risk_pct = float(cascade.overall_cascade_risk)
    st.progress(min(max(risk_pct / 100.0, 0.0), 1.0))
    st.write(f"**{risk_pct:.1f}%** · порог критичности: {config.CASCADE_CRITICAL_THRESHOLD:.0f}%")
    st.caption("модельные данные")

    if cascade.is_critical or risk_pct >= config.CASCADE_CRITICAL_THRESHOLD:
        st.markdown(
            alert_html(
                "⚠️ КРИТИЧЕСКИЙ РИСК КАСКАДНОГО ПРОСТОЯ",
                level="critical",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            lock_banner_html(
                "🔒 Интерфейс в режиме аварийного предупреждения. "
                "Примите меры по списку рекомендаций ниже. "
                "(учебная блокировка · модельные данные)"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("##### Узлы, которые встанут (цепочка)")
        if cascade.failed_nodes or cascade.affected_nodes:
            chain_ids = cascade.failed_nodes or cascade.failure_order
            # Показать полный путь влияния
            stop_list = cascade.affected_nodes or chain_ids
            for nid in stop_list:
                code, name, _ = config.NODE_DEFINITIONS[nid]
                mark = "⛔ отказ" if nid in cascade.failed_nodes else "⏸ простой"
                final_r = cascade.final_risks.get(nid, 0.0)
                st.write(f"- **{code}. {name}** — {mark}, риск {final_r:.1f}%")
        else:
            st.write("Прямых отказов нет, но сводный риск превысил порог.")

        damage = cascade.estimated_damage_rub
        st.metric(
            "Оценка ущерба",
            f"~{damage:,.0f} ₽".replace(",", " "),
            help="Синтетическая оценка (модельные данные)",
        )

        text = generate_recommendations(cascade.final_risks, cascade, nodes)
        st.markdown("##### 🤖 ИИ-Анализ и рекомендации")
        st.markdown(ai_box_html(text), unsafe_allow_html=True)

    else:
        st.markdown(
            alert_html("✅ Система стабильна", level="stable"),
            unsafe_allow_html=True,
        )
        st.info(generate_short_alert(cascade))
        text = generate_recommendations(cascade.final_risks, cascade, nodes)
        st.markdown("##### Превентивные рекомендации")
        st.markdown(ai_box_html(text), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа приложения."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        apply_custom_styles()
        _init_session_state()
        render_sidebar()

        st.markdown(
            page_header_html(config.APP_TITLE, config.APP_SUBTITLE),
            unsafe_allow_html=True,
        )

        nodes, cascade = _recompute()

        # Короткий статус на главном экране
        if cascade.is_critical:
            st.markdown(
                alert_html(generate_short_alert(cascade), level="critical"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                alert_html(generate_short_alert(cascade), level="stable"),
                unsafe_allow_html=True,
            )

        tab_mon, tab_cas, tab_ai = st.tabs(
            [
                "📊 Мониторинг",
                "🔗 Каскадный прогноз",
                "🤖 ИИ-Анализ и рекомендации",
            ]
        )

        with tab_mon:
            render_monitoring_tab(nodes, cascade)
        with tab_cas:
            render_cascade_tab(nodes, cascade)
        with tab_ai:
            render_ai_tab(nodes, cascade)

    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Критическая ошибка приложения: {exc}. "
            "Проверьте модули и зависимости (модельные данные)."
        )
        raise


if __name__ == "__main__":
    main()
