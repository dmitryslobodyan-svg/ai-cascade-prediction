"""
Каскадный анализ зависимостей на NetworkX (модельные данные).

Логика MVP:
- Граф: A → B → C → D → E
- Узел «отказывает», если риск > NODE_FAILURE_THRESHOLD (50%)
- При отказе каждый зависимый (потомки) получает +CASCADE_RISK_BONUS (30 п.п.)
- Каскад распространяется итеративно, пока появляются новые отказы
- Ущерб = сумма (часы_простоя × стоимость_часа) по затронутым узлам
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from config import (
    CASCADE_CRITICAL_THRESHOLD,
    CASCADE_RISK_BONUS,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    DEFAULT_DOWNTIME_HOURS,
    GRAPH_EDGES,
    NODE_DEFINITIONS,
    NODE_FAILURE_THRESHOLD,
    NODE_ORDER,
    RISK_GREEN_MAX,
    RISK_YELLOW_MAX,
)


@dataclass
class CascadeResult:
    """Результат моделирования каскадного простоя (модельные данные)."""

    # Риск каждого узла после каскада, %
    final_risks: Dict[str, float] = field(default_factory=dict)
    # Узлы, которые отказали (риск > порога)
    failed_nodes: List[str] = field(default_factory=list)
    # Узлы, затронутые каскадом (отказавшие + получившие бонус)
    affected_nodes: List[str] = field(default_factory=list)
    # Цепочка последствий: текстовые шаги
    cascade_chain: List[str] = field(default_factory=list)
    # Общий риск каскадного простоя 0–100% (для прогресс-бара UI)
    overall_cascade_risk: float = 0.0
    # Оценка ущерба в рублях (синтетика)
    estimated_damage_rub: float = 0.0
    # Критичен ли сценарий (overall >= 25%)
    is_critical: bool = False
    # Порядок распространения отказа
    failure_order: List[str] = field(default_factory=list)

    def failed_labels(self) -> List[str]:
        """Человекочитаемые имена отказавших узлов."""
        labels: List[str] = []
        for node_id in self.failed_nodes:
            if node_id in NODE_DEFINITIONS:
                code, name, _ = NODE_DEFINITIONS[node_id]
                labels.append(f"{code}. {name}")
            else:
                labels.append(node_id)
        return labels


def build_dependency_graph(
    edges: Optional[List[Tuple[str, str]]] = None,
    node_ids: Optional[List[str]] = None,
) -> nx.DiGraph:
    """
    Строит ориентированный граф зависимостей производственной линии.

    Returns
    -------
    nx.DiGraph
        Диграф: ребро u→v значит «отказ u влияет на v».
    """
    graph = nx.DiGraph()
    ids = node_ids or list(NODE_ORDER)
    graph.add_nodes_from(ids)

    for src, dst in edges or GRAPH_EDGES:
        graph.add_edge(src, dst)

    # Атрибуты для подписей
    for node_id in graph.nodes:
        if node_id in NODE_DEFINITIONS:
            code, name, short = NODE_DEFINITIONS[node_id]
            graph.nodes[node_id]["label"] = f"{code}. {short}"
            graph.nodes[node_id]["full_name"] = name
        else:
            graph.nodes[node_id]["label"] = str(node_id)
            graph.nodes[node_id]["full_name"] = str(node_id)

    return graph


def risk_to_color(risk_percent: float) -> str:
    """Цвет узла для Pyvis по текущему риску."""
    if risk_percent < RISK_GREEN_MAX:
        return COLOR_GREEN
    if risk_percent <= RISK_YELLOW_MAX:
        return COLOR_YELLOW
    return COLOR_RED


def _node_label(node_id: str) -> str:
    if node_id in NODE_DEFINITIONS:
        code, name, _ = NODE_DEFINITIONS[node_id]
        return f"{code}. {name}"
    return node_id


def simulate_cascade(
    nodes: Dict[str, Dict[str, Any]],
    *,
    failure_threshold: float = NODE_FAILURE_THRESHOLD,
    risk_bonus: float = CASCADE_RISK_BONUS,
    downtime_hours: float = DEFAULT_DOWNTIME_HOURS,
) -> CascadeResult:
    """
    Запускает каскадную симуляцию по графу зависимостей.

    Parameters
    ----------
    nodes :
        Словарь узлов с полем failure_probability (уже посчитанным в ml_model).
    failure_threshold :
        Порог отказа узла, %.
    risk_bonus :
        Надбавка к риску зависимых узлов при отказе предка, п.п.
    downtime_hours :
        Модельная длительность простоя затронутого узла, ч.

    Returns
    -------
    CascadeResult
        Итоговые риски, цепочка, ущерб, флаг критичности.
    """
    try:
        graph = build_dependency_graph()
        risks: Dict[str, float] = {
            node_id: float(nodes[node_id].get("failure_probability", 0.0))
            for node_id in NODE_ORDER
            if node_id in nodes
        }

        failed: Set[str] = set()
        affected: Set[str] = set()
        failure_order: List[str] = []
        chain: List[str] = []

        # Итеративное распространение: пока находятся новые отказы
        changed = True
        max_iterations = len(NODE_ORDER) + 2  # защита от бесконечного цикла
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Новые отказы на этой итерации (стабильный порядок NODE_ORDER)
            newly_failed = [
                nid
                for nid in NODE_ORDER
                if nid in risks
                and nid not in failed
                and risks[nid] > failure_threshold
            ]

            for nid in newly_failed:
                failed.add(nid)
                affected.add(nid)
                failure_order.append(nid)
                chain.append(
                    f"Узел {_node_label(nid)} отказал "
                    f"(риск {risks[nid]:.1f}% > {failure_threshold:.0f}%)"
                )
                changed = True

                # Все прямые и транзитивные потомки получают бонус
                descendants = nx.descendants(graph, nid)
                for child in descendants:
                    if child not in risks:
                        continue
                    old = risks[child]
                    risks[child] = min(100.0, old + risk_bonus)
                    affected.add(child)
                    chain.append(
                        f"  → {_node_label(child)}: риск "
                        f"{old:.1f}% → {risks[child]:.1f}% (+{risk_bonus:.0f} п.п.)"
                    )

        # Ущерб: модельные данные
        damage = 0.0
        for nid in sorted(affected, key=lambda x: NODE_ORDER.index(x) if x in NODE_ORDER else 99):
            hourly = float(nodes[nid].get("hourly_cost", 0.0))
            damage += hourly * downtime_hours

        overall = _compute_overall_cascade_risk(risks, failed, affected, graph)
        is_critical = overall >= CASCADE_CRITICAL_THRESHOLD

        return CascadeResult(
            final_risks={k: round(v, 1) for k, v in risks.items()},
            failed_nodes=[n for n in NODE_ORDER if n in failed],
            affected_nodes=[n for n in NODE_ORDER if n in affected],
            cascade_chain=chain,
            overall_cascade_risk=round(overall, 1),
            estimated_damage_rub=round(damage, 0),
            is_critical=is_critical,
            failure_order=failure_order,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Ошибка каскадного анализа: {exc}. Проверьте данные узлов."
        ) from exc


def _compute_overall_cascade_risk(
    risks: Dict[str, float],
    failed: Set[str],
    affected: Set[str],
    graph: nx.DiGraph,
) -> float:
    """
    Сводный риск каскадного простоя для прогресс-бара UI (0–100).

    Учитывает:
    - максимальный риск в цепочке;
    - долю отказавших / затронутых узлов;
    - «позицию» отказа (чем раньше в цепочке — тем выше каскадный потенциал).
    """
    if not risks:
        return 0.0

    max_risk = max(risks.values())
    avg_risk = sum(risks.values()) / len(risks)
    fail_ratio = len(failed) / max(len(risks), 1)
    affect_ratio = len(affected) / max(len(risks), 1)

    # Позиционный множитель: отказ ближе к началу цепочки опаснее
    position_penalty = 0.0
    for nid in failed:
        if nid in NODE_ORDER:
            # A=0 → высокий штраф, E=4 → ниже
            idx = NODE_ORDER.index(nid)
            position_penalty += (len(NODE_ORDER) - idx) / len(NODE_ORDER) * 15.0

    # Если есть отказ — поднимаем пол каскадного риска
    cascade_floor = 35.0 if failed else 0.0

    overall = (
        0.35 * max_risk
        + 0.25 * avg_risk
        + 0.20 * fail_ratio * 100.0
        + 0.10 * affect_ratio * 100.0
        + 0.10 * min(position_penalty, 40.0)
    )
    overall = max(overall, cascade_floor)

    # Без отказов, но высокий риск у узла — всё равно показываем рост шкалы
    if not failed and max_risk > 25.0:
        overall = max(overall, max_risk * 0.7)

    return float(min(100.0, overall))


def apply_cascade_risks_to_nodes(
    nodes: Dict[str, Dict[str, Any]],
    cascade: CascadeResult,
) -> Dict[str, Dict[str, Any]]:
    """
    Возвращает копию nodes с failure_probability после каскада.
    Удобно для таблицы/графа на вкладке «Каскадный прогноз».
    """
    updated = deepcopy(nodes)
    for node_id, risk in cascade.final_risks.items():
        if node_id in updated:
            updated[node_id]["failure_probability"] = risk
            updated[node_id]["failed"] = node_id in cascade.failed_nodes
            updated[node_id]["affected"] = node_id in cascade.affected_nodes
    return updated


def prepare_pyvis_payload(
    nodes: Dict[str, Dict[str, Any]],
    cascade: Optional[CascadeResult] = None,
) -> Dict[str, Any]:
    """
    Готовит данные для отрисовки интерактивного графа (Pyvis).

    Returns
    -------
    dict
        {
          "nodes": [{id, label, title, color, risk}, ...],
          "edges": [{from, to}, ...],
        }
    """
    risks: Dict[str, float]
    if cascade is not None:
        risks = cascade.final_risks
    else:
        risks = {
            nid: float(nodes[nid].get("failure_probability", 0.0))
            for nid in NODE_ORDER
            if nid in nodes
        }

    pyvis_nodes: List[Dict[str, Any]] = []
    for nid in NODE_ORDER:
        if nid not in nodes:
            continue
        node = nodes[nid]
        risk = float(risks.get(nid, node.get("failure_probability", 0.0)))
        code, full_name, short = NODE_DEFINITIONS[nid]
        title = (
            f"<b>{code}. {full_name}</b><br>"
            f"Температура: {node['temperature']} °C<br>"
            f"Нагрузка: {node['load']} %<br>"
            f"Наработка: {node['hours']} ч<br>"
            f"Риск отказа: {risk:.1f} %<br>"
            f"<i>модельные данные</i>"
        )
        pyvis_nodes.append(
            {
                "id": nid,
                "label": f"{code}\n{short}\n{risk:.0f}%",
                "title": title,
                "color": risk_to_color(risk),
                "risk": risk,
            }
        )

    pyvis_edges = [{"from": src, "to": dst} for src, dst in GRAPH_EDGES]
    return {"nodes": pyvis_nodes, "edges": pyvis_edges}


def summarize_cascade(cascade: CascadeResult) -> str:
    """Краткое текстовое резюме для логов / отладки."""
    lines = [
        f"Общий риск каскада: {cascade.overall_cascade_risk:.1f}% "
        f"({'КРИТИЧНО' if cascade.is_critical else 'стабильно'})",
        f"Отказавшие узлы: {', '.join(cascade.failed_labels()) or 'нет'}",
        f"Затронуто узлов: {len(cascade.affected_nodes)}",
        f"Оценка ущерба: ~{cascade.estimated_damage_rub:,.0f} руб. (модельные данные)",
    ]
    if cascade.cascade_chain:
        lines.append("Цепочка:")
        lines.extend(cascade.cascade_chain)
    return "\n".join(lines)
