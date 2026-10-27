"""
«Умная заглушка» ИИ-рекомендаций (без внешних API).

Генерирует персонализированный текст в формате «ИИ-анализ» на основе
правил Python: риски узлов, результат каскада, причины отклонений параметров.
Все цифры — модельные данные.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from config import (
    CASCADE_CRITICAL_THRESHOLD,
    HOURS_CRITICAL,
    LOAD_CRITICAL,
    NODE_DEFINITIONS,
    NODE_ORDER,
    TEMP_CRITICAL,
)
from cascade_engine import CascadeResult
from ml_model import explain_node_risk


# Рекомендации по типу узла (конкретные действия для демо)
_NODE_ACTIONS: Dict[str, List[str]] = {
    "A": [
        "Немедленно остановить подачу сырья и проверить зону загрузки на повреждения",
        "Провести алко-/допуск-контроль водителей и сменить бригаду загрузки",
        "Переключить подачу на резервный бункер (если доступен)",
    ],
    "B": [
        "Снизить скорость конвейера до 60–70% номинала до стабилизации",
        "Проверить натяжение ленты, ролики и датчики перегрузки",
        "Очистить зону скопления материала, устранить заторы",
    ],
    "C": [
        "Аварийно снизить нагрузку основного станка и включить охлаждение",
        "Сверить показания датчика температуры с ручным пирометром",
        "При подтверждённом перегреве — плановая остановка и техосмотр шпинделя",
    ],
    "D": [
        "Приостановить упаковочную линию до восстановления входного потока",
        "Проверить узлы термоусадки/этикетирования на перегрев",
        "Подготовить ручной контур упаковки для критичных заказов",
    ],
    "E": [
        "Заморозить отгрузку и уведомить логистику о возможной задержке",
        "Перераспределить готовые партии со склада буфера",
        "Согласовать окно отгрузки с клиентами (SLA)",
    ],
}

_PREVENTIVE_GENERIC: List[str] = [
    "Поддерживать температуру узлов ниже 100°C и нагрузку ниже 80%",
    "Планировать ТО при наработке > 5000 ч (приоритет — основной станок)",
    "Раз в смену сверять датчики температуры и нагрузки с контрольными точками",
    "Держать резерв по сырью и буфер на отгрузке на случай локального сбоя",
]


def _label(node_id: str) -> str:
    if node_id in NODE_DEFINITIONS:
        code, name, _ = NODE_DEFINITIONS[node_id]
        return f"{code}. {name}"
    return node_id


def _format_rub(amount: float) -> str:
    """Форматирование рублей с разделителями тысяч."""
    return f"{amount:,.0f}".replace(",", " ")


def _critical_nodes(
    risks: Dict[str, float],
    cascade: CascadeResult,
    threshold: float = 25.0,
) -> List[str]:
    """Узлы с высоким риском или уже в отказе/затронутые каскадом."""
    critical: List[str] = []
    for nid in NODE_ORDER:
        risk = float(risks.get(nid, cascade.final_risks.get(nid, 0.0)))
        if (
            nid in cascade.failed_nodes
            or nid in cascade.affected_nodes
            or risk >= threshold
        ):
            if nid not in critical:
                critical.append(nid)
    return critical


def _build_actions(
    critical_ids: Sequence[str],
    nodes: Optional[Dict[str, Dict[str, Any]]],
    limit: int = 5,
) -> List[str]:
    """Собирает конкретные рекомендации по критическим узлам."""
    actions: List[str] = []
    seen: set[str] = set()

    for nid in critical_ids:
        for action in _NODE_ACTIONS.get(nid, []):
            if action not in seen:
                actions.append(action)
                seen.add(action)
            if len(actions) >= limit:
                return actions

    # Если критических мало — добавить превентивные
    for action in _PREVENTIVE_GENERIC:
        if action not in seen:
            actions.append(action)
            seen.add(action)
        if len(actions) >= limit:
            break

    # Уточнения по фактическим параметрам
    if nodes:
        for nid in critical_ids:
            if nid not in nodes:
                continue
            node = nodes[nid]
            temp = float(node.get("temperature", 0))
            load = float(node.get("load", 0))
            hours = float(node.get("hours", 0))
            if temp > TEMP_CRITICAL:
                tip = (
                    f"Снизить температуру {_label(nid)} "
                    f"(сейчас {temp:.0f}°C > {TEMP_CRITICAL:.0f}°C)"
                )
                if tip not in seen:
                    actions.append(tip)
                    seen.add(tip)
            if load > LOAD_CRITICAL:
                tip = (
                    f"Уменьшить нагрузку {_label(nid)} "
                    f"(сейчас {load:.0f}% > {LOAD_CRITICAL:.0f}%)"
                )
                if tip not in seen:
                    actions.append(tip)
                    seen.add(tip)
            if hours > HOURS_CRITICAL:
                tip = (
                    f"Запланировать ТО для {_label(nid)} "
                    f"(наработка {hours:.0f} ч)"
                )
                if tip not in seen:
                    actions.append(tip)
                    seen.add(tip)

    return actions[:limit]


def _chain_description(cascade: CascadeResult) -> str:
    """Человекочитаемое описание цепочки последствий."""
    if cascade.failure_order:
        path = " → ".join(_label(n) for n in cascade.failure_order)
        affected = [
            _label(n)
            for n in cascade.affected_nodes
            if n not in cascade.failed_nodes
        ]
        if affected:
            return (
                f"Первичный отказ: {path}. "
                f"Под ударом зависимости: {', '.join(affected)}."
            )
        return f"Первичный отказ: {path}."

    if cascade.cascade_chain:
        return " ".join(cascade.cascade_chain[:4])

    return "Каскадных отказов не зафиксировано; наблюдаются локальные отклонения."


def generate_recommendations(
    risks: Dict[str, float],
    cascade_result: CascadeResult,
    nodes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Генерирует текст «ИИ-анализа» по правилам (умная заглушка).

    Parameters
    ----------
    risks :
        Словарь {node_id: вероятность отказа %}. Обычно — final_risks каскада
        или исходные риски до каскада.
    cascade_result :
        Результат simulate_cascade().
    nodes :
        Опционально — полные параметры узлов для более точных советов.

    Returns
    -------
    str
        Многострочный текст для вкладки «ИИ-Анализ и рекомендации».
    """
    try:
        # Нормализуем риски: предпочитаем final_risks каскада
        merged_risks: Dict[str, float] = {
            nid: float(cascade_result.final_risks.get(nid, risks.get(nid, 0.0)))
            for nid in NODE_ORDER
        }
        for nid, val in risks.items():
            merged_risks.setdefault(nid, float(val))

        critical_ids = _critical_nodes(merged_risks, cascade_result)
        critical_labels = [_label(n) for n in critical_ids] or ["нет"]

        if cascade_result.is_critical or (
            cascade_result.overall_cascade_risk >= CASCADE_CRITICAL_THRESHOLD
        ):
            return _format_critical(
                merged_risks=merged_risks,
                cascade=cascade_result,
                critical_labels=critical_labels,
                critical_ids=critical_ids,
                nodes=nodes,
            )
        return _format_stable(
            merged_risks=merged_risks,
            cascade=cascade_result,
            nodes=nodes,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            "🤖 ИИ-АНАЛИЗ СИТУАЦИИ:\n"
            f"Не удалось сформировать рекомендации: {exc}.\n"
            "Проверьте входные данные (модельные данные)."
        )


def _format_critical(
    *,
    merged_risks: Dict[str, float],
    cascade: CascadeResult,
    critical_labels: List[str],
    critical_ids: List[str],
    nodes: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    actions = _build_actions(critical_ids, nodes, limit=5)
    numbered = "\n".join(f"  {i}. {a}" for i, a in enumerate(actions, start=1))

    reasons_block = ""
    if nodes:
        lines: List[str] = []
        for nid in critical_ids[:3]:
            if nid not in nodes:
                continue
            why = "; ".join(explain_node_risk(nodes[nid]))
            lines.append(f"  • {_label(nid)}: {why}")
        if lines:
            reasons_block = "\nДИАГНОСТИКА УЗЛОВ:\n" + "\n".join(lines)

    damage = _format_rub(cascade.estimated_damage_rub)
    risk_lines = ", ".join(
        f"{_label(n)} — {merged_risks.get(n, 0):.1f}%" for n in critical_ids
    )

    return (
        "🤖 ИИ-АНАЛИЗ СИТУАЦИИ:\n"
        "На основании текущих параметров системы выявлен высокий риск "
        "каскадного простоя.\n"
        f"Общий риск каскада: {cascade.overall_cascade_risk:.1f}% "
        f"(порог {CASCADE_CRITICAL_THRESHOLD:.0f}%).\n"
        f"КРИТИЧЕСКИЕ УЗЛЫ: {', '.join(critical_labels)}\n"
        f"  ({risk_lines})\n"
        f"ЦЕПОЧКА ПОСЛЕДСТВИЙ: {_chain_description(cascade)}"
        f"{reasons_block}\n"
        "РЕКОМЕНДАЦИИ:\n"
        f"{numbered}\n"
        f"ОЦЕНКА УЩЕРБА: ~{damage} рублей (модельные данные)\n"
        "⚠️ Все оценки синтетические и предназначены для учебной демонстрации."
    )


def _format_stable(
    *,
    merged_risks: Dict[str, float],
    cascade: CascadeResult,
    nodes: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    # Топ узлов по риску для превентивных советов
    ranked = sorted(
        NODE_ORDER,
        key=lambda n: merged_risks.get(n, 0.0),
        reverse=True,
    )
    top = ranked[:2]
    actions = _build_actions(top, nodes, limit=3)
    # В стабильном режиме — больше превентивных
    for tip in _PREVENTIVE_GENERIC:
        if tip not in actions:
            actions.append(tip)
        if len(actions) >= 4:
            break
    numbered = "\n".join(f"  {i}. {a}" for i, a in enumerate(actions[:4], start=1))

    risk_overview = ", ".join(
        f"{_label(n)} {merged_risks.get(n, 0):.1f}%" for n in NODE_ORDER
    )

    return (
        "🤖 ИИ-АНАЛИЗ СИТУАЦИИ:\n"
        "На основании текущих параметров система стабильна. "
        "Критический каскадный простой не прогнозируется.\n"
        f"Общий риск каскада: {cascade.overall_cascade_risk:.1f}% "
        f"(порог тревоги {CASCADE_CRITICAL_THRESHOLD:.0f}%).\n"
        f"ТЕКУЩИЕ РИСКИ: {risk_overview}\n"
        "КРИТИЧЕСКИЕ УЗЛЫ: нет\n"
        "ЦЕПОЧКА ПОСЛЕДСТВИЙ: отказов не зафиксировано.\n"
        "ПРЕВЕНТИВНЫЕ РЕКОМЕНДАЦИИ:\n"
        f"{numbered}\n"
        "ОЦЕНКА УЩЕРБА: ~0 рублей (модельные данные)\n"
        "✅ Режим мониторинга. Все оценки синтетические (учебный MVP)."
    )


def generate_short_alert(cascade_result: CascadeResult) -> str:
    """Короткий текст для st.warning / st.success на вкладке ИИ."""
    if cascade_result.is_critical:
        n = len(cascade_result.failed_nodes) or len(cascade_result.affected_nodes)
        return (
            f"⚠️ КРИТИЧЕСКИЙ РИСК КАСКАДНОГО ПРОСТОЯ "
            f"({cascade_result.overall_cascade_risk:.1f}%). "
            f"Под угрозой узлов: {n}. "
            f"Ущерб ≈ {_format_rub(cascade_result.estimated_damage_rub)} ₽ "
            f"(модельные данные)."
        )
    return (
        f"✅ Система стабильна "
        f"(риск каскада {cascade_result.overall_cascade_risk:.1f}% "
        f"< {CASCADE_CRITICAL_THRESHOLD:.0f}%)."
    )
