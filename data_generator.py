"""
Генерация синтетических (модельных) данных для учебной производственной линии.

Модуль не обращается к внешним API и БД — все значения создаются локально
для демонстрации MVP студентам 2-го курса ГУУ.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_BASE_RISK,
    GRAPH_EDGES,
    HOURS_DEFAULT,
    HOURLY_DOWNTIME_COST,
    LOAD_DEFAULT,
    NODE_DEFINITIONS,
    NODE_ORDER,
    TEMP_DEFAULT,
    THREAT_SCENARIOS,
)


def _clip(value: float, low: float, high: float) -> float:
    """Ограничивает значение диапазоном [low, high]."""
    return float(np.clip(value, low, high))


def create_default_nodes(
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Создаёт словарь узлов со стартовыми модельными параметрами.

    Parameters
    ----------
    rng :
        Опциональный генератор NumPy для лёгкой вариативности демо-данных.
        Если None — используются фиксированные значения из config.

    Returns
    -------
    dict
        Ключ — id узла (A..E), значение — словарь параметров узла.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nodes: Dict[str, Dict[str, Any]] = {}
    for node_id in NODE_ORDER:
        code, full_name, short_name = NODE_DEFINITIONS[node_id]
        # Небольшой шум вокруг дефолтов — чтобы таблица выглядела «живой»
        temp = _clip(TEMP_DEFAULT + float(rng.normal(0, 5)), 20.0, 150.0)
        load = _clip(LOAD_DEFAULT + float(rng.normal(0, 8)), 0.0, 100.0)
        hours = _clip(HOURS_DEFAULT + float(rng.normal(0, 400)), 0.0, 10_000.0)

        nodes[node_id] = {
            "id": node_id,
            "code": code,
            "name": full_name,
            "short_name": short_name,
            "temperature": round(temp, 1),
            "load": round(load, 1),
            "hours": round(hours, 0),
            "base_risk": DEFAULT_BASE_RISK,
            "hourly_cost": HOURLY_DOWNTIME_COST[node_id],
            # failure_probability заполняется позже в ml_model
            "failure_probability": DEFAULT_BASE_RISK,
        }
    return nodes


def nodes_to_dataframe(nodes: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует словарь узлов в DataFrame для отображения в Streamlit.

    Колонки помечены как модельные данные.
    """
    rows: List[Dict[str, Any]] = []
    for node_id in NODE_ORDER:
        node = nodes[node_id]
        rows.append(
            {
                "Узел": f"{node['code']}. {node['name']}",
                "ID": node_id,
                "Температура (°C)": node["temperature"],
                "Нагрузка (%)": node["load"],
                "Наработка (ч)": node["hours"],
                "Базовый риск (%)": node["base_risk"],
                "Вероятность отказа (%)": round(float(node["failure_probability"]), 1),
                "Стоимость часа простоя (руб.)": node["hourly_cost"],
            }
        )
    df = pd.DataFrame(rows)
    # Явная пометка для UI / отчётов
    df.attrs["data_note"] = "модельные данные"
    return df


def get_graph_edges() -> List[tuple[str, str]]:
    """Возвращает копию списка рёбер учебной цепочки A→B→C→D→E."""
    return list(GRAPH_EDGES)


def apply_threat(
    nodes: Dict[str, Dict[str, Any]],
    threat_name: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Применяет сценарий угрозы: резко повышает параметры целевого узла.

    Parameters
    ----------
    nodes :
        Текущее состояние узлов.
    threat_name :
        Ключ из THREAT_SCENARIOS (название угрозы из UI).

    Returns
    -------
    dict
        Новый словарь узлов (исходный не мутируется).

    Raises
    ------
    KeyError
        Если название угрозы неизвестно.
    """
    if threat_name not in THREAT_SCENARIOS:
        known = ", ".join(THREAT_SCENARIOS.keys())
        raise KeyError(
            f"Неизвестная угроза: «{threat_name}». Доступные: {known}"
        )

    scenario = THREAT_SCENARIOS[threat_name]
    target = str(scenario["target_node"])
    updated = deepcopy(nodes)

    if target not in updated:
        raise KeyError(f"Целевой узел «{target}» отсутствует в модели.")

    node = updated[target]
    node["temperature"] = _clip(
        float(node["temperature"]) + float(scenario["temp_delta"]),
        20.0,
        150.0,
    )
    node["load"] = _clip(
        float(node["load"]) + float(scenario["load_delta"]),
        0.0,
        100.0,
    )
    # Резкий рост базового риска — симуляция аварии / сбоя
    node["base_risk"] = _clip(
        float(node["base_risk"]) + float(scenario["risk_boost"]),
        0.0,
        100.0,
    )
    node["temperature"] = round(float(node["temperature"]), 1)
    node["load"] = round(float(node["load"]), 1)
    node["base_risk"] = round(float(node["base_risk"]), 1)

    return updated


def reset_nodes(seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Сбрасывает параметры узлов к стартовому модельному состоянию.

    Parameters
    ----------
    seed :
        Зерно ГПСЧ для воспроизводимости демо.
    """
    rng = np.random.default_rng(seed)
    return create_default_nodes(rng=rng)


def update_node_params(
    nodes: Dict[str, Dict[str, Any]],
    node_id: str,
    *,
    temperature: Optional[float] = None,
    load: Optional[float] = None,
    hours: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Обновляет параметры одного узла из слайдеров Sidebar.

    Не изменяет исходный словарь — возвращает глубокую копию.
    """
    if node_id not in nodes:
        raise KeyError(f"Узел «{node_id}» не найден.")

    updated = deepcopy(nodes)
    node = updated[node_id]

    if temperature is not None:
        node["temperature"] = round(_clip(float(temperature), 20.0, 150.0), 1)
    if load is not None:
        node["load"] = round(_clip(float(load), 0.0, 100.0), 1)
    if hours is not None:
        node["hours"] = round(_clip(float(hours), 0.0, 10_000.0), 0)

    return updated


def list_threat_names() -> List[str]:
    """Список названий угроз для выпадающего списка в UI."""
    return list(THREAT_SCENARIOS.keys())
