"""
ML-модуль: расчёт вероятности отказа узла (модельные данные).

Подход для учебного MVP:
1. Эвристические правила из config (прозрачная логика для демо).
2. LogisticRegression (scikit-learn) на синтетической выборке —
   уточняет итоговую вероятность в диапазоне 0–100%.

Внешние API не используются.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_BASE_RISK,
    HOURS_CRITICAL,
    HOURS_RISK_BONUS,
    LOAD_CRITICAL,
    LOAD_RISK_BONUS,
    NODE_ORDER,
    TEMP_CRITICAL,
    TEMP_RISK_BONUS,
)


class NodeFailureModel:
    """
    Простая модель отказа узла на базе LogisticRegression.

    Обучается один раз на синтетических данных (модельные данные),
    затем используется для оценки P(отказ) по признакам узла.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler: StandardScaler = StandardScaler()
        self.model: LogisticRegression = LogisticRegression(
            max_iter=500,
            random_state=random_state,
        )
        self._is_fitted: bool = False

    def _make_synthetic_dataset(
        self,
        n_samples: int = 800,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует синтетическую обучающую выборку.

        Признаки: [temperature, load, hours, base_risk]
        Метка: 1 — отказ, 0 — норма (по правилам + шум).
        """
        rng = np.random.default_rng(self.random_state)

        temperature = rng.uniform(20.0, 150.0, size=n_samples)
        load = rng.uniform(0.0, 100.0, size=n_samples)
        hours = rng.uniform(0.0, 10_000.0, size=n_samples)
        base_risk = rng.uniform(0.0, 40.0, size=n_samples)

        # Правиловая «истинная» вероятность отказа (0–100)
        rule_risk = np.array(
            [
                rule_based_risk(t, l, h, b)
                for t, l, h, b in zip(temperature, load, hours, base_risk)
            ],
            dtype=float,
        )
        # Бинарная метка: отказ, если rule_risk > 50% (с небольшим шумом)
        noise = rng.normal(0.0, 5.0, size=n_samples)
        y = ((rule_risk + noise) > 50.0).astype(int)

        X = np.column_stack([temperature, load, hours, base_risk])
        return X, y

    def fit(self) -> "NodeFailureModel":
        """Обучает модель на синтетических модельных данных."""
        X, y = self._make_synthetic_dataset()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict_proba_percent(
        self,
        temperature: float,
        load: float,
        hours: float,
        base_risk: float,
    ) -> float:
        """
        Возвращает вероятность отказа в процентах (0–100).

        Если модель ещё не обучена — обучает её лениво.
        """
        if not self._is_fitted:
            self.fit()

        X = np.array([[temperature, load, hours, base_risk]], dtype=float)
        X_scaled = self.scaler.transform(X)
        # predict_proba[:, 1] — вероятность класса «отказ»
        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        return float(np.clip(proba * 100.0, 0.0, 100.0))


# Глобальный экземпляр модели (ленивая инициализация при первом вызове)
_MODEL: Optional[NodeFailureModel] = None


def get_model() -> NodeFailureModel:
    """Возвращает обученный синглтон модели."""
    global _MODEL
    if _MODEL is None:
        _MODEL = NodeFailureModel().fit()
    return _MODEL


def rule_based_risk(
    temperature: float,
    load: float,
    hours: float,
    base_risk: float = DEFAULT_BASE_RISK,
) -> float:
    """
    Эвристический расчёт риска узла (модельные данные).

    Формула:
        риск = base_risk
             + 20 п.п., если температура > 100°C
             + 15 п.п., если нагрузка > 80%
             + 10 п.п., если наработка > 5000 ч
             + плавная составляющая от параметров

    Returns
    -------
    float
        Риск отказа в процентах [0, 100].
    """
    risk = float(base_risk)

    # Пороговые надбавки из ТЗ / config
    if temperature > TEMP_CRITICAL:
        risk += TEMP_RISK_BONUS
    if load > LOAD_CRITICAL:
        risk += LOAD_RISK_BONUS
    if hours > HOURS_CRITICAL:
        risk += HOURS_RISK_BONUS

    # Плавный вклад (чтобы слайдеры давали заметный эффект до порогов)
    # Нормируем параметры к [0, 1] относительно критических значений
    temp_factor = max(0.0, (temperature - 60.0) / max(TEMP_CRITICAL - 60.0, 1.0))
    load_factor = max(0.0, (load - 50.0) / max(LOAD_CRITICAL - 50.0, 1.0))
    hours_factor = max(0.0, (hours - 2_000.0) / max(HOURS_CRITICAL - 2_000.0, 1.0))

    risk += 8.0 * min(temp_factor, 1.5)
    risk += 7.0 * min(load_factor, 1.5)
    risk += 5.0 * min(hours_factor, 1.5)

    return float(np.clip(risk, 0.0, 100.0))


def calculate_node_risk(
    temperature: float,
    load: float,
    hours: float,
    base_risk: float = DEFAULT_BASE_RISK,
    *,
    use_ml: bool = True,
) -> float:
    """
    Итоговая вероятность отказа узла (модельные данные), %.

    Комбинирует правиловую оценку и ML-вероятность:
        итоговый_риск = 0.6 * rule + 0.4 * ml
    так демо остаётся интерпретируемым, но формально использует sklearn.
    """
    try:
        rule = rule_based_risk(temperature, load, hours, base_risk)
        if not use_ml:
            return round(rule, 1)

        ml = get_model().predict_proba_percent(
            temperature, load, hours, base_risk
        )
        combined = 0.6 * rule + 0.4 * ml
        return round(float(np.clip(combined, 0.0, 100.0)), 1)
    except Exception as exc:  # noqa: BLE001 — учебный MVP, понятное сообщение
        # Fallback: только правила, чтобы UI не падал
        print(f"[ml_model] Ошибка ML, используем правила: {exc}")
        return round(rule_based_risk(temperature, load, hours, base_risk), 1)


def calculate_all_risks(
    nodes: Dict[str, Dict[str, Any]],
    *,
    use_ml: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Пересчитывает failure_probability для всех узлов.

    Returns
    -------
    dict
        Новая копия nodes с обновлённым полем failure_probability.
    """
    updated = deepcopy(nodes)
    for node_id in NODE_ORDER:
        if node_id not in updated:
            continue
        node = updated[node_id]
        node["failure_probability"] = calculate_node_risk(
            temperature=float(node["temperature"]),
            load=float(node["load"]),
            hours=float(node["hours"]),
            base_risk=float(node.get("base_risk", DEFAULT_BASE_RISK)),
            use_ml=use_ml,
        )
    return updated


def risk_status_color(risk_percent: float) -> str:
    """
    Возвращает ключ статуса по риску: 'green' | 'yellow' | 'red'.

    Пороги: <10% зелёный, 10–25% жёлтый, >25% красный.
    """
    from config import RISK_GREEN_MAX, RISK_YELLOW_MAX

    if risk_percent < RISK_GREEN_MAX:
        return "green"
    if risk_percent <= RISK_YELLOW_MAX:
        return "yellow"
    return "red"


def explain_node_risk(node: Dict[str, Any]) -> List[str]:
    """Краткое текстовое объяснение, почему риск вырос (для UI/ИИ-заглушки)."""
    reasons: List[str] = []
    temp = float(node["temperature"])
    load = float(node["load"])
    hours = float(node["hours"])
    base = float(node.get("base_risk", DEFAULT_BASE_RISK))

    if base > DEFAULT_BASE_RISK + 10:
        reasons.append(
            f"повышенный базовый риск ({base:.0f}%) — возможна авария/угроза"
        )
    if temp > TEMP_CRITICAL:
        reasons.append(
            f"температура {temp:.0f}°C > {TEMP_CRITICAL:.0f}°C (+{TEMP_RISK_BONUS:.0f} п.п.)"
        )
    if load > LOAD_CRITICAL:
        reasons.append(
            f"нагрузка {load:.0f}% > {LOAD_CRITICAL:.0f}% (+{LOAD_RISK_BONUS:.0f} п.п.)"
        )
    if hours > HOURS_CRITICAL:
        reasons.append(
            f"наработка {hours:.0f} ч > {HOURS_CRITICAL:.0f} ч (+{HOURS_RISK_BONUS:.0f} п.п.)"
        )
    if not reasons:
        reasons.append("параметры в допустимом диапазоне")
    return reasons
