"""
Расчёт OCEAN (5 черт) и 30 аспектов (фасетов) по ответам IPIP-NEO-120
с помощью библиотеки five-factor-e (ipipneo).
"""

from __future__ import annotations

from typing import Any

from ipipneo import IpipNeo
from ipipneo.model import FacetLevel, NormCubic
from ipipneo.utility import big5_ocean_is_valid, create_big5_dict

import ipipneo.facet as _ipipneo_facet
from ipipneo import norm as _ipipneo_norm

# Порядок и имена колонок в датасете (5 черт + 30 фасетов)
OCEAN_AND_FACET_ORDER = [
    "openness",
    "facet_imagination",
    "facet_artistic_interests",
    "facet_emotionality",
    "facet_adventurousness",
    "facet_intellect",
    "facet_liberalism",
    "conscientiousness",
    "facet_self_efficacy",
    "facet_orderliness",
    "facet_dutifulness",
    "facet_achievement_striving",
    "facet_self_discipline",
    "facet_cautiousness",
    "extraversion",
    "facet_friendliness",
    "facet_gregariousness",
    "facet_assertiveness",
    "facet_activity_level",
    "facet_excitement_seeking",
    "facet_cheerfulness",
    "agreeableness",
    "facet_trust",
    "facet_morality",
    "facet_altruism",
    "facet_cooperation",
    "facet_modesty",
    "facet_sympathy",
    "neuroticism",
    "facet_anxiety",
    "facet_anger",
    "facet_depression",
    "facet_self_consciousness",
    "facet_immoderation",
    "facet_vulnerability",
]

TRAIT_NAMES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

FACET_NAMES = [c for c in OCEAN_AND_FACET_ORDER if c.startswith("facet_")]

# Маппинг: (домен ipipneo, ключ в выводе) -> имя колонки датасета
# Для черт: (domain, "O"|"C"|"E"|"A"|"N") -> openness|conscientiousness|...
# Для фасетов: ключ в traits (snake_case) -> facet_<key>
_OCEAN_LETTER = {"openness": "O", "conscientiousness": "C", "extraversion": "E", "agreeableness": "A", "neuroticism": "N"}


def answers_dict_to_ipipneo_format(model_answers: dict[int, int]) -> dict[str, list[dict[str, int]]]:
    """
    Конвертирует словарь {question_id: answer} из parse_response
    в формат, ожидаемый five-factor-e: {"answers": [{"id_question": int, "id_select": int}, ...]}.

    Вход:
        model_answers: {question_id: answer}, answer 1–5, question_id 1–120.
    Выход:
        {"answers": [{"id_question": k, "id_select": v}, ...]} по всем ключам.
    """
    return {
        "answers": [
            {"id_question": q, "id_select": a}
            for q, a in sorted(model_answers.items())
        ]
    }


def sex_to_ipipneo(sex: Any) -> str:
    """
    Приводит пол из датасета к формату ipipneo: 'M', 'F' или 'N'.
    В датасете часто: 1 = мужской, 2 = женский.
    """
    if sex is None:
        return "N"
    if isinstance(sex, str):
        s = str(sex).strip().upper()
        if s in ("M", "F", "N"):
            return s
        if s in ("1", "MALE"):
            return "M"
        if s in ("2", "FEMALE"):
            return "F"
        return "N"
    if isinstance(sex, (int, float)):
        if sex == 1:
            return "M"
        if sex == 2:
            return "F"
    return "N"


def _personality_no_clamp(
    self: Any,
    size: int,
    big5: dict,
    traits: dict,
    label: str,
    norm_scale_min: int | None = None,
    norm_scale_max: int | None = None,
    facet_score_level_low: int | None = None,
    facet_score_level_high: int | None = None,
) -> dict:
    """Вариант Facet.personality без обрезки X[i] в 1 и 99 (кубическая формула как есть)."""
    big5_ocean_is_valid(label=label)
    big5, traits = big5.get(label, 0), traits.get(label, [])
    facet_score_level_low_value = (
        facet_score_level_low if facet_score_level_low is not None else FacetLevel.LOW.value
    )
    facet_score_level_high_value = (
        facet_score_level_high if facet_score_level_high is not None else FacetLevel.HIGH.value
    )
    X = [0] * size
    Y = [0] * size
    try:
        for i in range(1, 7):
            Y[i] = traits[i]
            if int(traits[i]) < facet_score_level_low_value:
                Y[i] = "low"
            elif facet_score_level_low_value <= int(traits[i]) <= facet_score_level_high_value:
                Y[i] = "average"
            elif int(traits[i]) > facet_score_level_high_value:
                Y[i] = "high"
            X[i] = (
                NormCubic.CONST1.value
                - (NormCubic.CONST2.value * traits[i])
                + (NormCubic.CONST3.value * traits[i] ** 2)
                - (NormCubic.CONST4.value * traits[i] ** 3)
            )
    except IndexError as e:
        raise BaseException(f"The number of questions setting is wrong: {str(e)}") from e
    return create_big5_dict(label=label, big5=big5, x=X, y=Y) or {}


def _normalize_no_clamp(
    normc: dict,
    percent: dict,
    norm_scale_min: int | None = None,
    norm_scale_max: int | None = None,
) -> dict:
    """Вариант Norm.normalize без обрезки в 1 и 99 — возвращаем percent как есть."""
    return {
        "O": percent.get("O", 0),
        "C": percent.get("C", 0),
        "E": percent.get("E", 0),
        "A": percent.get("A", 0),
        "N": percent.get("N", 0),
    }


def _personalities_to_flat_dict(personalities: list[dict]) -> dict[str, float]:
    """
    Разворачивает person['result']['personalities'] из ipipneo в плоский словарь
    с ключами как в датасете: openness, facet_imagination, ... в порядке OCEAN_AND_FACET_ORDER.
    """
    out: dict[str, float] = {}
    for block in personalities:
        for domain, data in block.items():
            if not isinstance(data, dict):
                continue
            letter = _OCEAN_LETTER.get(domain)
            if letter and letter in data:
                out[domain] = float(data[letter])
            traits_list = data.get("traits") or []
            for t in traits_list:
                if not isinstance(t, dict):
                    continue
                for k, v in t.items():
                    if k in ("trait", "score"):
                        continue
                    if isinstance(v, (int, float)):
                        out[f"facet_{k}"] = float(v)
                        break
    return out


def compute_ocean_facets(
    model_answers: dict[int, int],
    sex: Any,
    age: Any,
    *,
    question: int = 120,
) -> dict[str, float] | None:
    """
    Считает 5 черт OCEAN и 30 фасетов по ответам на IPIP-NEO.

    Вход:
        model_answers: {question_id: answer} (1–120, answer 1–5), как от parse_response.
        sex: пол (1/2 или M/F/N) — для норм ipipneo.
        age: возраст (годы).
        question: 120 или 300.

    Выход:
        Плоский dict с ключами как OCEAN_AND_FACET_ORDER и float-значениями 0–100,
        или None при ошибке/неполных ответах.
    """
    if not model_answers or len(model_answers) < 120:
        return None
    _saved_facet = _ipipneo_facet.Facet.personality
    _saved_norm = _ipipneo_norm.Norm.normalize
    _ipipneo_facet.Facet.personality = _personality_no_clamp
    _ipipneo_norm.Norm.normalize = _normalize_no_clamp
    try:
        ipip = IpipNeo(question=question)
        answers = answers_dict_to_ipipneo_format(model_answers)
        a = age
        if a is None or (isinstance(a, float) and (a != a)):  # NaN
            a = 30
        else:
            try:
                a = int(a)
            except (TypeError, ValueError):
                a = 30
        if not (10 <= a <= 110):
            a = 30
        res = ipip.compute(sex=sex_to_ipipneo(sex), age=a, answers=answers)
    except Exception:
        return None
    finally:
        _ipipneo_facet.Facet.personality = _saved_facet
        _ipipneo_norm.Norm.normalize = _saved_norm
    person = res.get("person") or res
    result = person.get("result") if isinstance(person, dict) else None
    if not result:
        return None
    personalities = result.get("personalities")
    if not personalities or len(personalities) != 5:
        return None
    flat = _personalities_to_flat_dict(personalities)
    # Гарантируем все 35 ключей; отсутствующие — 0.0
    return {k: flat.get(k, 0.0) for k in OCEAN_AND_FACET_ORDER}
