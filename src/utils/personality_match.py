import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.stats as sps
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import cohen_kappa_score

from src.utils import five_factor
from src.utils.parse import parse_response
from src.utils.prompt import build_full_prompt

# Имена 35 измерений и подмножества
OCEAN_AND_FACET_ORDER = five_factor.OCEAN_AND_FACET_ORDER
TRAIT_NAMES = five_factor.TRAIT_NAMES
FACET_NAMES = five_factor.FACET_NAMES

# Для IPIP-120 вопросы идут циклом N, E, O, A, C.
_TRAIT_BY_QID_MOD = {
    1: "neuroticism",
    2: "extraversion",
    3: "openness",
    4: "agreeableness",
    0: "conscientiousness",
}
_TRAIT_BLOCK_KEY = {
    "openness": "openness_items_24",
    "conscientiousness": "conscientiousness_items_24",
    "extraversion": "extraversion_items_24",
    "agreeableness": "agreeableness_items_24",
    "neuroticism": "neuroticism_items_24",
}


def build_trait_question_blocks(total_questions: int = 120) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {trait: [] for trait in TRAIT_NAMES}
    for q_id in range(1, total_questions + 1):
        trait = _TRAIT_BY_QID_MOD[q_id % 5]
        grouped[trait].append(q_id)
    return {
        _TRAIT_BLOCK_KEY[trait]: grouped[trait]
        for trait in TRAIT_NAMES
    }


TRAIT_QUESTION_BLOCKS = build_trait_question_blocks(total_questions=120)


def get_trait_question_blocks() -> dict[str, list[int]]:
    return {k: list(v) for k, v in TRAIT_QUESTION_BLOCKS.items()}


def _is_valid_number(v) -> bool:
    return isinstance(v, (int, float)) and not (v != v or np.isnan(v))  # noqa: E711


def _safe_mean(arr: list[float]) -> float:
    if not arr:
        return 0.0
    x = float(np.nanmean(arr))
    return 0.0 if (x != x or np.isnan(x)) else x  # noqa: E711


def _to_optional_float(v) -> float | None:
    return float(v) if _is_valid_number(v) else None


def _value_to_category(x: float) -> str:
    """ low <33, average 33–66, high >66 """
    if x < 33:
        return "low"
    if x <= 66:
        return "average"
    return "high"


def compute_five_factor_metrics(
    real_flat: dict[str, float],
    simulated_flat: dict[str, float],
    keys: list[str] | None = None,
) -> dict[str, float]:
    """
    Сравнение реальных и смоделированных OCEAN+30.

    Вход:
        real_flat, simulated_flat: словари с ключами из OCEAN_AND_FACET_ORDER.
        keys: по каким ключам считать (если None — все общие).

    Выход:
        mae_35, mae_per_dim, similarity_35, pearson_35, kappa_35,
        mean_similarity_facets, mean_similarity_traits,
        similarity_per_dim (dict по keys).
    """
    if keys is None:
        keys = [k for k in OCEAN_AND_FACET_ORDER if k in real_flat and k in simulated_flat]
    if not keys:
        return {
            "mae_35": 0.0,
            "mae_per_dim": {},
            "similarity_35": 0.0,
            "pearson_35": 0.0,
            "kappa_35": 0.0,
            "mean_similarity_facets": 0.0,
            "mean_similarity_traits": 0.0,
            "similarity_per_dim": {},
        }
    r_vec = np.array([real_flat[k] for k in keys])
    s_vec = np.array([simulated_flat[k] for k in keys])
    # MAE по каждому признаку и среднее
    mae_per_dim = {k: float(abs(real_flat[k] - simulated_flat[k])) for k in keys}
    mae_35 = float(np.mean(list(mae_per_dim.values())))
    # similarity по каждому измерению: 1 - |r-s|/100, затем среднее
    sim_per_dim = {k: 1.0 - abs(real_flat[k] - simulated_flat[k]) / 100.0 for k in keys}
    similarity_35 = float(np.mean(list(sim_per_dim.values())))
    # Pearson по 35 (или по keys)
    try:
        p = sps.pearsonr(r_vec, s_vec)[0]
        pearson_35 = 0.0 if (p != p or np.isnan(p)) else float(p)  # noqa: E711
    except Exception:
        pearson_35 = 0.0
    # Cohen's kappa: категории low / average / high
    real_cat = [_value_to_category(real_flat[k]) for k in keys]
    sim_cat = [_value_to_category(simulated_flat[k]) for k in keys]
    try:
        k = cohen_kappa_score(real_cat, sim_cat)
        kappa_35 = 0.0 if (k != k or np.isnan(k)) else float(k)  # noqa: E711
    except Exception:
        kappa_35 = 0.0
    # средняя similarity по 30 фасетам и по 5 чертам
    k_facets = [k for k in keys if k in FACET_NAMES]
    k_traits = [k for k in keys if k in TRAIT_NAMES]
    mean_similarity_facets = float(np.mean([sim_per_dim[k] for k in k_facets])) if k_facets else 0.0
    mean_similarity_traits = float(np.mean([sim_per_dim[k] for k in k_traits])) if k_traits else 0.0
    return {
        "mae_35": mae_35,
        "mae_per_dim": mae_per_dim,
        "similarity_35": similarity_35,
        "pearson_35": pearson_35,
        "kappa_35": kappa_35,
        "mean_similarity_facets": mean_similarity_facets,
        "mean_similarity_traits": mean_similarity_traits,
        "similarity_per_dim": sim_per_dim,
    }


def _empty_answer_block_similarity() -> dict[str, float | None]:
    return {k: None for k in TRAIT_QUESTION_BLOCKS}


def _build_unparsable_fitness(parse_status: str = "unparsable", error_message: str | None = None) -> dict:
    payload = {
        "similarity": None,
        "avg_diff": None,
        "pearson_corr": None,
        "model_answers": None,
        "simulated_ocean": None,
        "mae_35": None,
        "mae_per_dim": None,
        "similarity_35": None,
        "pearson_35": None,
        "kappa_35": None,
        "mean_similarity_facets": None,
        "mean_similarity_traits": None,
        "similarity_per_dim": None,
        "answer_block_similarity": _empty_answer_block_similarity(),
        "is_unparsable": True,
        "parse_status": parse_status,
    }
    if error_message:
        payload["error_message"] = error_message
    return payload


def _extract_response_text(response) -> str:
    if response is None:
        return ""

    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Некоторые провайдеры возвращают список content-блоков.
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def compute_answer_block_similarity(
    model_answers: dict[int, int] | None,
    participant,
) -> dict[str, float | None]:
    if not model_answers:
        return _empty_answer_block_similarity()

    out: dict[str, float | None] = {}
    for block_key, q_ids in TRAIT_QUESTION_BLOCKS.items():
        sims: list[float] = []
        for q_id in q_ids:
            model_ans = model_answers.get(q_id)
            if not isinstance(model_ans, int):
                continue
            human_ans = participant.get("i" + str(q_id))
            if human_ans is None or (isinstance(human_ans, float) and np.isnan(human_ans)):
                continue
            sims.append(1.0 - abs(model_ans - human_ans) / 4.0)
        out[block_key] = float(np.mean(sims)) if sims else None
    return out


def normalize_participant_score(score: dict) -> dict:
    pearson = score.get("pearson_corr")
    if isinstance(pearson, tuple):
        pearson = pearson[0]

    is_unparsable = bool(score.get("is_unparsable", False))
    if "is_unparsable" not in score:
        is_unparsable = score.get("model_answers") is None

    parse_status = score.get("parse_status")
    if not isinstance(parse_status, str):
        parse_status = "unparsable" if is_unparsable else "parsed"

    return {
        "similarity": _to_optional_float(score.get("similarity")),
        "avg_diff": _to_optional_float(score.get("avg_diff")),
        "pearson_corr": _to_optional_float(pearson),
        "mae_35": score.get("mae_35"),
        "mae_per_dim": score.get("mae_per_dim"),
        "similarity_35": score.get("similarity_35"),
        "pearson_35": score.get("pearson_35"),
        "kappa_35": score.get("kappa_35"),
        "mean_similarity_facets": score.get("mean_similarity_facets"),
        "mean_similarity_traits": score.get("mean_similarity_traits"),
        "similarity_per_dim": score.get("similarity_per_dim"),
        "answer_block_similarity": score.get("answer_block_similarity"),
        "is_unparsable": is_unparsable,
        "parse_status": parse_status,
    }


def aggregate_cluster_five_factor_metrics(participants_scores: list[dict]) -> dict[str, float]:
    """
    Усреднение five-factor метрик по тестовой выборке кластера.
    Учитываются только записи, где соответствующие поля не None.
    """
    agg: dict[str, list[float]] = {
        "mae_35": [],
        "similarity_35": [],
        "pearson_35": [],
        "kappa_35": [],
        "mean_similarity_facets": [],
        "mean_similarity_traits": [],
    }
    for s in participants_scores:
        for k in agg:
            v = s.get(k)
            if _is_valid_number(v):
                agg[k].append(float(v))

    out = {f"mean_{k}": _safe_mean(agg[k]) for k in ("mae_35", "similarity_35", "pearson_35", "kappa_35")}
    out["mean_similarity_facets"] = _safe_mean(agg["mean_similarity_facets"])
    out["mean_similarity_traits"] = _safe_mean(agg["mean_similarity_traits"])

    # MAE по каждому из 35 признаков: усреднение по участникам
    mae_per_dim_collect: dict[str, list[float]] = {}
    for s in participants_scores:
        mpd = s.get("mae_per_dim")
        if isinstance(mpd, dict):
            for dim, val in mpd.items():
                if _is_valid_number(val):
                    mae_per_dim_collect.setdefault(dim, []).append(float(val))
    out["mean_mae_per_dim"] = {dim: _safe_mean(vals) for dim, vals in mae_per_dim_collect.items()}

    return out


def aggregate_stage_metrics(
    participants_scores: list[dict],
    selected_facets: list[str] | None = None,
) -> dict:
    """
    Сводные метрики по участникам.

    Для similarity-метрик значения считаются только по валидным числам.
    Невалидные/unparsable ответы исключаются из средних, а в summary
    добавляются счётчики и доли исключений.
    """
    selected_facets = list(selected_facets or [])
    participants_total = len(participants_scores)
    unparsable_count = sum(1 for s in participants_scores if bool(s.get("is_unparsable")))
    parsed_count = participants_total - unparsable_count

    def _metric_exclusion(metric_key: str, summary_key: str) -> dict[str, int | float]:
        valid_count = sum(1 for s in participants_scores if _is_valid_number(s.get(metric_key)))
        excluded_count = participants_total - valid_count
        excluded_share = (excluded_count / participants_total) if participants_total else 0.0
        return {
            f"{summary_key}_valid_count": valid_count,
            f"{summary_key}_excluded_count": excluded_count,
            f"{summary_key}_excluded_share": excluded_share,
        }

    summary = {
        "mean_similarity": _safe_mean([float(s.get("similarity")) for s in participants_scores if _is_valid_number(s.get("similarity"))]),
        "mean_avg_diff": _safe_mean([float(s.get("avg_diff")) for s in participants_scores if _is_valid_number(s.get("avg_diff"))]),
        "mean_pearson_corr": _safe_mean([float(s.get("pearson_corr")) for s in participants_scores if _is_valid_number(s.get("pearson_corr"))]),
        "participants_total": participants_total,
        "participants_parsed_count": parsed_count,
        "participants_unparsable_count": unparsable_count,
        "participants_unparsable_share": (unparsable_count / participants_total) if participants_total else 0.0,
    }
    summary.update(_metric_exclusion("similarity", "similarity"))
    summary.update(_metric_exclusion("similarity_35", "similarity_35"))
    summary.update(_metric_exclusion("mean_similarity_facets", "similarity_facets"))
    summary.update(_metric_exclusion("mean_similarity_traits", "similarity_traits"))

    ff = aggregate_cluster_five_factor_metrics(participants_scores)
    summary.update(
        {
            "mean_mae_35": ff.get("mean_mae_35", 0.0),
            "mean_similarity_35": ff.get("mean_similarity_35", 0.0),
            "mean_pearson_35": ff.get("mean_pearson_35", 0.0),
            "mean_similarity_facets": ff.get("mean_similarity_facets", 0.0),
            "mean_similarity_traits": ff.get("mean_similarity_traits", 0.0),
        }
    )

    trait_similarity: dict[str, float] = {}
    for trait in TRAIT_NAMES:
        vals = []
        for s in participants_scores:
            per_dim = s.get("similarity_per_dim")
            if isinstance(per_dim, dict) and _is_valid_number(per_dim.get(trait)):
                vals.append(float(per_dim[trait]))
        trait_similarity[trait] = _safe_mean(vals)

    facet_similarity: dict[str, float] = {}
    for facet in selected_facets:
        vals = []
        for s in participants_scores:
            per_dim = s.get("similarity_per_dim")
            if isinstance(per_dim, dict) and _is_valid_number(per_dim.get(facet)):
                vals.append(float(per_dim[facet]))
        facet_similarity[facet] = _safe_mean(vals)

    answer_block_similarity: dict[str, float] = {}
    for block_key in TRAIT_QUESTION_BLOCKS:
        vals = []
        for s in participants_scores:
            blocks = s.get("answer_block_similarity")
            if isinstance(blocks, dict) and _is_valid_number(blocks.get(block_key)):
                vals.append(float(blocks[block_key]))
        answer_block_similarity[block_key] = _safe_mean(vals)

    return {
        "summary": summary,
        "trait_similarity": trait_similarity,
        "facet_similarity": facet_similarity,
        "answer_block_similarity": answer_block_similarity,
        "selected_facets": selected_facets,
        "trait_question_blocks": get_trait_question_blocks(),
    }


def evaluate_participants_batch(participants_df, genotype, task, model, batch_size=1):
    """
    Оценивает участников через fitness_function: по одному (batch_size<=1) или
    пачками с параллельными запросами к модели (batch_size>1).

    Вход:
        participants_df: DataFrame с участниками (как от .iterrows()).
        genotype, task, model: как для fitness_function.
        batch_size: 1 или None — последовательно; 5, 10, 20 — макс. число
                    одновременных запросов к модели (ThreadPoolExecutor).

    Выход:
        Список словарей score (как от fitness_function) в том же порядке, что
        participants_df.iterrows(). Средние считаются по ним так же, как раньше.
    """
    bs = int(batch_size or 0)
    if bs <= 1:
        return [
            fitness_function(participant, genotype, task, model)
            for _, participant in participants_df.iterrows()
        ]

    items = [(i, p) for i, p in participants_df.iterrows()]
    n = len(items)

    def _run_one(idx_p):
        _idx, p = idx_p
        return fitness_function(p, genotype, task, model)

    t0 = time.perf_counter()
    results = [None] * n
    done = 0
    progress_step = max(1, bs // 3)
    with ThreadPoolExecutor(max_workers=bs) as ex:
        futures = {ex.submit(_run_one, item): pos for pos, item in enumerate(items)}
        for fut in as_completed(futures):
            pos = futures[fut]
            try:
                results[pos] = fut.result()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                print(f"[warn] Ошибка при обработке участника: {err}")
                results[pos] = _build_unparsable_fitness(parse_status="request_error", error_message=err)
            done += 1
            if done % progress_step == 0 or done == n:
                elapsed = time.perf_counter() - t0
                print(f"  [batch] completed {done}/{n}, max_concurrent={bs}, elapsed={elapsed:.1f}s")

    wall_s = time.perf_counter() - t0
    # Запросы уходят параллельно: до bs потоков вызывают model.generate одновременно
    print(f"  [batch] {n} participants, max_concurrent={bs}, wall_time={wall_s:.1f}s")
    return results


def fitness_function(participant, genotype, task, model):
    """
    Вычисляет соответствие модели реальному участнику по метрикам схожести ответов.

    Вход:
        participant (pd.Series): данные участника с ответами на вопросы IPIP-NEO
        genotype (dict): конфигурация персонажа для генерации промпта
        task (dict): описание задачи с вопросами и форматом ответа
        model: объект модели для генерации ответов
    Выход:
        dict с метриками. Если ответ модели не удалось распарсить,
        similarity-поля возвращаются как None и помечаются:
        is_unparsable=True, parse_status="unparsable".
    """
    prompt = build_full_prompt(genotype, task, participant)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt["system"]),
        ("human", prompt["human"]),
    ])
    try:
        response = model.generate(prompt_template)
        response_text = _extract_response_text(response)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        print(f"[warn] Ошибка запроса к модели: {err}")
        return _build_unparsable_fitness(parse_status="request_error", error_message=err)

    try:
        model_answers = parse_response(response_text)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        print(f"[warn] Ошибка парсинга ответа модели: {err}")
        return _build_unparsable_fitness(parse_status="parse_error", error_message=err)

    if model_answers is None:
        return _build_unparsable_fitness(parse_status="unparsable")

    fitness = {
        "similarity": 0.0,
        "avg_diff": 0.0,
        "pearson_corr": 0.0,
        "model_answers": model_answers,
        "simulated_ocean": None,
        "mae_35": None,
        "mae_per_dim": None,
        "similarity_35": None,
        "pearson_35": None,
        "kappa_35": None,
        "mean_similarity_facets": None,
        "mean_similarity_traits": None,
        "similarity_per_dim": None,
        "answer_block_similarity": compute_answer_block_similarity(model_answers, participant),
        "is_unparsable": False,
        "parse_status": "parsed",
    }

    list_model_ans = []
    list_human_ans = []
    valid_count = 0
    for q_id, model_ans in model_answers.items():
        human_ans = participant.get("i" + str(q_id))
        if human_ans is None or (isinstance(human_ans, float) and np.isnan(human_ans)):
            continue
        list_model_ans.append(model_ans)
        list_human_ans.append(human_ans)
        fitness["similarity"] += 1.0 - abs(model_ans - human_ans) / 4.0
        fitness["avg_diff"] += abs(model_ans - human_ans)
        valid_count += 1

    if valid_count > 0:
        fitness["similarity"] /= valid_count
        fitness["avg_diff"] /= valid_count

    if len(list_model_ans) >= 2 and len(list_human_ans) >= 2:
        try:
            fitness["pearson_corr"] = float(sps.pearsonr(list_model_ans, list_human_ans)[0])
        except Exception:
            fitness["pearson_corr"] = None

    simulated_ocean = five_factor.compute_ocean_facets(
        model_answers,
        participant.get("sex"),
        participant.get("age", 30),
        question=120,
    )
    fitness["simulated_ocean"] = simulated_ocean

    if simulated_ocean is not None and len(model_answers) >= 120:
        real_flat = {}
        for k in OCEAN_AND_FACET_ORDER:
            if k not in participant.index:
                continue
            v = participant[k]
            if v is None or (isinstance(v, float) and (np.isnan(v) or v != v)):
                continue
            try:
                real_flat[k] = float(v)
            except (TypeError, ValueError):
                pass

        common = [k for k in OCEAN_AND_FACET_ORDER if k in real_flat and k in simulated_ocean]
        if len(common) >= 30:
            m = compute_five_factor_metrics(real_flat, simulated_ocean, keys=common)
            fitness["mae_35"] = m["mae_35"]
            fitness["mae_per_dim"] = m["mae_per_dim"]
            fitness["similarity_35"] = m["similarity_35"]
            fitness["pearson_35"] = m["pearson_35"]
            fitness["kappa_35"] = m["kappa_35"]
            fitness["mean_similarity_facets"] = m["mean_similarity_facets"]
            fitness["mean_similarity_traits"] = m["mean_similarity_traits"]
            fitness["similarity_per_dim"] = m["similarity_per_dim"]

    return fitness
