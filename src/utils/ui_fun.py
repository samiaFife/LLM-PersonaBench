import time
import numpy as np
import scipy.stats as sps
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import cohen_kappa_score

from src.utils.prompt import build_full_prompt
from src.utils.parse import parse_response
from src.utils import five_factor

# Имена 35 измерений и подмножества
OCEAN_AND_FACET_ORDER = five_factor.OCEAN_AND_FACET_ORDER
TRAIT_NAMES = five_factor.TRAIT_NAMES
FACET_NAMES = five_factor.FACET_NAMES


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
            if v is not None and isinstance(v, (int, float)) and not (v != v or np.isnan(v)):  # noqa: E711
                agg[k].append(float(v))

    def _safe_mean(arr: list[float]) -> float:
        if not arr:
            return 0.0
        x = float(np.nanmean(arr))
        return 0.0 if (x != x or np.isnan(x)) else x  # noqa: E711

    out = {f"mean_{k}": _safe_mean(agg[k]) for k in ("mae_35", "similarity_35", "pearson_35", "kappa_35")}
    out["mean_similarity_facets"] = _safe_mean(agg["mean_similarity_facets"])
    out["mean_similarity_traits"] = _safe_mean(agg["mean_similarity_traits"])

    # MAE по каждому из 35 признаков: усреднение по участникам
    mae_per_dim_collect: dict[str, list[float]] = {}
    for s in participants_scores:
        mpd = s.get("mae_per_dim")
        if isinstance(mpd, dict):
            for dim, val in mpd.items():
                if isinstance(val, (int, float)) and not (val != val or np.isnan(val)):  # noqa: E711
                    mae_per_dim_collect.setdefault(dim, []).append(float(val))
    out["mean_mae_per_dim"] = {dim: _safe_mean(vals) for dim, vals in mae_per_dim_collect.items()}

    return out


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
    with ThreadPoolExecutor(max_workers=bs) as ex:
        results = list(ex.map(_run_one, items))
    wall_s = time.perf_counter() - t0
    # Запросы уходят параллельно: до bs потоков вызывают model.generate одновременно
    print(f"  [batch] {n} participants, max_concurrent={bs}, wall_time={wall_s:.1f}s")
    return results


def fitness_function_ans(participant, genotype, task, questions_id, model):
    """
    Вычисляет соответствие модели реальному участнику по метрикам схожести ответов.
    
    Вход:
        participant (pd.Series): данные участника с ответами на вопросы IPIP-NEO
        genotype (dict): конфигурация персонажа для генерации промпта
        task (dict): описание задачи с вопросами и форматом ответа
        model: объект модели для генерации ответов
    Выход:
        dict: словарь с метриками {'similarity': float, 'avg_diff': float, 'pearson_corr': float}
              similarity - схожесть ответов (0-1),
              avg_diff - средняя абсолютная разница ответов,
              pearson_corr - корреляция Пирсона между ответами
    """
    prompt = build_full_prompt(genotype, task, participant)
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt["system"]),
            ("human", prompt["human"])
        ])
    response = model.generate(prompt_template)
    model_answers = parse_response(response.content)
    
    if model_answers is None:
        return {
            'similarity': 0.0, 'avg_diff': 0.0, 'pearson_corr': 0.0,
            'model_answers': None,
        }

    fitness = {}
    fitness['similarity'] = 0.0
    fitness['avg_diff'] = 0.0
    fitness['pearson_corr'] = 0.0
    lsit_model_ans = []
    lsit_human_ans = []
    valid_count = 0  # Счетчик валидных ответов (где human_ans is не None/NaN)
    for q_id, model_ans in model_answers.items():
        if q_id not in questions_id:
            continue
        human_ans = participant['i' + str(q_id)]
        # пропускаем отсутствующие или NaN ответы человека
        if human_ans is None or (isinstance(human_ans, float) and np.isnan(human_ans)):
            continue
        lsit_model_ans.append(model_ans)
        lsit_human_ans.append(human_ans)
        fitness['similarity'] += 1 - abs(model_ans - human_ans) / 4
        fitness['avg_diff'] += abs(model_ans - human_ans)
        valid_count += 1
    # Делим только на количество валидных ответов
    if valid_count > 0:
        fitness['similarity'] /= valid_count
        fitness['avg_diff'] /= valid_count
    else:
        # Если нет валидных ответов, возвращаем 0
        fitness['similarity'] = 0.0
        fitness['avg_diff'] = 0.0
    if len(lsit_model_ans) >= 2 and len(lsit_human_ans) >= 2:
        fitness['pearson_corr'] = sps.pearsonr(lsit_model_ans, lsit_human_ans)
    else:
        fitness['pearson_corr'] = 0.0

    # Five-factor: OCEAN+30 из ответов модели и сравнение с реальным участником
    fitness['model_answers'] = model_answers
    return fitness