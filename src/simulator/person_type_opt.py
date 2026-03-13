import json
import time
from pathlib import Path

import pandas as pd

from src.evolution.evoluter import GAEvoluter
from src.evolution.init_population import init_population
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.parse_args import parse_args_from_yaml
from src.evolution.utils import (
    clean_evoprompt_response,
    parse_str_to_genotype,
    validate_and_repair_genotype,
)
from src.models.registry import get_model
from src.utils.personality_match import (
    aggregate_stage_metrics,
    evaluate_participants_batch,
    get_trait_question_blocks,
    normalize_participant_score,
)
from src.utils.save_result import save_log
from src.utils.time import TimeEstimator, format_time


def _get_project_root():
    """Определяет корневую директорию проекта относительно расположения файла или текущей рабочей директории."""
    file_path = Path(__file__).resolve()
    candidate = file_path.parent.parent.parent
    if (candidate / "src").exists():
        return candidate

    cwd = Path.cwd()
    current = cwd
    while current != current.parent:
        if (current / "src").exists():
            return current
        current = current.parent
    return cwd


def _load_traits(config):
    """Загружает traits из config['prompt']['traits_path'] или из встроенного src.prompt.traits."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("traits_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            project_root = _get_project_root()
            p = project_root / p
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    from src.prompt.traits import traits

    return traits


def _load_facets(config):
    """Загружает facets из config['prompt']['facets_path'] или из встроенного src.prompt.facets."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("facets_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            project_root = _get_project_root()
            p = project_root / p
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    from src.prompt.facets import facets

    return facets


def _load_system(config):
    """Загружает system из config['prompt']['system_path'] или из встроенного src.prompt.system."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("system_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            project_root = _get_project_root()
            p = project_root / p
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    from src.prompt.system import system

    return system


def _load_trait_target_values(config):
    """Целевые значения черт по кластерам (для модификатора по совпадению)."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("traits_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = _get_project_root() / p
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("trait_target_values", {})
            return {int(k): v for k, v in raw.items()}
    from src.prompt.traits import trait_target_values

    return trait_target_values


def _load_facet_target_values(config):
    """Целевые значения фасетов по кластерам (для модификатора по совпадению)."""
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("facets_path")
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = _get_project_root() / p
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("facet_target_values", {})
            return {int(k): v for k, v in raw.items()}
    from src.prompt.facets import facet_target_values

    return facet_target_values


def _evaluate_participants_on_test(
    test_participants,
    genotype,
    task,
    model,
    participant_batch_size,
    results_dir,
    cluster,
    selected_facets,
    csv_filename,
):
    scores = evaluate_participants_batch(
        test_participants,
        genotype,
        task,
        model,
        participant_batch_size,
    )

    participant_scores = []
    rows_answers = []

    for (index, participant), score in zip(list(test_participants.iterrows()), scores):
        participant_scores.append(normalize_participant_score(score))

        model_answers = score.get("model_answers") or {}
        row = {"case": participant.get("case", index)}
        for i in range(1, 121):
            row[f"i{i}"] = model_answers.get(i)
        rows_answers.append(row)

    cluster_dir = results_dir / f"cluster_{cluster}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if rows_answers:
        answers_df = pd.DataFrame(rows_answers)
    else:
        answers_df = pd.DataFrame(columns=["case", *[f"i{i}" for i in range(1, 121)]])

    csv_path = cluster_dir / csv_filename
    answers_df.to_csv(csv_path, index=False)

    stage_metrics = aggregate_stage_metrics(participant_scores, selected_facets)
    return {
        "participant_scores": participant_scores,
        "stage_metrics": stage_metrics,
        "answers_csv": str(Path(f"cluster_{cluster}") / csv_filename).replace("\\", "/"),
    }


def _build_stage_payload(stage_metrics: dict, prompt, answers_csv: str | None = None) -> dict:
    payload = {
        "summary": stage_metrics.get("summary", {}),
        "trait_similarity": stage_metrics.get("trait_similarity", {}),
        "facet_similarity": stage_metrics.get("facet_similarity", {}),
        "answer_block_similarity": stage_metrics.get("answer_block_similarity", {}),
        "selected_facets": stage_metrics.get("selected_facets", []),
        "trait_question_blocks": stage_metrics.get("trait_question_blocks", get_trait_question_blocks()),
        "prompt": prompt,
    }
    if answers_csv is not None:
        payload["answers_csv"] = answers_csv
    return payload


# ГЛАВНЫЙ ЦИКЛ ЭКСПЕРИМЕНТА
def run_experiment(config):
    """
    config: словарь с конфигурацией эксперимента, включающий:
        - data: настройки данных (file_path, cluster, num_participants)
        - model: настройки модели (name, provider, temperature)
        - evolution: настройки эволюционного алгоритма
        - experiment: настройки эксперимента (seed, save_every_generation)
        - results_dir: путь к директории для сохранения результатов
        - experiment_id: уникальный идентификатор эксперимента
    """
    results_dir = Path(config["results_dir"])
    experiment_id = config["experiment_id"]

    experiment_log = {
        "experiment_id": experiment_id,
        "status": "started",
        "config": config,
        "clusters": {},
    }
    save_log(experiment_log, results_dir, "experiment_log.json")

    result_log = {
        "experiment_id": experiment_id,
        "clusters": {},
    }
    save_log(result_log, results_dir, "result_log.json")

    traits = _load_traits(config)
    facets = _load_facets(config)
    system = _load_system(config)
    trait_target_values = _load_trait_target_values(config)
    facet_target_values = _load_facet_target_values(config)

    fixed_modifiers = system["intensity_modifiers"]

    print("📦 Загрузка модели...")
    model = get_model(config["model"])
    print(f"✅ Модель для симуляции загружена: {config['model'].get('model_name', 'неизвестно')}\n")

    evolution_model = None
    if "evolution" in config and config["evolution"].get("llm_for_evolution"):
        print("📦 Загрузка модели для эволюции...")
        evolution_model_config = {
            "model_name": config["evolution"]["llm_for_evolution"],
            "provider": config["evolution"].get("provider", "cloud"),
            "temperature": config["evolution"].get("temperature", 0.7),
            "timeout": config["evolution"].get("timeout")
            or config["model"].get("timeout")
            or config["model"].get("request_timeout"),
            "max_retries": config["evolution"].get("max_retries") or config["model"].get("max_retries"),
        }
        evolution_model = get_model(evolution_model_config)
        print(f"✅ Модель для эволюции загружена: {evolution_model_config['model_name']}\n")

    print("📂 Загрузка данных участников...")
    data_participants = pd.read_csv(config["data"]["file_path"])
    print(f"✅ Загружено участников: {len(data_participants)}\n")

    print("📋 Загрузка вопросов IPIP-NEO...")
    with open("data/IPIP-NEO/120/questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    ipip_neo_questions = data.get("questions")
    print(f"✅ Загружено вопросов: {len(ipip_neo_questions)}\n")

    task = {
        "task": system["task"],
        "ipip_neo": ipip_neo_questions,
        "response_format": system["response_format"],
    }

    clusters_list = config["data"]["clusters"]
    experiment_time_estimator = TimeEstimator(total_items=len(clusters_list))
    experiment_time_estimator.start()

    for cluster_idx, cluster in enumerate(clusters_list):
        cluster_start_time = time.time()
        experiment_time_estimator.start_item()

        print(f"\n{'#' * 70}")
        print(f"📊 ОБРАБОТКА КЛАСТЕРА: {cluster}")
        print(f"{'#' * 70}\n")

        cluster_trait_targets = trait_target_values.get(cluster, {})
        cluster_facet_targets = facet_target_values.get(cluster, {})
        trait_formulations = traits[cluster]
        facet_formulations = facets[cluster]

        base_genotype = {
            "role_definition": system["role"],
            "trait_formulations": trait_formulations,
            "facet_formulations": facet_formulations,
            "intensity_modifiers": system["intensity_modifiers"],
            "critic_formulations": system["critic_internal"],
            "template_structure": system["template_structure"],
            "trait_targets": {k: cluster_trait_targets[k] for k in trait_formulations if k in cluster_trait_targets},
            "facet_targets": {k: cluster_facet_targets[k] for k in facet_formulations if k in cluster_facet_targets},
        }
        genotype = base_genotype.copy()
        selected_facets = list(facet_formulations.keys())

        n_participants = config["data"]["num_participants"]
        total_participants = data_participants[data_participants["clusters"] == cluster].iloc[:n_participants]

        train_size = int(n_participants * 0.6)
        test_size = n_participants - train_size
        train_participants = total_participants.iloc[:train_size]
        test_participants = total_participants.iloc[train_size:]
        print(f"👥 Отобрано участников для кластера {cluster}: {len(total_participants)}")
        print(f"👥 Train: {train_size},  Test: {test_size}")

        participant_batch_size = int(
            (config.get("simulation") or {}).get("participant_batch_size")
            or (config.get("evolution") or {}).get("participant_batch_size", 1)
            or 1
        )

        print(f"\n📊 ПРОГОН БЕЗ ОПТИМИЗАЦИИ на Test (batch_size={participant_batch_size})")
        non_opt_results = _evaluate_participants_on_test(
            test_participants,
            base_genotype,
            task,
            model,
            participant_batch_size,
            results_dir,
            cluster,
            selected_facets,
            csv_filename="before_optimization_test_answers.csv",
        )
        before_stage = _build_stage_payload(
            stage_metrics=non_opt_results["stage_metrics"],
            prompt=base_genotype,
            answers_csv=non_opt_results["answers_csv"],
        )
        print("✅ Прогон без оптимизации завершён")
        print(f"  - Средняя схожесть: {before_stage['summary'].get('mean_similarity', 0):.4f}")
        print(f"  - Средняя разница: {before_stage['summary'].get('mean_avg_diff', 0):.4f}")
        print(f"  - Средняя корреляция Пирсона: {before_stage['summary'].get('mean_pearson_corr', 0):.4f}\n")

        optimization_enabled = "evolution" in config and config["evolution"].get("algorithm")
        optimization_generations_stage = {"generations": []}
        if optimization_enabled:
            cluster_progress = experiment_time_estimator.get_progress_info(completed_items=cluster_idx)
            print(f"🧬 Запуск эволюционной оптимизации для кластера {cluster} | {cluster_progress}")

            evo_args = parse_args_from_yaml(config["evolution"])
            evaluator = MyEvaluator(
                evo_args,
                task,
                model,
                fixed_modifiers,
                template_genotype=base_genotype,
                config=config,
            )
            evaluator.dev_participants = train_participants
            evaluator.logger.info(
                f"MyEvaluator: установлено {len(train_participants)} участников для оценки (dev_participants)"
            )

            model_for_evolution = evolution_model if evolution_model is not None else model
            evoluter = GAEvoluter(evo_args, evaluator, evolution_model=model_for_evolution, config=config)
            evoluter.population = init_population(base_genotype, config, evo_args.popsize, model_for_evolution)
            evoluter.evolute()

            best_str_raw = evoluter.population[0]
            best_str = clean_evoprompt_response(best_str_raw)
            best_str = validate_and_repair_genotype(best_str, fixed_modifiers, base_genotype, config)
            genotype = parse_str_to_genotype(best_str, fixed_modifiers, config, template_genotype=base_genotype)
            print("✅ Эволюция завершена. Лучший генотип сохранён.")

            for gen_data in getattr(evoluter, "generation_logs", []):
                stage = gen_data.get("best_stage_summary") or {}
                optimization_generations_stage["generations"].append(
                    {
                        "generation": gen_data.get("generation"),
                        "best_score": gen_data.get("best_score", 0.0),
                        "mean_score": gen_data.get("mean_score", 0.0),
                        "best_prompt": gen_data.get("best_prompt"),
                        "summary": stage.get("summary", {}),
                        "trait_similarity": stage.get("trait_similarity", {}),
                        "facet_similarity": stage.get("facet_similarity", {}),
                        "answer_block_similarity": stage.get("answer_block_similarity", {}),
                        "selected_facets": stage.get("selected_facets", selected_facets),
                        "trait_question_blocks": stage.get("trait_question_blocks", get_trait_question_blocks()),
                    }
                )
        else:
            print("⚠️  Эволюционная оптимизация не включена в конфиге. Используется базовый генотип.")

        after_stage = None
        if optimization_enabled:
            print(f"📊 ОЦЕНКА ОПТИМИЗИРОВАННОГО ГЕНОТИПА на Test (batch_size={participant_batch_size})")
            opt_results = _evaluate_participants_on_test(
                test_participants,
                genotype,
                task,
                model,
                participant_batch_size,
                results_dir,
                cluster,
                selected_facets,
                csv_filename="after_optimization_test_answers.csv",
            )
            after_stage = _build_stage_payload(
                stage_metrics=opt_results["stage_metrics"],
                prompt=genotype,
                answers_csv=opt_results["answers_csv"],
            )
        else:
            print("⏭️  Повторный этап after_optimization_test пропущен (режим без оптимизации).")

        final_stage = after_stage if after_stage is not None else before_stage

        cluster_total_time = time.time() - cluster_start_time
        experiment_time_estimator.finish_item()
        experiment_progress = experiment_time_estimator.get_progress_info(completed_items=cluster_idx + 1)

        print(f"\n{'=' * 70}")
        print(f"📈 ИТОГОВЫЕ СРЕДНИЕ ПОКАЗАТЕЛИ КЛАСТЕРА {cluster}")
        print(f"{'=' * 70}")
        print(f"- Средняя схожесть (similarity): {final_stage['summary'].get('mean_similarity', 0):.4f}")
        print(f"- Средняя разница (avg_diff): {final_stage['summary'].get('mean_avg_diff', 0):.4f}")
        print(f"- Средняя корреляция Пирсона (pearson_corr): {final_stage['summary'].get('mean_pearson_corr', 0):.4f}")
        print("  Five-factor (OCEAN+30):")
        print(f"  - MAE (mean |real−sim|): {final_stage['summary'].get('mean_mae_35', 0):.4f}")
        print(f"  - Similarity по 35: {final_stage['summary'].get('mean_similarity_35', 0):.4f}")
        print(f"  - Similarity по 30 фасетам: {final_stage['summary'].get('mean_similarity_facets', 0):.4f}")
        print(f"  - Similarity по 5 чертам: {final_stage['summary'].get('mean_similarity_traits', 0):.4f}")
        print(f"  - Pearson по 35: {final_stage['summary'].get('mean_pearson_35', 0):.4f}")
        print(f"{'=' * 70}")
        print("⏱️  Статистика времени кластера:")
        print(f"- Общее время: {format_time(cluster_total_time)}")
        print(f"- Всего обработано участников: {n_participants}")
        print(f"- Прогресс эксперимента: {experiment_progress}")
        print(f"{'=' * 70}")
        print(f"✅ Обработка кластера {cluster} завершена\n")

        cluster_log = {
            "cluster_id": cluster,
            "start_time": cluster_start_time,
            "end_time": time.time(),
            "total_time": cluster_total_time,
            "participants_total": len(total_participants),
            "participants_train": len(train_participants),
            "participants_test": len(test_participants),
            "stages": {
                "before_optimization_test": before_stage,
                "after_optimization_test": after_stage,
                "optimization_generations": optimization_generations_stage,
            },
        }

        result_log["clusters"][str(cluster)] = cluster_log
        save_log(result_log, results_dir, "result_log.json")

        experiment_log["clusters"][str(cluster)] = {
            "status": "completed",
            "total_time": cluster_total_time,
        }
        save_log(experiment_log, results_dir, "experiment_log.json")

    experiment_log["status"] = "completed"
    save_log(experiment_log, results_dir, "experiment_log.json")
    return experiment_log
