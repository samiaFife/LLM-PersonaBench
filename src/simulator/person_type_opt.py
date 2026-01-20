import pandas as pd
import json
import time
from pathlib import Path
import random

from src.models.registry import get_model

from src.utils.time import format_time, TimeEstimator
from src.utils.save_result import save_log
from src.utils.personality_match import (
    evaluate_participants_batch,
    fitness_function,
    aggregate_cluster_five_factor_metrics,
    OCEAN_AND_FACET_ORDER,
)

from src.evolution.evoluter import GAEvoluter
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.utils import genotype_to_evoprompt_str, parse_str_to_genotype, clean_evoprompt_response, validate_and_repair_genotype
from src.evolution.init_population import init_population
from src.evolution.parse_args import parse_args_from_yaml


def _load_traits(config):
    """Загружает traits из config['prompt']['traits_path'] или из встроенного src.prompt.traits."""
    prompt_cfg = config.get('prompt') or {}
    path = prompt_cfg.get('traits_path')
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    from src.prompt.traits import traits
    return traits


def _load_facets(config):
    """Загружает facets из config['prompt']['facets_path'] или из встроенного src.prompt.facets."""
    prompt_cfg = config.get('prompt') or {}
    path = prompt_cfg.get('facets_path')
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    from src.prompt.facets import facets
    return facets


def _load_system(config):
    """Загружает system из config['prompt']['system_path'] или из встроенного src.prompt.system."""
    prompt_cfg = config.get('prompt') or {}
    path = prompt_cfg.get('system_path')
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    from src.prompt.system import system
    return system


def _evaluate_participants_on_test(
    test_participants, genotype, task, model, participant_batch_size, 
    results_dir, cluster, cluster_start_time, csv_filename="simulated_ocean_test.csv"
):
    """
    Вспомогательная функция для оценки участников на тестовой выборке.
    Возвращает словарь с результатами оценки.
    
    Args:
        csv_filename: имя файла для сохранения CSV результатов (по умолчанию "simulated_ocean_test.csv")
    """
    test_participants_scores = []
    rows_simulated = []

    scores = evaluate_participants_batch(
        test_participants, genotype, task, model, participant_batch_size
    )
    for (index, participant), score in zip(list(test_participants.iterrows()), scores):
        # Метрики по сырым ответам
        participant_score = {
            'similarity': score['similarity'],
            'avg_diff': score['avg_diff'],
            'pearson_corr': score['pearson_corr'][0] if isinstance(score['pearson_corr'], tuple) else score['pearson_corr'],
        }
        # Five-factor метрики (OCEAN+30), в т.ч. mae_per_dim — MAE по каждому из 35 признаков
        for k in ('mae_35', 'mae_per_dim', 'similarity_35', 'pearson_35', 'kappa_35', 'mean_similarity_facets', 'mean_similarity_traits'):
            participant_score[k] = score.get(k)
        test_participants_scores.append(participant_score)

        # Строка для CSV: case, source, OCEAN+30, mae_<dim> по 35 признакам, i1..i120
        model_answers = score.get('model_answers') or {}
        simulated_ocean = score.get('simulated_ocean') or {}
        mae_per_dim = score.get('mae_per_dim') or {}
        row = {
            'case': participant.get('case', index),
            'source': 'simulated',
        }
        for k in OCEAN_AND_FACET_ORDER:
            row[k] = simulated_ocean.get(k)
        for k in OCEAN_AND_FACET_ORDER:
            row['mae_' + k] = mae_per_dim.get(k)
        for i in range(1, 121):
            row['i' + str(i)] = model_answers.get(i)
        rows_simulated.append(row)

    # Сохранение смоделированных ответов и OCEAN+30 в CSV (для сравнения с реальными по case)
    if rows_simulated:
        (results_dir / f"cluster_{cluster}").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_simulated).to_csv(
            results_dir / f"cluster_{cluster}" / csv_filename,
            index=False,
        )

    # Усреднение five-factor метрик по кластеру
    agg_ff = aggregate_cluster_five_factor_metrics(test_participants_scores)

    mean_similarity_participants = sum([i['similarity'] for i in test_participants_scores]) / len(test_participants_scores)
    mean_avg_diff_participants = sum([i['avg_diff'] for i in test_participants_scores]) / len(test_participants_scores)
    mean_pearson_corr_participants = sum([i['pearson_corr'] for i in test_participants_scores]) / len(test_participants_scores)

    return {
        'test_participants_scores': test_participants_scores,
        'agg_ff': agg_ff,
        'mean_similarity': mean_similarity_participants,
        'mean_avg_diff': mean_avg_diff_participants,
        'mean_pearson_corr': mean_pearson_corr_participants,
    }


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
    results_dir = Path(config['results_dir'])
    experiment_id = config['experiment_id']
    experiment_log = {
            'experiment_id': experiment_id,
            'status': 'started',
            'config': config,
            'clusters': {}
        }
    save_log(experiment_log, results_dir, "experiment_log.json")
    result_cluster = {
            'experiment_id': experiment_id,
            'clusters': {}
        }
    save_log(result_cluster, results_dir, "result_log.json")
    
    # Инициализация результатов для оптимизированной и неоптимизированной симуляций
    result_non_opt_log = {
            'experiment_id': experiment_id,
            'clusters': {}
        }
    result_opt_log = {
            'experiment_id': experiment_id,
            'clusters': {}
        }

    # Загрузка traits, facets, system из путей в конфиге или встроенных по умолчанию
    traits = _load_traits(config)
    facets = _load_facets(config)
    system = _load_system(config)

    # Фиксированные модификаторы интенсивности (не оптимизируются пока)
    fixed_modifiers = system['intensity_modifiers']

    print(f"📦 Загрузка модели...")
    model = get_model(config['model'])
    print(f"✅ Модель для симуляции загружена: {config['model'].get('model_name', 'неизвестно')}\n")
    
    # Загрузка модели для эволюции (если указана)
    evolution_model = None
    if 'evolution' in config and config['evolution'].get('llm_for_evolution'):
        print(f"📦 Загрузка модели для эволюции...")
        evolution_model_config = {
            'model_name': config['evolution']['llm_for_evolution'],
            'provider': config['evolution'].get('provider', 'cloud'), 
            'temperature': config['evolution'].get('temperature', 0.7),
            # опционально: чтобы эволюционные LLM-вызовы не "висели" бесконечно
            'timeout': config['evolution'].get('timeout') or config['model'].get('timeout') or config['model'].get('request_timeout'),
            'max_retries': config['evolution'].get('max_retries') or config['model'].get('max_retries'),
        }
        evolution_model = get_model(evolution_model_config)
        print(f"✅ Модель для эволюции загружена: {evolution_model_config['model_name']}\n")
    
    print(f"📂 Загрузка данных участников...")
    data_participants = pd.read_csv(config['data']['file_path'])
    print(f"✅ Загружено участников: {len(data_participants)}\n")
    
    print(f"📋 Загрузка вопросов IPIP-NEO...")
    with open('data/IPIP-NEO/120/questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    ipip_neo_questions = data.get('questions')
    print(f"✅ Загружено вопросов: {len(ipip_neo_questions)}\n")

    task = {
        'task': system['task'],
        'ipip_neo': ipip_neo_questions,
        'response_format': system['response_format'],
    }

    # Инициализация планировщика времени для всего эксперимента
    clusters_list = config['data']['clusters']
    experiment_time_estimator = TimeEstimator(total_items=len(clusters_list))
    experiment_time_estimator.start()

    for cluster_idx, cluster in enumerate(clusters_list):
        # Запуск таймера для кластера
        cluster_start_time = time.time()
        experiment_time_estimator.start_item()
        cluster_log = {
            'cluster_id': cluster,
            'start_time': cluster_start_time,
            'participants': []
        }
        
        print(f"\n{'#'*70}")
        print(f"📊 ОБРАБОТКА КЛАСТЕРА: {cluster}")
        print(f"{'#'*70}\n")
        base_genotype = {
            'role_definition': system['role'],
            'trait_formulations': traits[cluster],
            'facet_formulations': facets[cluster],
            'intensity_modifiers': system['intensity_modifiers'],
            'critic_formulations': system['critic_internal'],
            'template_structure': system['template_structure'],
        }
        genotype = base_genotype.copy()

        # Фильтрация участников кластера
        n_participants = config['data']['num_participants']
        total_participants = data_participants[data_participants['clusters'] == cluster].iloc[:n_participants]
        
        train_size = int(n_participants * 0.6)  # 60%
        test_size = n_participants - train_size  # 40% (остаток)
        train_participants =  total_participants.iloc[:train_size]
        test_participants = total_participants.iloc[train_size:]
        print(f"👥 Отобрано участников для кластера {cluster}: {len(total_participants)}")
        print(f"👥 Train: {train_size},  Test: {test_size}")

        # === ПРОГОН БЕЗ ОПТИМИЗАЦИИ НА ТЕСТОВОЙ ВЫБОРКЕ ===
        participant_batch_size = int(
            (config.get('simulation') or {}).get('participant_batch_size')
            or (config.get('evolution') or {}).get('participant_batch_size', 1)
            or 1
        )
        print(f"\n📊 ПРОГОН БЕЗ ОПТИМИЗАЦИИ на Test (batch_size={participant_batch_size})")
        non_opt_results = _evaluate_participants_on_test(
            test_participants, base_genotype, task, model, participant_batch_size,
            results_dir, cluster, cluster_start_time, csv_filename="simulated_ocean_test_non_opt.csv"
        )
        
        # Сохранение результатов без оптимизации
        non_opt_cluster_log = {
            'cluster_id': cluster,
            'start_time': cluster_start_time,
            'end_time': time.time(),
            'genotype': base_genotype,
            'task': task,
            'summary': {
                'mean_similarity': non_opt_results['mean_similarity'],
                'mean_avg_diff': non_opt_results['mean_avg_diff'],
                'mean_pearson_corr': non_opt_results['mean_pearson_corr'],
                'total_participants': len(total_participants),
                **non_opt_results['agg_ff'],
            },
            'final_participants_scores': non_opt_results['test_participants_scores'],
        }
        result_non_opt_log['clusters'][str(cluster)] = non_opt_cluster_log
        save_log(result_non_opt_log, results_dir, "result_non_opt_log.json")
        
        print(f"✅ Прогон без оптимизации завершён")
        print(f"  - Средняя схожесть: {non_opt_results['mean_similarity']:.4f}")
        print(f"  - Средняя разница: {non_opt_results['mean_avg_diff']:.4f}")
        print(f"  - Средняя корреляция Пирсона: {non_opt_results['mean_pearson_corr']:.4f}\n")

        # === ЭВОЛЮЦИОННАЯ ОПТИМИЗАЦИЯ (если включена в config) ===
        if 'evolution' in config and config['evolution'].get('algorithm'):
            cluster_progress = experiment_time_estimator.get_progress_info(completed_items=cluster_idx)
            print(f"🧬 Запуск эволюционной оптимизации для кластера {cluster} | {cluster_progress}")

            evo_args = parse_args_from_yaml(config['evolution'])
            evaluator = MyEvaluator(evo_args, task, model, fixed_modifiers, config=config)
            evaluator.dev_participants = train_participants  # Train participants как dev-set
            evaluator.logger.info(f"MyEvaluator: установлено {len(train_participants)} участников для оценки (dev_participants)")

            # Используем модель эволюции, если она загружена, иначе основную модель
            model_for_evolution = evolution_model if evolution_model is not None else model
            evoluter = GAEvoluter(evo_args, evaluator, evolution_model=model_for_evolution, config=config)

            # Инициализация популяции строками-генотипами
            evoluter.population = init_population(base_genotype, config, evo_args.popsize, model_for_evolution)

            # Запуск эволюции
            evoluter.evolute()

            # Лучший генотип после эволюции
            best_str_raw = evoluter.population[0]  # Первая — лучшая после сортировки в evolute()
            best_str = clean_evoprompt_response(best_str_raw)
            # Чиним на случай битого JSON
            best_str = validate_and_repair_genotype(best_str, fixed_modifiers, base_genotype, config)
            genotype = parse_str_to_genotype(best_str, fixed_modifiers, config)
            print(f"✅ Эволюция завершена. Лучший генотип сохранён.")
            generations_log = {
                "generations": evoluter.generation_logs if hasattr(evoluter, 'generation_logs') else [],
                "final_population": evoluter.population
            }
            save_log(generations_log, results_dir / f"cluster_{cluster}", f"evolution_generations.json")
        else:
            print(f"⚠️  Эволюционная оптимизация не включена в конфиге. Используется базовый генотип.")

        # === ФИНАЛЬНАЯ ОЦЕНКА ОПТИМИЗИРОВАННОГО ГЕНОТИПА ===
        print(f"📊 ОЦЕНКА ОПТИМИЗИРОВАННОГО ГЕНОТИПА на Test (batch_size={participant_batch_size})")
        opt_results = _evaluate_participants_on_test(
            test_participants, genotype, task, model, participant_batch_size,
            results_dir, cluster, cluster_start_time
        )
        
        test_participants_scores = opt_results['test_participants_scores']
        agg_ff = opt_results['agg_ff']
        mean_similarity_participants = opt_results['mean_similarity']
        mean_avg_diff_participants = opt_results['mean_avg_diff']
        mean_pearson_corr_participants = opt_results['mean_pearson_corr']

        # Финальное время кластера
        cluster_total_time = time.time() - cluster_start_time
        
        # Обновляем планировщик эксперимента
        experiment_time_estimator.finish_item()
        experiment_progress = experiment_time_estimator.get_progress_info(completed_items=cluster_idx + 1)
        
        print(f"\n{'='*70}")
        print(f"📈 ИТОГОВЫЕ СРЕДНИЕ ПОКАЗАТЕЛИ КЛАСТЕРА {cluster}")
        print(f"{'='*70}")
        print(f"- Средняя схожесть (similarity): {mean_similarity_participants:.4f}")
        print(f"- Средняя разница (avg_diff): {mean_avg_diff_participants:.4f}")
        print(f"- Средняя корреляция Пирсона (pearson_corr): {mean_pearson_corr_participants:.4f}")
        print(f"  Five-factor (OCEAN+30):")
        print(f"  - MAE (mean |real−sim|): {agg_ff.get('mean_mae_35', 0):.4f}")
        print(f"  - Similarity по 35: {agg_ff.get('mean_similarity_35', 0):.4f}")
        print(f"  - Similarity по 30 фасетам: {agg_ff.get('mean_similarity_facets', 0):.4f}")
        print(f"  - Similarity по 5 чертам: {agg_ff.get('mean_similarity_traits', 0):.4f}")
        print(f"  - Pearson по 35: {agg_ff.get('mean_pearson_35', 0):.4f}")
        print(f"  - Cohen's kappa (low/avg/high): {agg_ff.get('mean_kappa_35', 0):.4f}")
        print(f"{'='*70}")
        print(f"⏱️  Статистика времени кластера:")
        print(f"- Общее время: {format_time(cluster_total_time)}")
        print(f"- Всего обработано участников: {n_participants}")
        print(f"- Прогресс эксперимента: {experiment_progress}")
        print(f"{'='*70}")
        print(f"✅ Обработка кластера {cluster} завершена\n")
        
        cluster_log['end_time'] = time.time()
        cluster_log['total_time'] = cluster_total_time
        cluster_log['genotype'] = genotype
        cluster_log['task'] = task 
        cluster_log['summary'] = {
            'mean_similarity': mean_similarity_participants,
            'mean_avg_diff': mean_avg_diff_participants,
            'mean_pearson_corr': mean_pearson_corr_participants,
            'total_participants': len(total_participants),
            **agg_ff,
        }
        # Сохраняем детальные скоры по участникам для финальной оценки
        cluster_log['final_participants_scores'] = test_participants_scores
        cluster_log.pop('participants')
        
        # Сохранение результатов оптимизированной симуляции
        result_opt_log['clusters'][str(cluster)] = cluster_log
        save_log(result_opt_log, results_dir, "result_opt_log.json")
        
        # Также сохраняем в старый файл для обратной совместимости
        result_cluster['clusters'][str(cluster)] = cluster_log
        save_log(result_cluster, results_dir, "result_log.json")
    

    return experiment_log

