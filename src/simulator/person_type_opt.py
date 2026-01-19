import pandas as pd
import json
import time
from pathlib import Path
import random

from src.models.registry import get_model

from src.prompt.traits import traits
from src.prompt.facets import facets
from src.prompt.system import system

from src.utils.time import format_time
from src.utils.save_result import save_log
from src.utils.personality_match import fitness_function

from src.evolution.evoluter import GAEvoluter
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.utils import genotype_to_evoprompt_str, parse_str_to_genotype, clean_evoprompt_response, validate_and_repair_genotype
from src.evolution.init_population import init_population
from src.evolution.parse_args import parse_args_from_yaml



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
            'temperature': config['evolution'].get('temperature', 0.7)
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


    for cluster in config['data']['clusters']:
        # Запуск таймера для кластера
        cluster_start_time = time.time()
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

        # === ЭВОЛЮЦИОННАЯ ОПТИМИЗАЦИЯ (если включена в config) ===
        if 'evolution' in config and config['evolution'].get('algorithm'):
            print(f"🧬 Запуск эволюционной оптимизации для кластера {cluster}")

        evo_args = parse_args_from_yaml(config['evolution'])
        evaluator = MyEvaluator(evo_args, task, model, fixed_modifiers, config=config)
        evaluator.dev_participants = train_participants  # Train participants как dev-set

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

        # === ФИНАЛЬНАЯ ОЦЕНКА ОПТИМИЗИРОВАННОГО ГЕНОТИПА ===
        test_participants_scores = []
        
        for idx, (index, participant) in enumerate(test_participants.iterrows(), 1):
            score = fitness_function(participant, genotype, task, model)
            # Сохраняем все три метрики для каждого участника
            participant_score = {
                'similarity': score['similarity'],
                'avg_diff': score['avg_diff'],
                'pearson_corr': score['pearson_corr'][0] if isinstance(score['pearson_corr'], tuple) else score['pearson_corr']
            }
            test_participants_scores.append(participant_score)

        mean_similarity_participants = sum([i['similarity'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_avg_diff_participants = sum([i['avg_diff'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_pearson_corr_participants = sum([i['pearson_corr'] for i in test_participants_scores]) / len(test_participants_scores)

        # Финальное время кластера
        cluster_total_time = time.time() - cluster_start_time
        
        print(f"\n{'='*70}")
        print(f"📈 ИТОГОВЫЕ СРЕДНИЕ ПОКАЗАТЕЛИ КЛАСТЕРА {cluster}")
        print(f"{'='*70}")
        print(f"- Средняя схожесть (similarity): {mean_similarity_participants:.4f}")
        print(f"- Средняя разница (avg_diff): {mean_avg_diff_participants:.4f}")
        print(f"- Средняя корреляция Пирсона (pearson_corr): {mean_pearson_corr_participants:.4f}")
        print(f"{'='*70}")
        print(f"⏱️  Статистика времени кластера:")
        print(f"- Общее время: {format_time(cluster_total_time)}")
        print(f"- Всего обработано участников: {n_participants}")
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
            'total_participants': total_participants
        }
        # Сохраняем детальные скоры по участникам для финальной оценки
        cluster_log['final_participants_scores'] = test_participants_scores
        cluster_log.pop('participants')
        result_cluster['clusters'][str(cluster)]= cluster_log
        save_log(result_cluster, results_dir, "result_log.json")
    

    return experiment_log

