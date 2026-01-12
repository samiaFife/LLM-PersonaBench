import pandas as pd
import json

import time
from pathlib import Path

from src.models.registry import get_model

from src.prompt.traits import traits
from src.prompt.facets import facets
from src.prompt.system import system

from src.utils.time import format_time
from src.utils.save_result import save_log
from src.utils.personality_match import fitness_function

from src.evolution.evoluter import GAEvoluter
from src.evolution.my_evaluator import MyEvaluator
from src.evolution.utils import genotype_to_str, parse_str_to_genotype



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

    print(f"📦 Загрузка модели...")
    model = get_model(config['model'])
    print(f"✅ Модель загружена: {config['model'].get('model_name', 'неизвестно')}\n")
    
    print(f"📂 Загрузка данных участников...")
    data_participants = pd.read_csv(config['data']['file_path'])
    print(f"✅ Загружено участников: {len(data_participants)}\n")
    
    print(f"📋 Загрузка вопросов IPIP-NEO...")
    with open('data/IPIP-NEO/120/questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    ipip_neo_questions = data.get('questions')
    print(f"✅ Загружено вопросов: {len(ipip_neo_questions)}\n")

    ###
    #evo_args = parse_args_from_yaml(config['evolution'])  # Адаптированный parse
    ###

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
        genotype = {
            'role_definition': system['role'],
            'trait_formulations': traits[cluster],
            'facet_formulations': facets[cluster],
            'intensity_modifiers': system['intensity_modifiers'],
            'critic_formulations': system['critic_internal'],
            'template_structure': system['template_structure'],
        }
        task = {
            'task': system['task'],
            'ipip_neo': ipip_neo_questions,
            'response_format': system['response_format'],
        }
        
        test_participants = data_participants[data_participants['clusters'] == cluster].iloc[:config['data']['num_participants']]
        total_participants = len(test_participants)
        print(f"👥 Отобрано участников для кластера {cluster}: {total_participants}")

        ###
        # evaluator = MyEvaluator(evo_args)
        # evaluator.dev_participants = test_participants
        # важно понять как эволюционированный промт потом запарсить, чтобы потом из вытащить нужный части и собрать промт для моделирования
        ###

        test_participants_scores = []
        iteration_times = []
        
        for idx, (index, participant) in enumerate(test_participants.iterrows(), 1):
            iteration_start_time = time.time()
            # ГЛАВНАЯ часть цикла: обращение к модели
            score = fitness_function(participant, genotype, task, model)
            test_participants_scores.append(score)
            
            iteration_time = time.time() - iteration_start_time
            iteration_times.append(iteration_time)
            
            # Вычисление времени для кластера
            cluster_elapsed = time.time() - cluster_start_time
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            remaining_iterations = total_participants - idx
            eta = avg_iteration_time * remaining_iterations

            participant_result = {
                'participant_id': int(index),
                'similarity': score['similarity'],
                'avg_diff': score['avg_diff'],
                'pearson_corr': score['pearson_corr'],
                'iteration_time': iteration_time
            }
            cluster_log['participants'].append(participant_result)
            # Сохраняем промежуточный лог каждые N участников
            if idx % config['data']['save_every_n'] == 0:
                experiment_log['clusters'][str(cluster)] = cluster_log
                save_log(experiment_log, results_dir, "experiment_log.json")

            #####
            # Возможно тут надо будет добавить усреднение скора по всем участникам к этому моменту прошедшим для отдачи этого результата в EvoPrompt
            #####
            
            print(f"📊 Результаты оценки соответствия:")
            print(f"- Схожесть (similarity): {score['similarity']:.4f}")
            print(f"- Средняя разница (avg_diff): {score['avg_diff']:.4f}")
            print(f"- Корреляция Пирсона (pearson_corr): {score['pearson_corr'][0]:.4f}" if isinstance(score['pearson_corr'], tuple) else f"   • Корреляция Пирсона (pearson_corr): {score['pearson_corr']:.4f}")
            print(f"{'='*70}\n")

            print(f"⏱️  Время: прошедшее {format_time(cluster_elapsed)} | "
                  f"итерация {format_time(iteration_time)} | "
                  f"ETA {format_time(eta)}")

        mean_similarity_participants = sum([i['similarity'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_avg_diff_participants = sum([i['avg_diff'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_pearson_corr_participants = sum([i['pearson_corr'][0] if isinstance(i['pearson_corr'], tuple) else i['pearson_corr'] for i in test_participants_scores]) / len(test_participants_scores)

        # Финальное время кластера
        cluster_total_time = time.time() - cluster_start_time
        avg_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
        
        print(f"\n{'='*70}")
        print(f"📈 ИТОГОВЫЕ СРЕДНИЕ ПОКАЗАТЕЛИ КЛАСТЕРА {cluster}")
        print(f"{'='*70}")
        print(f"- Средняя схожесть (similarity): {mean_similarity_participants:.4f}")
        print(f"- Средняя разница (avg_diff): {mean_avg_diff_participants:.4f}")
        print(f"- Средняя корреляция Пирсона (pearson_corr): {mean_pearson_corr_participants:.4f}")
        print(f"{'='*70}")
        print(f"⏱️  Статистика времени кластера:")
        print(f"- Общее время: {format_time(cluster_total_time)}")
        print(f"- Среднее время на участника: {format_time(avg_iteration_time)}")
        print(f"- Всего обработано участников: {total_participants}")
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
        cluster_log.pop('participants')
        result_cluster['clusters'][str(cluster)]= cluster_log
        save_log(result_cluster, results_dir, "result_log.json")


    # initial_population_data = create_initial_population(config)  первоначальные промты

    # TODO: Здесь будет реализована логика эксперимента:
    # 1. Загрузка данных участников +
    # 2. Инициализация эволюционного алгоритма (GA или DE)
    # 3. Определение фитнес-функции +
    # 4. Запуск эволюции
    # 5. Сохранение результатов
    

    return experiment_log

