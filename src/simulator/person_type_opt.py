import pandas as pd
import scipy.stats as sps
import json

import time
from typing import Dict, List, Optional
from pathlib import Path

from src.models.registry import get_model
from langchain_core.prompts import ChatPromptTemplate

from src.prompt.traits import traits
from src.prompt.facets import facets
from src.prompt.system import system

from src.utils.prompt import build_full_prompt
from src.utils.parse import parse_response
from src.utils.time import format_time
from src.utils.save_result import save_log

def fitness_function(participant, genotype, task, model):
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
    print(f"{'='*70}")
    print(f"🔄 Отправка запроса к модели...")
    response = model.generate(prompt_template)
    model_answers = parse_response(response.content)
    
    if model_answers is None:
        print(f"❌ Ошибка: не удалось получить ответы модели\n")
        return {'similarity': 0.0, 'avg_diff': 0.0, 'pearson_corr': 0.0}
    
    print(f"✅ Получено ответов от модели: {len(model_answers)}")

    fitness = {}
    fitness['similarity'] = 0.0
    fitness['avg_diff'] = 0.0
    fitness['pearson_corr'] = 0.0
    lsit_model_ans = []
    lsit_human_ans = []
    for q_id, model_ans in model_answers.items():
        human_ans = participant['i' + str(q_id)]
        lsit_model_ans.append(model_ans)
        lsit_human_ans.append(human_ans)
        if human_ans is not None:
            fitness['similarity'] += 1 - abs(model_ans - human_ans) / 4
            fitness['avg_diff'] += abs(model_ans - human_ans)
    fitness['similarity'] /= len(model_answers)
    fitness['avg_diff'] /= len(model_answers)
    fitness['pearson_corr'] = sps.pearsonr(lsit_model_ans, lsit_human_ans)

    print(f"📊 Результаты оценки соответствия:")
    print(f"- Схожесть (similarity): {fitness['similarity']:.4f}")
    print(f"- Средняя разница (avg_diff): {fitness['avg_diff']:.4f}")
    print(f"- Корреляция Пирсона (pearson_corr): {fitness['pearson_corr'][0]:.4f}" if isinstance(fitness['pearson_corr'], tuple) else f"   • Корреляция Пирсона (pearson_corr): {fitness['pearson_corr']:.4f}")
    print(f"{'='*70}\n")
    return fitness

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

