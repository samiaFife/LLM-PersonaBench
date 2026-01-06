import pandas as pd
import json
from pathlib import Path

from src.models.registry import get_model
from langchain_core.prompts import ChatPromptTemplate

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

    print(config['model'])
    model = get_model(config['model'])
    #data_participants = pd.read_csv(config['data']['file_path'])
    # initial_population_data = create_initial_population(config)  первоначальные промты

    # TODO: Здесь будет реализована логика эксперимента:
    # 1. Загрузка данных участников
    # 2. Инициализация эволюционного алгоритма (GA или DE)
    # 3. Определение фитнес-функции
    # 4. Запуск эволюции
    # 5. Сохранение результатов
    
    # Пример: сохраняем информацию о начале эксперимента
    experiment_log = {
        'experiment_id': experiment_id,
        'status': 'started',
        'config': config
    }
    
    log_file = results_dir / "experiment_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False, default=str)
    
    print("Эксперимент инициализирован. Логика эксперимента будет реализована позже.")

    # Создаём LangChain prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Ты профессиональный инженер"),
        ("human", "Расскажи что-нибудь о себе")
    ])


    response = model.generate(prompt_template)
    print(f"Ввод: Расскажи что-нибудь о себе")
    print(f"Ответ модели: {response.content}")

    return experiment_log

