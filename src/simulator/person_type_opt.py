import pandas as pd
import json
from pathlib import Path

from src.models.registry import get_model
from langchain_core.prompts import ChatPromptTemplate

from src.prompt.traits import traits
from src.prompt.facets import facets
from src.prompt.system import system

def build_full_prompt(genotype, task):

    traits_text = "\n".join([f"- {trait}: {description}" for trait, description in genotype['trait_formulations'].items()])
    facets_text = "\n".join([f"- {facet}: {description}" for facet, description in genotype['facet_formulations'].items()])
    
    system = f"""{genotype['role_definition']}
        Your traits:
        {t}
        Your specific behavioral aspects:
        {facets_text}
    """

    human = f"""{task['task']}
    Questions:
    {task['ipip_neo_questions']}

    {task['response_format']}
    """
    prompt = {
        "system": system,
        "human": human
    }
    return prompt

def fitness_function(genotype, task, model, config):
    """
    config: общая конфигурация эксперимента.
    Возвращает: fitness_score (float) - чем выше, тем лучше соответствие человека и модели.
    """
    # перед педедачей в промт нужно определить интенсивность относительно конкретного человека
    prompt = build_full_prompt(genotype, task)
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt["system"]),
            ("human", prompt["human"])
        ])
    response = model.generate(prompt_template)
    print(f"Ввод: Охарактеризуй этот кластер")
    print(f"Ответ модели: {response.content}")

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

    model = get_model(config['model'])
    data_participants = pd.read_csv(config['data']['file_path'])
    
    with open('data/IPIP-NEO/120/questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    ipip_neo_questions = data.get('questions')

    for cluster in config['data']['clusters']:
        test_participants = data_participants[data_participants['clusters'] == cluster].iloc[:config['data']['num_participants']]
        genotype = {
            'role_definition': system['role'],
            'trait_formulations': traits[cluster],
            'facet_formulations': facets[cluster],
            'intensity_modifiers': system['intensity_modifiers'],
            'ciritic_formulations': system['critic_internal'],
            'template_structure': system['template_structure'],
        }
        task = {
            'task': system['task'],
            'ipip_neo': ipip_neo_questions,
            'response_format': system['response_format'],
        }

        fitness_function(genotype, task, model)


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
    

    return experiment_log

