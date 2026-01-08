import pandas as pd
import json
import re
from typing import Dict, List, Optional
from pathlib import Path

from src.models.registry import get_model
from langchain_core.prompts import ChatPromptTemplate

from src.prompt.traits import traits
from src.prompt.facets import facets
from src.prompt.system import system

import bisect
def get_modifier_bisect(value, modifiers_config):
    """
    находит позицику вставки и определяет моификатор
    """
    idx = bisect.bisect_right(modifiers_config['boundaries'], value) - 1
    idx = max(0, min(idx, len(modifiers_config['modifiers']) - 1))
    return modifiers_config['modifiers'][idx]

def build_full_prompt(genotype, task, participant):
    """
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
    """
    traits_text = []
    for trait, description in genotype['trait_formulations'].items():
        modifier = get_modifier_bisect(participant[trait], genotype['intensity_modifiers'])
        traits_text.append(f"- This trait ({trait}) describes you {modifier}: {description}")
    traits_text = "\n".join(traits_text)

    facets_text = []
    for facet, description in genotype['facet_formulations'].items():
        modifier = get_modifier_bisect(participant[facet], genotype['intensity_modifiers'])
        facets_text.append(f"- This facet ({facet}) describes you {modifier}: {description}")
    facets_text = "\n".join(facets_text)
    
    system = f"""{genotype['role_definition']}
        Your traits:
        {traits_text}
        Your specific behavioral aspects:
        {facets_text}
        
        Internal reflection guideline:
        {genotype['critic_formulations']}
    """

    questions_text = "\n".join([
        f"{q['id']}. {q['text']}" for q in task['ipip_neo']
    ])
    human = f"""{task['task']}
    Questions:
    {questions_text}

    {task['response_format']}
    """
    prompt = {
        "system": system,
        "human": human
    }
    return prompt

def _validate_and_convert(data):
    """
    Внутренняя валидация: ожидаем list[dict] с question_id и answer.
    """
    if not isinstance(data, list):
        return None
    
    result: Dict[int, int] = {}
    seen_ids = set()
    
    for item in data:
        if not isinstance(item, dict):
            return None
        q_id = item.get("question_id")
        answer = item.get("answer")
        
        if not isinstance(q_id, int) or not isinstance(answer, int):
            return None
        if not (1 <= q_id <= 120) or not (1 <= answer <= 5):
            return None
        if q_id in seen_ids:
            return None  # дубли
        seen_ids.add(q_id)
        result[q_id] = answer
    
    return result

def parse_response(response_content):
    """
    Парсит ответ модели в формат: dict {question_id: answer (1-5)}.
    """
    if not response_content:
        return None
    
    content = response_content.strip()
    
    # Шаг 1: Пытаемся прямой json.loads
    try:
        data = json.loads(content)
        return _validate_and_convert(data)
    except json.JSONDecodeError:
        pass
    
    # Шаг 2: Ищем JSON-массив внутри текста
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
            return _validate_and_convert(data)
        except json.JSONDecodeError:
            pass

    print(f"Не удалось запарсить JSON. Raw content:\n{content[:500]}...")  # debug
    return None

# Функция для расчета соответствия модели участнику
def fitness_function(participant, genotype, task, model):
    """
    config: общая конфигурация эксперимента.
    Возвращает: fitness_score (float) - чем выше, тем лучше соответствие человека и модели.
    """
    prompt = build_full_prompt(genotype, task, participant)
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt["system"]),
            ("human", prompt["human"])
        ])
    response = model.generate(prompt_template)
    model_answers = parse_response(response.content)

    fitness = 0.0
    for q_id, model_ans in model_answers.items():
        human_ans = participant['i' + str(q_id)]
        if human_ans is not None:
            fitness += 1 - abs(model_ans - human_ans) / 4
    fitness /= len(model_answers)

    print(f"Ответ модели: {fitness}")
    return model_answers

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
            'critic_formulations': system['critic_internal'],
            'template_structure': system['template_structure'],
        }
        task = {
            'task': system['task'],
            'ipip_neo': ipip_neo_questions,
            'response_format': system['response_format'],
        }

        test_participants_score = []
        for index, participant in test_participants.iterrows():
            score = fitness_function(participant, genotype, task, model)
            test_participants_score.append(score)

        mean_test_participants_score = sum(test_participants_score) / len(test_participants_score)
        print('Моделирование законченно')




    # initial_population_data = create_initial_population(config)  первоначальные промты

    # TODO: Здесь будет реализована логика эксперимента:
    # 1. Загрузка данных участников +
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

