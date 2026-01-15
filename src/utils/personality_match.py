import scipy.stats as sps
from langchain_core.prompts import ChatPromptTemplate

from src.utils.prompt import build_full_prompt
from src.utils.parse import parse_response

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
    response = model.generate(prompt_template)
    model_answers = parse_response(response.content)
    
    if model_answers is None:
        return {'similarity': 0.0, 'avg_diff': 0.0, 'pearson_corr': 0.0}

    fitness = {}
    fitness['similarity'] = 0.0
    fitness['avg_diff'] = 0.0
    fitness['pearson_corr'] = 0.0
    lsit_model_ans = []
    lsit_human_ans = []
    valid_count = 0  # Счетчик валидных ответов (где human_ans is not None)
    for q_id, model_ans in model_answers.items():
        human_ans = participant['i' + str(q_id)]
        lsit_model_ans.append(model_ans)
        lsit_human_ans.append(human_ans)
        if human_ans is not None:
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
    fitness['pearson_corr'] = sps.pearsonr(lsit_model_ans, lsit_human_ans)
    return fitness