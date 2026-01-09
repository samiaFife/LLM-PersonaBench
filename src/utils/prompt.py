import bisect

def get_modifier_bisect(value, modifiers_config):
    """
    Определяет модификатор интенсивности для заданного значения.
    
    Вход:
        value (float): числовое значение для определения интенсивности
        modifiers_config (dict): конфигурация с ключами 'boundaries' (список границ) 
                                 и 'modifiers' (список модификаторов)
    Выход:
        str: модификатор интенсивности, соответствующий интервалу значения
    """
    idx = bisect.bisect_right(modifiers_config['boundaries'], value) - 1
    idx = max(0, min(idx, len(modifiers_config['modifiers']) - 1))
    return modifiers_config['modifiers'][idx]

def build_full_prompt(genotype, task, participant):
    """
    Строит полный промпт для модели на основе генотипа, задачи и данных участника.
    
    Вход:
        genotype (dict): конфигурация персонажа (роль, формулировки traits/facets, 
                         модификаторы интенсивности, внутренний критик)
        task (dict): описание задачи (инструкция, вопросы IPIP-NEO, формат ответа)
        participant (pd.Series): данные участника с оценками по traits/facets и ответами на вопросы
    Выход:
        dict: промпт с ключами 'system' (системное сообщение) и 'human' (запрос пользователя)
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