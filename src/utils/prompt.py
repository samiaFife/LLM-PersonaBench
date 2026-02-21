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


def get_modifier_by_match(participant_value: float, target_value: float, modifiers_config: dict) -> str:
    """
    Определяет модификатор интенсивности по степени совпадения значения участника
    с целевым значением, для которого было построено утверждение.
    Если утверждение строилось для низких значений черты и у участника низкие значения —
    утверждение описывает его сильно (very strongly). Если целевое и значение участника
    расходятся — слабо (very little).
    
    Вход:
        participant_value (float): значение черты/фасета участника (0–100)
        target_value (float): целевое значение, для которого построено утверждение
        modifiers_config (dict): конфигурация с 'boundaries' и 'modifiers'
    Выход:
        str: модификатор (напр. "very strongly" при хорошем совпадении, "very little" при расхождении)
    """
    distance = abs(float(participant_value) - float(target_value))
    boundaries = modifiers_config.get('boundaries', [0, 20, 40, 60, 80, 100])
    modifiers = modifiers_config.get('modifiers', [
        "very little", "slightly", "moderately", "quite strongly", "very strongly"
    ])
    # Малое расстояние (хорошее совпадение) → сильный модификатор; большое → слабый.
    # Границы заданы для значения; для расстояния используем те же и обратный порядок модификаторов.
    idx = bisect.bisect_right(boundaries, distance) - 1
    idx = max(0, min(idx, len(modifiers) - 1))
    reverse_idx = len(modifiers) - 1 - idx
    return modifiers[reverse_idx]

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
    trait_targets = genotype.get('trait_targets') or {}
    facet_targets = genotype.get('facet_targets') or {}
    modifiers_cfg = genotype['intensity_modifiers']

    traits_text = []
    for trait, description in genotype['trait_formulations'].items():
        value = participant.get(trait, participant.get(str(trait).lower()))
        if value is None:
            continue
        target = trait_targets.get(trait, trait_targets.get(str(trait).lower()))
        if target is not None:
            modifier = get_modifier_by_match(value, target, modifiers_cfg)
        else:
            modifier = get_modifier_bisect(value, modifiers_cfg)
        traits_text.append(f"- This trait ({trait}) describes you {modifier}: {description}")
    traits_text = "\n".join(traits_text)

    facets_text = []
    for facet, description in genotype['facet_formulations'].items():
        value = participant.get(facet, participant.get(str(facet).lower()))
        if value is None:
            continue
        target = facet_targets.get(facet, facet_targets.get(str(facet).lower()))
        if target is not None:
            modifier = get_modifier_by_match(value, target, modifiers_cfg)
        else:
            modifier = get_modifier_bisect(value, modifiers_cfg)
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