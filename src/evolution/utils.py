import json

# Адаптировано из EvoPrompt utils.py, плюс наши функции для genotype
def genotype_to_evoprompt_str(genotype, config):
    """
    Конвертирует genotype в строку для эволюции (пока без intensity_modifiers).
    Формат: JSON с оптимизируемыми частями (по genotype_params).
    """
    evo_genotype = {}
    params = config['evolution']['genotype_params']
    
    # Добавляем только то, что оптимизируем
    if params.get('role_definition', False):
        evo_genotype['role_definition'] = genotype['role_definition']
    
    if params.get('trait_formulations', False):
        evo_genotype['trait_formulations'] = genotype['trait_formulations']
    
    if params.get('facet_formulations', False):
        evo_genotype['facet_formulations'] = genotype['facet_formulations']
    
    if params.get('critic_formulations', False):
        evo_genotype['critic_formulations'] = genotype['critic_formulations']

    return json.dumps(evo_genotype, ensure_ascii=False, indent=2)


def clean_evoprompt_response(text):
    """
    Очищает ответ EvoPrompt, оставляя только JSON.
    """
    # Удаляем все до первой {
    start = text.find('{')
    if start == -1:
        return '{}'
    
    # Находим парную закрывающую }
    brace_count = 0
    end = -1
    
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    
    if end == -1:
        return '{}'
    
    return text[start:end]

def parse_str_to_genotype(geno_str, fixed_modifiers, config):
    """
    Парсит строку обратно в genotype, добавляя fixed intensity_modifiers
    Требует, чтобы EvoPrompt возвращал ВАЛИДНЫЙ JSON
    """
    # Очищаем ответ от возможного мусора
    cleaned_str = clean_evoprompt_response(geno_str)
    
    try:
        evo_genotype = json.loads(cleaned_str)
        full_genotype = {'intensity_modifiers': fixed_modifiers}
        
        # Валидируем и добавляем только ожидаемые поля
        params = config['evolution']['genotype_params']
        
        if params.get('role_definition', False):
            if 'role_definition' in evo_genotype:
                full_genotype['role_definition'] = str(evo_genotype['role_definition'])
            else:
                raise ValueError("Missing 'role_definition' in evolved genotype")
        
        if params.get('trait_formulations', False):
            if 'trait_formulations' in evo_genotype:
                full_genotype['trait_formulations'] = dict(evo_genotype['trait_formulations'])
            else:
                raise ValueError("Missing 'trait_formulations' in evolved genotype")
        
        if params.get('facet_formulations', False):
            if 'facet_formulations' in evo_genotype:
                full_genotype['facet_formulations'] = dict(evo_genotype['facet_formulations'])
            else:
                raise ValueError("Missing 'facet_formulations' in evolved genotype")
        
        if params.get('critic_formulations', False):
            if 'critic_formulations' in evo_genotype:
                full_genotype['critic_formulations'] = str(evo_genotype['critic_formulations'])
            else:
                raise ValueError("Missing 'critic_formulations' in evolved genotype")
        
        return full_genotype
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from EvoPrompt: {str(e)}")

# После добавить другие нужные helpers из EvoPrompt utils.py (импортируйте или скопируйте нужные, e.g., read_lines для init)
