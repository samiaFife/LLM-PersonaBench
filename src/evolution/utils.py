import json

from src.utils.five_factor import TRAIT_NAMES, FACET_NAMES

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


def _normalize_trait_keys(d: dict) -> dict:
    """
    Приводит ключи trait_formulations к нижнему регистру,
    чтобы совпадали с колонками датасета (openness, conscientiousness, ...).
    """
    normalized = {}
    for k, v in (d or {}).items():
        key = str(k).strip()
        key_lower = key.lower()
        if key_lower in TRAIT_NAMES:
            normalized[key_lower] = v
        else:
            normalized[key] = v
    return normalized


def _normalize_facet_keys(d: dict) -> dict:
    """
    Приводит ключи facet_formulations к нижнему регистру (facet_*)
    для соответствия колонкам датасета.
    """
    normalized = {}
    for k, v in (d or {}).items():
        key = str(k).strip()
        key_lower = key.lower()
        if key_lower in FACET_NAMES:
            normalized[key_lower] = v
        elif key_lower.startswith("facet_") and key_lower.replace(" ", "_") in FACET_NAMES:
            normalized[key_lower.replace(" ", "_")] = v
        else:
            normalized[key] = v
    return normalized

def parse_str_to_genotype(geno_str, fixed_modifiers, config, template_genotype=None):
    """
    Парсит строку обратно в genotype, добавляя fixed intensity_modifiers.
    Если передан template_genotype, из него копируются trait_targets и facet_targets
    (целевые значения для модификатора по совпадению; не эволюционируют).
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
                full_genotype['trait_formulations'] = _normalize_trait_keys(dict(evo_genotype['trait_formulations']))
            else:
                raise ValueError("Missing 'trait_formulations' in evolved genotype")
        
        if params.get('facet_formulations', False):
            if 'facet_formulations' in evo_genotype:
                full_genotype['facet_formulations'] = _normalize_facet_keys(dict(evo_genotype['facet_formulations']))
            else:
                raise ValueError("Missing 'facet_formulations' in evolved genotype")
        
        if params.get('critic_formulations', False):
            if 'critic_formulations' in evo_genotype:
                full_genotype['critic_formulations'] = str(evo_genotype['critic_formulations'])
            else:
                raise ValueError("Missing 'critic_formulations' in evolved genotype")
        
        if template_genotype:
            full_genotype['trait_targets'] = template_genotype.get('trait_targets', {})
            full_genotype['facet_targets'] = template_genotype.get('facet_targets', {})
        
        return full_genotype
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from EvoPrompt: {str(e)}")
    

def validate_and_repair_genotype(geno_str, fixed_modifiers, template_genotype, config=None):
    """
    Валидирует и чинит genotype JSON, если LLM вернула неполный/битый JSON.
    """
    try:
        # Создаем минимальный config если не передан
        if config is None:
            config = {'evolution': {'genotype_params': {}}}
            # Определяем какие поля должны быть на основе template_genotype
            if 'role_definition' in template_genotype:
                config['evolution']['genotype_params']['role_definition'] = True
            if 'trait_formulations' in template_genotype:
                config['evolution']['genotype_params']['trait_formulations'] = True
            if 'facet_formulations' in template_genotype:
                config['evolution']['genotype_params']['facet_formulations'] = True
            if 'critic_formulations' in template_genotype:
                config['evolution']['genotype_params']['critic_formulations'] = True
        
        genotype = parse_str_to_genotype(geno_str, fixed_modifiers, config, template_genotype=template_genotype)
        
        # Проверяем обязательные поля
        repaired = {}
        
        # Копируем существующие поля (trait_targets/facet_targets не в JSON эволюции, берутся из template в parse)
        for key in ['role_definition', 'trait_formulations', 'facet_formulations', 'critic_formulations']:
            if key in genotype:
                repaired[key] = genotype[key]
            elif key in template_genotype:
                # Заполняем из шаблона, если поле отсутствует
                repaired[key] = template_genotype[key]
            else:
                # Создаем пустое значение
                if 'formulations' in key:
                    repaired[key] = {}
                else:
                    repaired[key] = ""
        
        return json.dumps(repaired, ensure_ascii=False)
        
    except (json.JSONDecodeError, ValueError):
        # Если JSON битый, возвращаем шаблон
        return json.dumps(template_genotype, ensure_ascii=False)


# После добавить другие нужные helpers из EvoPrompt utils.py (импортируйте или скопируйте нужные, e.g., read_lines для init)
