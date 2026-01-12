from external.evoprompt.llm_client import llm_query  # Core для мутации

def my_mutate(prompt_str, prob, llm_model, config):
    if random.random() > prob:
        return prompt_str
    # Адаптированный шаблон из template_ga.py: focus на trait/facet_formulations
    template = "Мутируй описание черт личности для лучшей coherentности, не меняя суть: {prompt}"
    return llm_query(template.format(prompt=prompt_str), llm_model, config['llm_for_evolution'])

def my_crossover(parent1, parent2, llm_model, config):
    # Аналогично, из evoluter.py ga_mutate_crossover
    template = "Скрести формулировки из двух шаблонов: Parent1: {p1}\nParent2: {p2}\nСохрани структуру."
    child1 = llm_query(template.format(p1=parent1, p2=parent2), llm_model, config['llm_for_evolution'])
    child2 = llm_query(template.format(p1=parent2, p2=parent1), llm_model, config['llm_for_evolution'])
    return child1, child2

# В evoluter.py (скопированном) переопределите ga_mutate_crossover/de_mutate на эти, если нужно.



import random
import json
from typing import Dict, Tuple, List, Optional
from .utils import parse_str_to_genotype
from external.evoprompt.llm_client import llm_query

def get_optimization_fields_instruction(config):
    """Создает инструкцию о том, какие поля оптимизируются"""
    params = config['evolution']['genotype_params']
    fields = []
    
    if params.get('role_definition', False):
        fields.append("'role_definition' (system role description)")
    if params.get('trait_formulations', False):
        fields.append("'trait_formulations' (Big Five personality trait descriptions)")
    if params.get('facet_formulations', False):
        fields.append("'facet_formulations' (specific behavioral facet descriptions)")
    if params.get('critic_formulations', False):
        fields.append("'critic_formulations' (self-criticism instructions)")
    
    if fields:
        return f"Focus on optimizing these fields: {', '.join(fields)}"
    return "Optimize the entire personality description"

def get_json_structure_instruction():
    """Инструкция о формате JSON"""
    return """IMPORTANT: You must ALWAYS return valid JSON in this exact structure:
    {
        "role_definition": "text description of the system role",
        "trait_formulations": {
            "trait_name1": "description of trait 1",
            "trait_name2": "description of trait 2"
        },
        "facet_formulations": {
            "facet_name1": "description of facet 1", 
            "facet_name2": "description of facet 2"
        },
        "critic_formulations": "text of self-criticism instructions"
    }

    Return ONLY the JSON object, no additional text.
    """

def personality_mutation(prompt_str, mutation_rate, llm_model, config):
    """
    Мутация personality промта через LLM.
    Аналог GA mutation из EvoPrompt, но для JSON-формата personality.
    """
    if random.random() > mutation_rate:
        return prompt_str
    
    try:
        # Создаем контекст для мутации на основе того, что оптимизируем
        optimization_instruction = get_optimization_fields_instruction(config)
        
        # Шаблон мутации, аналогичный EvoPrompt template_ga.py
        mutation_template = f"""As an expert in personality psychology and prompt engineering, 
        improve the following personality description for better simulation of human behavior in psychological questionnaires.

        {optimization_instruction}

        Current personality description (JSON format):
        {prompt_str}

        Improvement instructions:
        1. Make ONE specific change to improve psychological accuracy
        2. Enhance clarity for LLM interpretation  
        3. Maintain consistency with personality theory
        4. Keep behavioral predictions specific and testable

        {get_json_structure_instruction()}
        """
        
        # Вызываем LLM для мутации
        mutated = llm_query(
            data=mutation_template,
            client=llm_model,
            type=config['evolution']['llm_for_evolution'],
            task=False,
            temperature=0.7,  # Небольшая случайность для мутации
            max_tokens=1500
        )
        
        # Проверяем, что вернулась строка (может вернуться список)
        if isinstance(mutated, list):
            mutated = mutated[0] if mutated else prompt_str
        
        return mutated
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in mutation: {e}")
        return prompt_str
    except Exception as e:
        print(f"Mutation error: {e}")
        return prompt_str

def personality_crossover(parent1_str, parent2_str, llm_model, config, fixed_modifiers):
    """
    Кроссовер двух personality промтов через LLM.
    Аналог GA crossover из EvoPrompt, но для JSON-формата.
    """
    try:
        # Парсим родителей для валидации
        parent1 = parse_str_to_genotype(parent1_str, fixed_modifiers)
        parent2 = parse_str_to_genotype(parent2_str, fixed_modifiers)
        
        optimization_instruction = get_optimization_fields_instruction(config)
        
        # Шаблон кроссовера, аналогичный EvoPrompt
        crossover_template = f"""As an expert in personality psychology, create two new personality descriptions 
        by combining the best elements from two existing descriptions.

        {optimization_instruction}

        Parent 1 (JSON):
        {json.dumps(parent1, indent=2)}

        Parent 2 (JSON):
        {json.dumps(parent2, indent=2)}

        Combination instructions:
        1. For child 1: Take the most effective elements from parent 1 and supplement with complementary elements from parent 2
        2. For child 2: Take the most effective elements from parent 2 and supplement with complementary elements from parent 1  
        3. Ensure both children are psychologically coherent and consistent
        4. Maintain the exact same JSON structure

        Return TWO separate JSON objects in this format:
        {{
        "child1": {{...}},
        "child2": {{...}}
        }}

        Return ONLY the JSON object with two children, no additional text.
        """
        
        # Вызываем LLM для кроссовера
        crossover_result = llm_query(
            data=crossover_template,
            client=llm_model,
            type=config['evolution']['llm_for_evolution'],
            task=False,
            temperature=0.5,  # Умеренная случайность для разнообразия
            max_tokens=2500
        )
        
        if isinstance(crossover_result, list):
            crossover_result = crossover_result[0] if crossover_result else '{"child1": {}, "child2": {}}'
        
        try:
            result_dict = json.loads(crossover_result)
            child1 = json.dumps(result_dict.get('child1', parent1), ensure_ascii=False)
            child2 = json.dumps(result_dict.get('child2', parent2), ensure_ascii=False)
            return child1, child2
        except json.JSONDecodeError:
            # Fallback: вернуть родителей, если не удалось распарсить
            return parent1_str, parent2_str
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in crossover: {e}")
        return parent1_str, parent2_str
    except Exception as e:
        print(f"Crossover error: {e}")
        return parent1_str, parent2_str

##################
# Все функции ниже надо доработать
##################

def personality_selection(population: List[str], fitness_scores: List[float], 
                         selection_method: str = "tournament", tournament_size: int = 3) -> Tuple[str, str]:
    """
    Селекция родителей для кроссовера.
    Аналог selection из EvoPrompt genetic_algorithm.py
    """
    if len(population) < 2:
        return population[0], population[0] if population else ("", "")
    
    if selection_method == "tournament":
        # Турнирная селекция
        def tournament_select():
            contestants = random.sample(list(zip(population, fitness_scores)), 
                                       min(tournament_size, len(population)))
            return max(contestants, key=lambda x: x[1])[0]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        # Гарантируем разных родителей (если популяция больше 1)
        while parent1 == parent2 and len(population) > 1:
            parent2 = tournament_select()
            
    elif selection_method == "roulette":
        # Рулеточная селекция (пропорциональная фитнесу)
        min_score = min(fitness_scores)
        adjusted_scores = [s - min_score + 0.001 for s in fitness_scores]
        total = sum(adjusted_scores)
        
        if total <= 0:
            parent1, parent2 = random.sample(population, 2)
        else:
            probs = [s / total for s in adjusted_scores]
            parent1 = random.choices(population, weights=probs, k=1)[0]
            parent2 = random.choices(population, weights=probs, k=1)[0]
            
            while parent1 == parent2 and len(population) > 1:
                parent2 = random.choices(population, weights=probs, k=1)[0]
    
    else:  # random selection
        parent1, parent2 = random.sample(population, 2)
    
    return parent1, parent2

def validate_and_repair_genotype(geno_str: str, template_genotype: Dict) -> str:
    """
    Валидирует и чинит genotype JSON, если LLM вернула неполный/битый JSON.
    """
    try:
        genotype = json.loads(geno_str)
        
        # Проверяем обязательные поля
        repaired = {}
        
        # Копируем существующие поля
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
        
    except json.JSONDecodeError:
        # Если JSON битый, возвращаем шаблон
        return json.dumps(template_genotype, ensure_ascii=False)

# Алиасы для совместимости с существующим кодом
my_mutate = personality_mutation
my_crossover = personality_crossover