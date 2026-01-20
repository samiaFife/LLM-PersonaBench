from .llm_wrapper import llm_query  # Используем новую обертку для работы с src.models
import random
import json
from typing import Dict, Tuple, List, Optional
from .utils import parse_str_to_genotype

def my_mutate(prompt_str, prob, evolution_model, config):
    """
    Старая функция-алиас для совместимости.
    evolution_model - объект модели из src.models
    """
    if random.random() > prob:
        return prompt_str
    # Адаптированный шаблон из template_ga.py: focus на trait/facet_formulations
    template = "Мутируй описание черт личности для лучшей coherentности, не меняя суть: {prompt}"
    return llm_query(template.format(prompt=prompt_str), evolution_model, task=False, temperature=0.7)

def my_crossover(parent1, parent2, evolution_model, config):
    """
    Старая функция-алиас для совместимости.
    evolution_model - объект модели из src.models
    """
    # Аналогично, из evoluter.py ga_mutate_crossover
    template = "Скрести формулировки из двух шаблонов: Parent1: {p1}\nParent2: {p2}\nСохрани структуру."
    child1 = llm_query(template.format(p1=parent1, p2=parent2), evolution_model, task=False, temperature=0.5)
    child2 = llm_query(template.format(p1=parent2, p2=parent1), evolution_model, task=False, temperature=0.5)
    return child1, child2

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

def personality_mutation(prompt_str, mutation_rate, evolution_model, config):
    """
    Мутация personality промта через LLM.
    Аналог GA mutation из EvoPrompt, но для JSON-формата personality.
    
    Args:
        prompt_str: строка с JSON-генотипом
        mutation_rate: вероятность мутации
        evolution_model: объект модели из src.models для эволюции
        config: конфигурация эксперимента
    """
    if random.random() > mutation_rate:
        return prompt_str
    
    try:
        # Создаем контекст для мутации на основе того, что оптимизируем
        optimization_instruction = get_optimization_fields_instruction(config)
        
        # Шаблон мутации, аналогичный EvoPrompt template_ga.py
        mutation_template = f"""Optimize the following personality prompt components to better simulate human personality in psychological questionnaires, based on Big Five (OCEAN) model and IPIP-NEO facets. The optimized prompt will be used by an LLM to role-play a consistent personality while completing the IPIP-NEO-120 questionnaire, ensuring responses align with one coherent profile (not random or inconsistent).

        Focus on improving: {optimization_instruction}

        Current prompt JSON:
        {prompt_str}

        Requirements:
        1. Make small, targeted changes to improve psychological coherence, aligned with Big Five/IPIP-NEO research (e.g., refine traits like Neuroticism to include facets such as Anxiety or Depression).
        2. Ensure the optimized descriptions are psychologically coherent and consistent, with trait and facet formulations behaviorally grounded using specific, observable examples of manifestations in behaviors or questionnaire responses (e.g., 'often disagrees with statements about feeling calm under pressure'), without rigid if-then rules.
        3. Maintain the original meaning and personality profile, promoting stable, reproducible responses in repeated IPIP-NEO-120 tests (optimization evaluated via similarity in questionnaire answers for high test-retest reliability).
        4. Make the language more natural and precise for LLM role-playing.

        {get_json_structure_instruction()}"""
        
        # Вызываем LLM для мутации через новую обертку
        mutated = llm_query(
            data=mutation_template,
            model=evolution_model,
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

def personality_crossover(parent1_str, parent2_str, evolution_model, config, fixed_modifiers):
    """
    Кроссовер двух personality промтов через LLM.
    Аналог GA crossover из EvoPrompt, но для JSON-формата.
    
    Args:
        parent1_str: строка с JSON-генотипом первого родителя
        parent2_str: строка с JSON-генотипом второго родителя
        evolution_model: объект модели из src.models для эволюции
        config: конфигурация эксперимента
        fixed_modifiers: фиксированные модификаторы интенсивности
    """
    try:
        # Парсим родителей для валидации (нужен config)
        if not config or 'evolution' not in config:
            raise ValueError("config with 'evolution' key is required for crossover")
        parent1 = parse_str_to_genotype(parent1_str, fixed_modifiers, config)
        parent2 = parse_str_to_genotype(parent2_str, fixed_modifiers, config)
        
        optimization_instruction = get_optimization_fields_instruction(config)
        
        # Шаблон кроссовера, аналогичный EvoPrompt
        crossover_template = f"""Combine and optimize formulations from two parent personality prompts to create two improved child versions, better simulating human personality in psychological questionnaires based on Big Five (OCEAN) model and IPIP-NEO facets. The resulting prompts will be used by an LLM to role-play a consistent personality while completing the IPIP-NEO-120 questionnaire, ensuring responses align with one coherent profile (not random or inconsistent).

        Focus on optimizing these fields: {optimization_instruction}

        Parent1 JSON:
        {json.dumps(parent1, indent=2)}

        Parent2 JSON:
        {json.dumps(parent2, indent=2)}

        Requirements:
        1. Blend the best elements from both parents, making changes aligned with Big Five/IPIP-NEO research (e.g., merge Extraversion facets while preserving psychological accuracy).
        2. Ensure combined formulations are specific and behaviorally grounded with observable examples of how traits/facets manifest in behaviors or questionnaire responses (e.g., 'frequently agrees with items about enjoying group activities'), without rigid if-then rules.
        3. Maintain consistency with the original profiles, ensuring stable, reproducible responses in repeated IPIP-NEO-120 tests (optimization evaluated via similarity in questionnaire answers for high test-retest reliability).
        4. Make the two children psychologically coherent and consistent.
        5. Maintain the exact same JSON structure.

        Return TWO separate JSON objects in this format:
        {{
        "child1": {{...}},
        "child2": {{...}}
        }}

        Return ONLY the JSON object with two children, no additional text.
        """
        
        # Вызываем LLM для кроссовера через новую обертку
        crossover_result = llm_query(
            data=crossover_template,
            model=evolution_model,
            task=False,
            temperature=0.5,  # Умеренная случайность для разнообразия
            max_tokens=2500
        )
        
        if isinstance(crossover_result, list):
            crossover_result = crossover_result[0] if crossover_result else '{"child1": {}, "child2": {}}'
        
        try:
            result_dict = json.loads(crossover_result)
            child1 = json.dumps(result_dict['child1'], ensure_ascii=False)
            child2 = json.dumps(result_dict['child2'], ensure_ascii=False)
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

def personality_selection(population, fitness_scores, 
                         selection_method, tournament_size):
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


# Алиасы для совместимости с существующим кодом
my_mutate = personality_mutation
my_crossover = personality_crossover