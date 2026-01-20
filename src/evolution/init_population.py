import random
from .llm_wrapper import paraphrase  # Используем новую обертку для работы с src.models
from .utils import genotype_to_evoprompt_str

def init_population(base_genotype, config, pop_size, evolution_model):
    """
    Инициализирует популяцию строк-генотипов
    Вариации: paraphrase на формулировках
    
    Args:
        base_genotype: базовый генотип (dict)
        config: конфигурация эксперимента
        pop_size: размер популяции
        evolution_model: объект модели из src.models для эволюции
    """
    base_str = genotype_to_evoprompt_str(base_genotype, config)
    instruction = """You should ALWAYS return the result in the following JSON format:
    {
        "role_definition": "text of the role definition",
        "trait_formulations": {"trait1": "description1", "trait2": "description2"},
        "facet_formulations": {"facet1": "description1", "facet2": "description2"},
        "critic_formulations": "text of the self-criticism instructions"
    }
    """

    params = config['evolution']['genotype_params']
    optimizations = []
    if params.get('role_definition', False):
        optimizations.append("role definition (system prompt)")
    if params.get('trait_formulations', False):
        optimizations.append("personality trait descriptions")
    if params.get('facet_formulations', False):
        optimizations.append("behavioral facet descriptions")
    if params.get('critic_formulations', False):
        optimizations.append("self-criticism instructions")
    optimizations_text = ", ".join(optimizations)
        
    population = [base_str]
    for _ in range(1, pop_size):
        para_prompt = f"""Optimize the following personality prompt components to better simulate human personality in psychological questionnaires, based on Big Five (OCEAN) model and IPIP-NEO facets. The optimized prompt will be used by an LLM to role-play a consistent personality while completing the IPIP-NEO-120 questionnaire, ensuring responses align with one coherent profile (not random or inconsistent).
        Focus on improving: {optimizations_text}

        Current prompt JSON:
        {base_str}

        Requirements:
        1. Make the language more natural, psychologically accurate, and aligned with Big Five/IPIP-NEO research (e.g., ensure traits like Extraversion include facets such as Activity Level or Assertiveness).
        2. Ensure formulations are specific and behaviorally grounded with observable examples of how traits/facets manifest in everyday behaviors or questionnaire responses (e.g., 'tends to strongly agree with statements about seeking excitement in social settings'), without rigid if-then rules.
        3. Maintain consistency with the original personality profile, ensuring stable, reproducible responses in repeated IPIP-NEO-120 tests (optimization evaluated via similarity in questionnaire answers for high test-retest reliability).
        4. Improve clarity for LLM understanding, promoting coherent role-playing.

        response_format: {instruction}"""
        variant = paraphrase(para_prompt, evolution_model, temperature=0.7)
        population.append(variant)
    random.shuffle(population)  # Для randomness
    return population


    