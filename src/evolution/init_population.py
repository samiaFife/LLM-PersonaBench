import random
from external.evoprompt.llm_client import paraphrase  # Для вариаций
from .utils import genotype_to_evoprompt_str

def init_population(base_genotype, config, pop_size, llm_model):
    """
    Инициализирует популяцию строк-генотипов
    Вариации: paraphrase на формулировках
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
        para_prompt = f"""Optimize the following personality prompt components to better simulate human personality in psychological questionnaires. 
        Focus on improving: {optimizations_text}
        
        Current prompt JSON:
        {base_str}
        
        Requirements:
        1. Make the language more natural and psychologically accurate
        2. Ensure formulations are specific and behaviorally grounded
        3. Maintain consistency with the original personality profile
        4. Improve clarity for LLM understanding
        
        response_format: {instruction}"""
        variant = paraphrase(para_prompt, llm_model, config['llm_for_evolution'])
        population.append(variant)
    random.shuffle(population)  # Для randomness
    return population


    