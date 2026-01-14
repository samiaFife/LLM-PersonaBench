from external.evoprompt.args import parse_args as base_parse  # Base argparse

def parse_args_from_yaml(evo_config):
    """
    Парсит evo_config (config['evolution']) в EvoPrompt args
    """
    args = base_parse([])  # Пустой для defaults
    args.evo_mode = evo_config['algorithm'].lower()
    args.popsize = evo_config['population_size']
    args.budget = evo_config['num_generations']  # generations
    args.mutation_prob = evo_config['mutation_rate']
    args.crossover_prob = evo_config['crossover_rate']
    args.sel_mode = evo_config['selection_method']  
    args.llm_type = evo_config['llm_for_evolution']
   
    return args