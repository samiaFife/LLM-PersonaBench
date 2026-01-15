import argparse
import sys
from external.evoprompt.args import parse_args as base_parse  # Base argparse

def parse_args_from_yaml(evo_config):
    """
    Парсит evo_config (config['evolution']) в EvoPrompt args
    """
    # Сохраняем оригинальный sys.argv
    original_argv = sys.argv.copy()
    try:
        # Временно заменяем sys.argv на минимальный список для получения дефолтных значений
        # parse_args() читает из sys.argv, поэтому нужно оставить хотя бы имя скрипта
        script_name = sys.argv[0] if sys.argv else 'script.py'
        sys.argv = [script_name]
        args = base_parse()  # Вызываем без аргументов для defaults
    finally:
        # Восстанавливаем оригинальный sys.argv
        sys.argv = original_argv
    
    # Устанавливаем значения из конфига
    args.evo_mode = evo_config['algorithm'].lower()
    args.popsize = evo_config['population_size']
    args.budget = evo_config['num_generations']  # generations
    args.mutation_prob = evo_config['mutation_rate']
    args.crossover_prob = evo_config['crossover_rate']
    args.sel_mode = evo_config['selection_method']  
    args.llm_type = evo_config['llm_for_evolution']
    
    # Устанавливаем дефолтные значения для полей, используемых в Evaluator.__init__
    # если они не были установлены через base_parse()
    if not hasattr(args, 'dataset') or args.dataset is None:
        args.dataset = "sst2"  # Дефолт из args.py
    if not hasattr(args, 'task') or args.task is None:
        args.task = "cls"  # Дефолт для нашей задачи
    if not hasattr(args, 'position') or args.position is None:
        args.position = "pre"  # Дефолт из args.py
    if not hasattr(args, 'language_model') or args.language_model is None:
        args.language_model = "gpt"  # Дефолт для нашей задачи
    if not hasattr(args, 'setting') or args.setting is None:
        args.setting = "default"  # Дефолт из args.py
    if not hasattr(args, 'output') or args.output is None:
        # output будет установлен позже в person_type_opt.py, но для безопасности
        args.output = "./output"
   
    return args