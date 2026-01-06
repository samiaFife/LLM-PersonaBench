# Структурный псевдокод

# 1. ИМПОРТЫ из EvoPrompt
from external.EvoPrompt.evoluter import EvolutionEngine # или GeneticAlgorithm, DifferentialEvolution
from external.EvoPrompt.population import Population
# Возможно, также нужны: Evaluator (базовый класс), Mutator, Crossover

# Импорты вашего проекта
from src.models.registry import get_model
from src.data.loader import load_participants_for_cluster # подгрузку участников можно сделать прям здесь
from src.prompt_builder import build_full_prompt  # модуль для сборки промта из шаблона
from src.utils.response_parser import parse_questionnaire_response # надо будет написать потом поместим в utils
from src.utils.metrics import calculate_agreement_score # надо будет написать потом поместим в utils

# надо разобраться как этот файл будет запускаться из-под launch_experiment и как в него будут передаваться данные config

# 2. КОНФИГУРАЦИЯ (загружается из config.yaml)
config = {
    'model': 'gpt-4',
    'provider': 'cloud_api',
    'cluster_id': 0, # вопрос стоит ли в одном эксперименте прогонять все варианты кластеров или стоит выделять? 
    'data': 'путь к файлу',
    'num_participants': 50,
    'evolution': {
        'algorithm': 'GA',  # или 'DE'
        'population_size': 10,
        'num_generations': 20,
        'mutation_rate': 0.1,
        # ... другие параметры EvoPrompt
    }
}

# 3. ОПРЕДЕЛЕНИЕ ФИТНЕС-ФУНКЦИИ
# Это САМАЯ ВАЖНАЯ часть. EvoPrompt будет вызывать эту функцию для оценки каждого промта.
def fitness_function(genotype, config):
    """
    genotype: словарь с оптимизируемыми параметрами, например:
        {
            'trait_formulations': {...},
            'facet_formulations': {...},
            'intensity_modifiers': [...],
            'template_structure': '...'
        }
    config: общая конфигурация эксперимента.

    Возвращает: fitness_score (float) - чем выше, тем лучше соответствие человека и модели.
    """
    total_score = 0.0
    num_participants_evaluated = 0

    # Загружаем подвыборку участников для текущего кластера
    participants = load_participants_for_cluster(config['cluster_id'], config['num_participants'])

    for participant in participants:
        # А. Строим полный промт, используя genotype (оптимизируемые параметры) и данные участника
        # Ваш build_full_prompt теперь использует не жесткие facets.py/traits.py, а genotype
        full_prompt = build_full_prompt(
            participant_data=participant,
            base_traits=genotype['trait_formulations'],   # Оптимизируемые!
            base_facets=genotype['facet_formulations'],   # Оптимизируемые!
            intensity_map=genotype['intensity_modifiers'], # Оптимизируемые!
            template_order=genotype['template_structure']  # Оптимизируемые!
        )

        # Б. Генерируем ответ модели на опросник
        llm = get_llm(config['model'])
        raw_response = llm.generate(full_prompt, questionnaire=participant.questionnaire)

        # В. Парсим ответ, извлекая ответы модели на вопросы
        model_answers = parse_questionnaire_response(raw_response)

        # Г. Сравниваем ответы модели и участника, вычисляем метрику
        score = calculate_agreement_score(model_answers, participant.answers)

        total_score += score
        num_participants_evaluated += 1

    # Усредняем оценку по всем участникам
    avg_score = total_score / num_participants_evaluated
    return avg_score

# 4. ИНИЦИАЛИЗАЦИЯ ЭВОЛЮЦИОННОГО ДВИЖКА EvoPrompt
def create_initial_population(config):
    """
    Создает начальную популяцию 'генотипов'.
    В качестве начальных точек можно использовать:
    1. Текущие формулировки из facets.py и traits.py (закомментированный словарь с числами)
    2. Их текстовые описания (текущие prompts)
    3. Случайные вариации на их основе
    """
    initial_genotypes = []
    for _ in range(config['evolution']['population_size']):
        # Пример: создаем генотип, основанный на ваших текущих данных, но с небольшими вариациями
        genotype = {
            'system_formulations': prompt.system['role'],  # отправная точка (базовые утверждени про то как себя вести и кто ты)
            'trait_formulations': traits[str(config['cluster_id'])],  # отправная точка
            'facet_formulations': facets[str(config['cluster_id'])],  # отправная точка
            'intensity_modifiers': ['часто', 'иногда', 'редко'],      # пример списка для оптимизации (возможно нужно больше вариантов)
            'ciritic_formulations': prompt.critic['internal'],        # отправная точка
            'template_structure': 'TRAITS_FACETS_CRITIC'              # пример параметра порядка
            # в соответсвии со всеми этими блоками построить build_full_prompt
        }
        # TODO: добавить небольшую случайную мутацию для разнообразия популяции
        initial_genotypes.append(genotype)
    return initial_genotypes

# 5. ГЛАВНЫЙ ЦИКЛ ЭКСПЕРИМЕНТА
def main():
    # 5.1 Подготовка
    initial_population_data = create_initial_population(config)

    # 5.2 Создание объектов EvoPrompt
    # Population управляет множеством особей (генотипов) и их оценками
    population = Population(
        individuals=initial_population_data,
        fitness_func=lambda genotype: fitness_function(genotype, config)
    )

    # EvolutionEngine (или GA/DE) выполняет эволюционные операции
    evo_engine = EvolutionEngine(
        population=population,
        algorithm=config['evolution']['algorithm'],
        mutation_rate=config['evolution']['mutation_rate'],
        # ... другие параметры EvoPrompt
    )

    # 5.3 ЭВОЛЮЦИОННЫЙ ЦИКЛ
    best_fitness_overall = -float('inf')
    best_genotype_overall = None

    for generation in range(config['evolution']['num_generations']):
        print(f"\n--- Поколение {generation+1} ---")

        # A. EvoPrompt выполняет один шаг эволюции
        # ВНУТРИ этого метода происходит:
        #   1. Выбор родителей (селекция)
        #   2. Применение мутации/скрещивания (с использованием LLM через шаблоны EvoPrompt)
        #   3. Создание новых кандидатов (генотипов)
        #   4. Оценка новых кандидатов (вызов нашей fitness_function)
        #   5. Обновление популяции
        evo_engine.step()

        # Б. Получаем лучший результат текущего поколения
        current_best = population.get_best_individual()
        print(f"Лучшая оценка в поколении: {current_best.fitness}")

        # В. Сохраняем лучший результат за всю историю
        if current_best.fitness > best_fitness_overall:
            best_fitness_overall = current_best.fitness
            best_genotype_overall = current_best.genotype
            # TODO: сохранить best_genotype_overall в файл/базу

        # Г. Логирование и визуализация прогресса (опционально)
        log_generation_stats(generation, population)

    # 6. ФИНАЛИЗАЦИЯ
    print(f"\nЭволюция завершена. Лучшая достигнутая оценка: {best_fitness_overall}")
    print("Лучший генотип (оптимизированные параметры промта):")
    print(best_genotype_overall)

    # TODO: Сохранить финальный лучший промт в формате, готовом к использованию в вашей системе.

if __name__ == "__main__":
    main()