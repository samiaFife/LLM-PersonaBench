import time
from external.evoprompt.evaluator import Evaluator  # Base из EvoPrompt
from src.utils.time import format_time
from src.utils.save_result import save_log
from src.utils.personality_match import fitness_function, evaluate_participants_batch
from .utils import parse_str_to_genotype, validate_and_repair_genotype

class MyEvaluator(Evaluator):
    def __init__(self, args, task, model, fixed_modifiers, template_genotype=None, config=None):
        super().__init__(args)
        self.task = task  # Из person_type_opt (вопросы, формат)
        self.model = model  # LLM для generate
        self.fixed_modifiers = fixed_modifiers  # Из system
        self.dev_participants = []  # Устанавливаем позже
        self.config = config  # Сохраняем config для использования в forward
        self.template_genotype = template_genotype or {}
        # Пачка участников для параллельных запросов к модели (1 = только последовательно)
        cfg = config or {}
        self.participant_batch_size = int((cfg.get('evolution') or {}).get('participant_batch_size', 1) or 1)

    def forward(self, prompt_str, config=None):
        """
        Вычисляет скалярный фитнес: усреднённый по dev_participants.
        prompt_str — строка-шаблон genotype без modifiers.
        Возвращает кортеж: (mean_similarity, detailed_scores)
        где detailed_scores - список словарей с метриками по каждому участнику
        """
        # Проверка: dev_participants должен быть установлен перед вызовом forward
        # Проверяем для DataFrame (используем .empty) и для списка (используем len)
        if self.dev_participants is None:
            raise ValueError(
                "dev_participants не установлен! Установите evaluator.dev_participants перед вызовом forward(). "
                "Это должно быть сделано в person_type_opt.py после создания evaluator."
            )
        # Для DataFrame используем .empty, для списка - len
        if hasattr(self.dev_participants, 'empty'):
            if self.dev_participants.empty:
                raise ValueError(
                    "dev_participants пуст! Установите evaluator.dev_participants перед вызовом forward(). "
                    "Это должно быть сделано в person_type_opt.py после создания evaluator."
                )
        elif len(self.dev_participants) == 0:
            raise ValueError(
                "dev_participants пуст! Установите evaluator.dev_participants перед вызовом forward(). "
                "Это должно быть сделано в person_type_opt.py после создания evaluator."
            )
        
        # Нужен config для parse_str_to_genotype
        if config is None:
            config = getattr(self, 'config', None)
        if config is None:
            # Если config не передан, создаем минимальный из args
            config = {'evolution': {'genotype_params': {}}}
        # Чиним неполный/битый генотип от модели, используя шаблон
        repaired_str = validate_and_repair_genotype(
            prompt_str, self.fixed_modifiers, self.template_genotype, config
        )
        genotype = parse_str_to_genotype(repaired_str, self.fixed_modifiers, config)
        scores = evaluate_participants_batch(
            self.dev_participants, genotype, self.task, self.model, self.participant_batch_size
        )
        test_participants_scores = []
        for score in scores:
            participant_score = {
                'similarity': score['similarity'],
                'avg_diff': score['avg_diff'],
                'pearson_corr': score['pearson_corr'][0] if isinstance(score['pearson_corr'], tuple) else score['pearson_corr']
            }
            test_participants_scores.append(participant_score)

        mean_similarity_participants = sum([i['similarity'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_avg_diff_participants = sum([i['avg_diff'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_pearson_corr_participants = sum([i['pearson_corr'] for i in test_participants_scores]) / len(test_participants_scores)

        # Сохраняем детальные скоры для логирования
        self.last_detailed_scores = {
            'mean_similarity': mean_similarity_participants,
            'mean_avg_diff': mean_avg_diff_participants,
            'mean_pearson_corr': mean_pearson_corr_participants,
            'participants_scores': test_participants_scores
        }

        return mean_similarity_participants
