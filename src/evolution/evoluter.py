import os
import random
import json
import numpy as np

from src.evolution.operators import my_mutate, my_crossover
from src.evolution.utils import clean_evoprompt_response, validate_and_repair_genotype, genotype_to_evoprompt_str, parse_str_to_genotype
from src.utils.save_result import save_log  # Опционально для логов


class Evoluter:
    """
    Базовый класс для эволюции (адаптировано из EvoPrompt).
    """
    def __init__(self, args, evaluator, evolution_model=None, config=None):
        self.args = args
        self.evaluator = evaluator
        self.evolution_model = evolution_model  # Модель для эволюции (из src.models)
        self.config = config  # Конфигурация эксперимента
        self.population = []  # Список строк-генотипов (JSON)
        self.scores = []
        self.generation_logs = []  # Для сохранения истории поколений

    def evolute(self):
        """
        Основной цикл эволюции.
        """
        raise NotImplementedError("Реализуется в подклассах")


class GAEvoluter(Evoluter):
    """
    Genetic Algorithm эволюция (адаптировано под нашу задачу).
    """
    def __init__(self, args, evaluator, evolution_model=None, config=None):
        super().__init__(args, evaluator, evolution_model, config)
        self.population_size = args.popsize
        self.num_generations = args.budget  # Или args.num_generations
        self.mutation_prob = args.mutation_prob
        self.crossover_prob = args.crossover_prob
        self.selection_method = args.sel_mode.lower()  # 'tournament', 'roulette', etc.
        self.detailed_scores_per_prompt = []  # Инициализируем список для детальных скоров

    def evaluate_population(self):
        """
        Оценивает всю популяцию, возвращает scores.
        Сохраняет детальные скоры по участникам для каждого промта.
        """
        self.scores = []
        self.detailed_scores_per_prompt = []  # Детальные скоры для каждого промта
        
        for prompt_str in self.population:
            raw_score = self.evaluator.forward(prompt_str, self.config)
            score = raw_score if isinstance(raw_score, float) else 0.0  # На случай ошибки
            self.scores.append(score)
            
            # Сохраняем детальные скоры (все три метрики по участникам)
            if hasattr(self.evaluator, 'last_detailed_scores'):
                self.detailed_scores_per_prompt.append(self.evaluator.last_detailed_scores.copy())
            else:
                # Fallback если детальные скоры не сохранены
                self.detailed_scores_per_prompt.append({
                    'mean_similarity': score,
                    'mean_avg_diff': 0.0,
                    'mean_pearson_corr': 0.0,
                    'participants_scores': []
                })
        
        # Сортируем популяцию по scores descending (лучший на [0])
        sorted_idx = np.argsort(self.scores)[::-1]
        self.population = [self.population[i] for i in sorted_idx]
        self.scores = [self.scores[i] for i in sorted_idx]
        self.detailed_scores_per_prompt = [self.detailed_scores_per_prompt[i] for i in sorted_idx]

    def select_parents(self) -> tuple:
        """
        Выбор родителей по методу из args.sel_mode.
        """
        if self.selection_method == "tournament":
            # ВАЖНО: при popsize=2..3 и tournament_size==popsize турнир всегда включает лучшего,
            # поэтому оба родителя могут выбираться одинаковыми и цикл ниже "залипает".
            n = len(self.population)
            if n <= 1:
                return self.population[0], self.population[0]
            if n == 2:
                # единственная разумная пара
                return self.population[0], self.population[1]

            # Турнирная селекция: берём размер < n, чтобы был шанс выбрать не только лучшего
            tournament_size = min(3, n - 1)

            def tournament():
                candidates = random.sample(list(enumerate(self.scores)), tournament_size)
                return max(candidates, key=lambda x: x[1])[0]  # Индекс лучшего

            parent1_idx = tournament()
            # Ограничиваем число попыток, чтобы исключить бесконечный цикл
            parent2_idx = parent1_idx
            for _ in range(50):
                parent2_idx = tournament()
                if parent2_idx != parent1_idx:
                    break
            if parent2_idx == parent1_idx:
                # Фолбэк: выбираем любого другого
                parent2_idx = (parent1_idx + 1) % n
            return self.population[parent1_idx], self.population[parent2_idx]

        elif self.selection_method == "roulette":
            # Рулетка
            min_score = min(self.scores)
            adjusted = [s - min_score + 0.001 for s in self.scores]  # Избегаем отрицательных
            total = sum(adjusted)
            probs = [s / total for s in adjusted]
            parent1 = random.choices(self.population, weights=probs, k=1)[0]
            parent2 = random.choices(self.population, weights=probs, k=1)[0]
            while parent1 == parent2 and len(self.population) > 1:
                parent2 = random.choices(self.population, weights=probs, k=1)[0]
            return parent1, parent2

        else:  # random
            return random.sample(self.population, 2)

    def evolute(self):
        """
        GA цикл.
        """
        print(f"🧬 Старт GA эволюции: pop_size={self.population_size}, generations={self.num_generations}")

        # Первая оценка
        self.evaluate_population()
        best_score = self.scores[0]
        best_prompt = self.population[0]
        best_detailed = self.detailed_scores_per_prompt[0]
        
        # Сохраняем все промты и скоры для поколения 0
        generation_data = {
            "generation": 0,
            "best_score": best_score,
            "mean_score": np.mean(self.scores),
            "best_prompt": best_prompt,
            "best_detailed_scores": best_detailed,
            "population": self.population.copy(),  # Все промты поколения
            "population_scores": self.scores.copy(),  # Все скоры популяции
            "population_detailed_scores": [s.copy() for s in self.detailed_scores_per_prompt]  # Детальные скоры для каждого промта
        }
        self.generation_logs.append(generation_data)
        print(f"Gen 0: best = {best_score:.4f}, mean = {np.mean(self.scores):.4f}")

        for gen in range(1, self.num_generations + 1):
            new_population = []

            # Элитизм: сохраняем топ-2
            new_population.extend(self.population[:2])

            # Генерация потомков
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()

                # Кроссовер
                if random.random() < self.crossover_prob:
                    child1, child2 = my_crossover(parent1, parent2, self.evolution_model, self.config, self.evaluator.fixed_modifiers)
                else:
                    child1, child2 = parent1, parent2

                # Мутация
                child1 = my_mutate(child1, self.mutation_prob, self.evolution_model, self.config)
                child2 = my_mutate(child2, self.mutation_prob, self.evolution_model, self.config)

                # Чиним битый JSON
                child1 = clean_evoprompt_response(child1)
                # Используем лучший промт как шаблон для ремонта
                template_str = self.population[0]
                if self.config:
                    template_genotype = parse_str_to_genotype(template_str, self.evaluator.fixed_modifiers, self.config)
                    child1 = validate_and_repair_genotype(child1, self.evaluator.fixed_modifiers, template_genotype, self.config)
                    # Конвертируем обратно в строку
                    repaired_genotype = parse_str_to_genotype(child1, self.evaluator.fixed_modifiers, self.config)
                    child1 = genotype_to_evoprompt_str(repaired_genotype, self.config)
                
                child2 = clean_evoprompt_response(child2)
                if self.config:
                    child2 = validate_and_repair_genotype(child2, self.evaluator.fixed_modifiers, template_genotype, self.config)
                    repaired_genotype = parse_str_to_genotype(child2, self.evaluator.fixed_modifiers, self.config)
                    child2 = genotype_to_evoprompt_str(repaired_genotype, self.config)

                new_population.extend([child1, child2])

            # Обрезаем до pop_size
            new_population = new_population[:self.population_size]
            self.population = new_population

            # Оценка новой популяции
            self.evaluate_population()

            best_score = self.scores[0]
            best_prompt = self.population[0]
            best_detailed = self.detailed_scores_per_prompt[0]
            
            # Сохраняем все промты и скоры для текущего поколения
            generation_data = {
                "generation": gen,
                "best_score": best_score,
                "mean_score": np.mean(self.scores),
                "best_prompt": best_prompt,
                "best_detailed_scores": best_detailed,
                "population": self.population.copy(),  # Все промты поколения
                "population_scores": self.scores.copy(),  # Все скоры популяции
                "population_detailed_scores": [s.copy() for s in self.detailed_scores_per_prompt]  # Детальные скоры для каждого промта
            }
            self.generation_logs.append(generation_data)
            print(f"Gen {gen}: best = {best_score:.4f}, mean = {np.mean(self.scores):.4f}")

        print(f"🧬 Эволюция завершена. Финальный best_score: {self.scores[0]:.4f}")