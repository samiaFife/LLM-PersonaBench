import random

import numpy as np

from src.evolution.operators import my_crossover, my_mutate
from src.evolution.utils import (
    clean_evoprompt_response,
    genotype_to_evoprompt_str,
    parse_str_to_genotype,
    validate_and_repair_genotype,
)
from src.utils.time import TimeEstimator, format_time


class Evoluter:
    """
    Базовый класс для эволюции (адаптировано из EvoPrompt).
    """

    def __init__(self, args, evaluator, evolution_model=None, config=None):
        self.args = args
        self.evaluator = evaluator
        self.evolution_model = evolution_model
        self.config = config
        self.population = []
        self.scores = []
        self.generation_logs = []

    def evolute(self):
        raise NotImplementedError("Реализуется в подклассах")


class GAEvoluter(Evoluter):
    """
    Genetic Algorithm эволюция (адаптировано под нашу задачу).
    """

    def __init__(self, args, evaluator, evolution_model=None, config=None):
        super().__init__(args, evaluator, evolution_model, config)
        self.population_size = args.popsize
        self.num_generations = args.budget
        self.mutation_prob = args.mutation_prob
        self.crossover_prob = args.crossover_prob
        self.selection_method = args.sel_mode.lower()
        self.detailed_scores_per_prompt = []

    def evaluate_population(self):
        """
        Оценивает всю популяцию и сохраняет детальные stage-метрики
        для каждого промпта.
        """
        self.scores = []
        self.detailed_scores_per_prompt = []

        for prompt_str in self.population:
            raw_score = self.evaluator.forward(prompt_str, self.config)
            score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
            self.scores.append(score)

            if hasattr(self.evaluator, "last_detailed_scores"):
                self.detailed_scores_per_prompt.append(self.evaluator.last_detailed_scores.copy())
            else:
                self.detailed_scores_per_prompt.append(
                    {
                        "prompt": prompt_str,
                        "stage_metrics": {
                            "summary": {
                                "mean_similarity": score,
                                "mean_avg_diff": 0.0,
                                "mean_pearson_corr": 0.0,
                                "mean_mae_35": 0.0,
                                "mean_similarity_35": 0.0,
                                "mean_pearson_35": 0.0,
                                "mean_similarity_facets": 0.0,
                                "mean_similarity_traits": 0.0,
                            },
                            "trait_similarity": {},
                            "facet_similarity": {},
                            "answer_block_similarity": {},
                            "selected_facets": [],
                            "trait_question_blocks": {},
                        },
                    }
                )

        sorted_idx = np.argsort(self.scores)[::-1]
        self.population = [self.population[i] for i in sorted_idx]
        self.scores = [self.scores[i] for i in sorted_idx]
        self.detailed_scores_per_prompt = [self.detailed_scores_per_prompt[i] for i in sorted_idx]

    def _append_generation_log(self, generation: int):
        best_score = float(self.scores[0]) if self.scores else 0.0
        mean_score = float(np.mean(self.scores)) if self.scores else 0.0
        best_detailed = self.detailed_scores_per_prompt[0] if self.detailed_scores_per_prompt else {}

        self.generation_logs.append(
            {
                "generation": generation,
                "best_score": best_score,
                "mean_score": mean_score,
                "best_prompt": best_detailed.get("prompt") or (self.population[0] if self.population else ""),
                "best_stage_summary": best_detailed.get("stage_metrics") or {},
            }
        )

    def select_parents(self) -> tuple:
        if self.selection_method == "tournament":
            n = len(self.population)
            if n <= 1:
                return self.population[0], self.population[0]
            if n == 2:
                return self.population[0], self.population[1]

            tournament_size = min(3, n - 1)

            def tournament():
                candidates = random.sample(list(enumerate(self.scores)), tournament_size)
                return max(candidates, key=lambda x: x[1])[0]

            parent1_idx = tournament()
            parent2_idx = parent1_idx
            for _ in range(50):
                parent2_idx = tournament()
                if parent2_idx != parent1_idx:
                    break
            if parent2_idx == parent1_idx:
                parent2_idx = (parent1_idx + 1) % n
            return self.population[parent1_idx], self.population[parent2_idx]

        if self.selection_method == "roulette":
            min_score = min(self.scores)
            adjusted = [s - min_score + 0.001 for s in self.scores]
            total = sum(adjusted)
            probs = [s / total for s in adjusted]
            parent1 = random.choices(self.population, weights=probs, k=1)[0]
            parent2 = random.choices(self.population, weights=probs, k=1)[0]
            while parent1 == parent2 and len(self.population) > 1:
                parent2 = random.choices(self.population, weights=probs, k=1)[0]
            return parent1, parent2

        return random.sample(self.population, 2)

    def evolute(self):
        print(f"🧬 Старт GA эволюции: pop_size={self.population_size}, generations={self.num_generations}")

        time_estimator = TimeEstimator(total_items=self.num_generations)
        time_estimator.start()
        time_estimator.start_item()

        self.evaluate_population()
        time_estimator.finish_item()
        self._append_generation_log(generation=0)

        progress_info = time_estimator.get_progress_info(completed_items=1)
        print(f"Gen 0: best = {self.scores[0]:.4f}, mean = {np.mean(self.scores):.4f} | {progress_info}")

        for gen in range(1, self.num_generations):
            time_estimator.start_item()
            new_population = []
            new_population.extend(self.population[:2])

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()

                if random.random() < self.crossover_prob:
                    child1, child2 = my_crossover(
                        parent1,
                        parent2,
                        self.evolution_model,
                        self.config,
                        self.evaluator.fixed_modifiers,
                    )
                else:
                    child1, child2 = parent1, parent2

                child1 = my_mutate(child1, self.mutation_prob, self.evolution_model, self.config)
                child2 = my_mutate(child2, self.mutation_prob, self.evolution_model, self.config)

                child1 = clean_evoprompt_response(child1)
                template_str = self.population[0]
                if self.config:
                    template_genotype = parse_str_to_genotype(template_str, self.evaluator.fixed_modifiers, self.config)
                    child1 = validate_and_repair_genotype(child1, self.evaluator.fixed_modifiers, template_genotype, self.config)
                    repaired_genotype = parse_str_to_genotype(child1, self.evaluator.fixed_modifiers, self.config)
                    child1 = genotype_to_evoprompt_str(repaired_genotype, self.config)

                child2 = clean_evoprompt_response(child2)
                if self.config:
                    child2 = validate_and_repair_genotype(child2, self.evaluator.fixed_modifiers, template_genotype, self.config)
                    repaired_genotype = parse_str_to_genotype(child2, self.evaluator.fixed_modifiers, self.config)
                    child2 = genotype_to_evoprompt_str(repaired_genotype, self.config)

                new_population.extend([child1, child2])

            self.population = new_population[: self.population_size]
            self.evaluate_population()
            time_estimator.finish_item()
            self._append_generation_log(generation=gen)

            progress_info = time_estimator.get_progress_info(completed_items=gen + 1)
            print(f"Gen {gen}: best = {self.scores[0]:.4f}, mean = {np.mean(self.scores):.4f} | {progress_info}")

        total_evolution_time = time_estimator.get_elapsed()
        print(f"🧬 Эволюция завершена. Финальный best_score: {self.scores[0]:.4f} | Общее время: {format_time(total_evolution_time)}")
