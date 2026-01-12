import time
from external.evoprompt.evaluator import Evaluator  # Base из EvoPrompt
from src.utils.time import format_time
from src.utils.save_result import save_log
from src.utils.personality_match import fitness_function
from .utils import parse_str_to_genotype

class MyEvaluator(Evaluator):
    def __init__(self, args, task, model, fixed_modifiers):
        super().__init__(args)
        self.task = task  # Из person_type_opt (вопросы, формат)
        self.model = model  # LLM для generate
        self.fixed_modifiers = fixed_modifiers  # Из system
        self.dev_participants = []  # Устанавливаем позже

    def forward(self, prompt_str, config, total_participants, cluster, cluster_start_time, cluster_log, experiment_log, result_cluster, results_dir):
        """
        Вычисляет скалярный фитнес: усреднённый по dev_participants.
        prompt_str — строка-шаблон genotype без modifiers.
        """
        genotype = parse_str_to_genotype(prompt_str, self.fixed_modifiers)
        test_participants_scores = []
        iteration_times = []
        for idx, (index, participant) in enumerate(self.dev_participants.iterrows(), 1):
            iteration_start_time = time.time()
            # ГЛАВНАЯ часть цикла: обращение к модели
            score = fitness_function(participant, genotype, self.task, self.model)
            test_participants_scores.append(score)

            iteration_time = time.time() - iteration_start_time
            iteration_times.append(iteration_time)
            
            # Вычисление времени для кластера
            cluster_elapsed = time.time() - cluster_start_time
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            remaining_iterations = total_participants - idx
            eta = avg_iteration_time * remaining_iterations

            participant_result = {
                'participant_id': int(index),
                'similarity': score['similarity'],
                'avg_diff': score['avg_diff'],
                'pearson_corr': score['pearson_corr'],
                'iteration_time': iteration_time
            }
            cluster_log['participants'].append(participant_result)
            # Сохраняем промежуточный лог каждые N участников
            if idx % config['data']['save_every_n'] == 0:
                experiment_log['clusters'][str(cluster)] = cluster_log
                save_log(experiment_log, results_dir, "experiment_log.json")
            
            print(f"📊 Результаты оценки соответствия:")
            print(f"- Схожесть (similarity): {score['similarity']:.4f}")
            print(f"- Средняя разница (avg_diff): {score['avg_diff']:.4f}")
            print(f"- Корреляция Пирсона (pearson_corr): {score['pearson_corr'][0]:.4f}" if isinstance(score['pearson_corr'], tuple) else f"   • Корреляция Пирсона (pearson_corr): {score['pearson_corr']:.4f}")
            print(f"{'='*70}\n")

            print(f"⏱️  Время: прошедшее {format_time(cluster_elapsed)} | "
                    f"итерация {format_time(iteration_time)} | "
                    f"ETA {format_time(eta)}")

        mean_similarity_participants = sum([i['similarity'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_avg_diff_participants = sum([i['avg_diff'] for i in test_participants_scores]) / len(test_participants_scores)
        mean_pearson_corr_participants = sum([i['pearson_corr'][0] if isinstance(i['pearson_corr'], tuple) else i['pearson_corr'] for i in test_participants_scores]) / len(test_participants_scores)

        # Финальное время кластера
        cluster_total_time = time.time() - cluster_start_time
        avg_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
        
        print(f"\n{'='*70}")
        print(f"📈 ИТОГОВЫЕ СРЕДНИЕ ПОКАЗАТЕЛИ КЛАСТЕРА {cluster}")
        print(f"{'='*70}")
        print(f"- Средняя схожесть (similarity): {mean_similarity_participants:.4f}")
        print(f"- Средняя разница (avg_diff): {mean_avg_diff_participants:.4f}")
        print(f"- Средняя корреляция Пирсона (pearson_corr): {mean_pearson_corr_participants:.4f}")
        print(f"{'='*70}")
        print(f"⏱️  Статистика времени кластера:")
        print(f"- Общее время: {format_time(cluster_total_time)}")
        print(f"- Среднее время на участника: {format_time(avg_iteration_time)}")
        print(f"- Всего обработано участников: {total_participants}")
        print(f"{'='*70}")
        print(f"✅ Обработка кластера {cluster} завершена\n")
        
        cluster_log['end_time'] = time.time()
        cluster_log['total_time'] = cluster_total_time
        cluster_log['genotype'] = genotype
        cluster_log['task'] = self.task 
        cluster_log['summary'] = {
            'mean_similarity': mean_similarity_participants,
            'mean_avg_diff': mean_avg_diff_participants,
            'mean_pearson_corr': mean_pearson_corr_participants,
            'total_participants': total_participants
        }
        cluster_log.pop('participants')
        result_cluster['clusters'][str(cluster)]= cluster_log
        save_log(result_cluster, results_dir, "result_log.json")
        return mean_similarity_participants
