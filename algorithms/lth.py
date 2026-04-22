"""
Longest Tail Heuristic (LTH) - базовый алгоритм
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from core.utils import Timer
from core.Algorithm import Algorithm


class LTH(Algorithm):
    """
    Longest Tail Heuristic (LTH) из статьи
    Алгоритм 2
    """

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу с помощью LTH.

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        with Timer() as timer:
            # Если нет заданий, сразу возвращаем
            if self.n == 0:
                stats = {
                    'algorithm': 'LTH',
                    'execution_time': 0.0,  # Явно указываем 0
                    'iterations': 0,
                    'C_max': 0
                }
                return [], 0, stats

            schedule, C_max = self._lth_algorithm()

        stats = {
            'algorithm': 'LTH',
            'execution_time': timer.get_elapsed(),
            'iterations': self.n,
            'C_max': C_max
        }

        return schedule, C_max, stats

    def _lth_algorithm(self) -> Tuple[List[int], float]:
        """Основной алгоритм LTH"""
        # Инициализация
        τ = 0  # current time
        S = []  # scheduled jobs
        C = 0  # makespan
        r_prime = self._initialize_release_times()

        # Структуры предшествования
        π = self._get_predecessors()
        σ = self._get_successors()

        while len(S) < self.n:
            # Q = {i ∉ S | π(i) ⊆ S}
            Q = [j for j in self.jobs if j not in S and π[j].issubset(S)]

            if not Q:
                break

            # Выбор задачи для планирования
            available = [j for j in Q if r_prime[j] <= τ]

            if available:
                # Выбираем задачу с максимальным q_i среди доступных
                k = max(available, key=lambda j: self.jobs[j].q_i)
            else:
                # Нет доступных задач, выбираем с минимальным r_prime
                min_r = min(r_prime[j] for j in Q)
                candidates = [j for j in Q if r_prime[j] == min_r]
                k = max(candidates, key=lambda j: self.jobs[j].q_i)

            # Определяем время начала
            s_k = max(τ, r_prime[k])

            # Обновляем release times последователей
            for j in σ[k]:
                required_start = s_k + self.l_matrix[k][j]
                if r_prime[j] < required_start:
                    r_prime[j] = required_start

            # Добавляем в расписание
            S.append(k)
            τ = s_k + self.jobs[k].d_i

            # Обновляем makespan
            completion_with_delivery = τ + self.jobs[k].q_i
            if completion_with_delivery > C:
                C = completion_with_delivery

        return S, C

    def _initialize_release_times(self) -> Dict[int, float]:
        """Инициализация release times с учетом предшествования"""
        r_prime = {job_id: job.r_i for job_id, job in self.jobs.items()}

        # Обновляем на основе предшествования
        updated = True
        while updated:
            updated = False
            for i in self.jobs:
                for j in self.jobs:
                    if self.l_matrix[i][j] > 0:
                        required_start = r_prime[i] + self.l_matrix[i][j]
                        if r_prime[j] < required_start:
                            r_prime[j] = required_start
                            updated = True

        return r_prime

    def _get_predecessors(self) -> Dict[int, set]:
        """Возвращает словарь предшественников"""
        π = defaultdict(set)
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    π[j].add(i)
        return π

    def _get_successors(self) -> Dict[int, set]:
        """Возвращает словарь последователей"""
        σ = defaultdict(set)
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    σ[i].add(j)
        return σ