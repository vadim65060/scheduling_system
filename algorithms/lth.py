"""
Longest Tail Heuristic (LTH) - базовый алгоритм
Строгая реализация Algorithm 2 из статьи.
"""

from typing import List, Dict, Tuple, Optional

from core.Algorithm import Algorithm
from core.utils import Timer


class LTH(Algorithm):
    """
    Longest Tail Heuristic (LTH) из статьи.
    Алгоритм 2 - построение расписания по правилу Джексона с учётом DPC.
    """

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу с помощью LTH.

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        with Timer() as timer:
            if self.n == 0:
                stats = {
                    'algorithm': 'LTH',
                    'execution_time': 0.0,
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
        """
        Основной алгоритм LTH (Algorithm 2).

        На каждом шаге выбирает доступную работу с максимальным q_i.
        Если доступных работ нет, ждёт ближайшую по времени готовности.
        Учитывает все DPC (включая с l_ij = 0) при проверке доступности.
        """
        tau = 0.0
        S = []
        C = 0.0

        # Модифицированные времена поступления с учётом DPC
        r_prime = self._initialize_release_times()

        # Структуры предшествования (используют self.dpc_pairs)
        pred = self._get_predecessors()
        succ = self._get_successors()

        while len(S) < self.n:
            # Q = {i not in S | all predecessors of i are in S}
            Q = [j for j in self.jobs if j not in S and pred[j].issubset(S)]

            if not Q:
                break

            # Выбор работы для планирования
            available = [j for j in Q if r_prime[j] <= tau]

            if available:
                # Выбираем работу с максимальным q_i среди доступных
                k = max(available, key=lambda j: self.jobs[j].q_i)
            else:
                # Нет доступных работ - ждём ближайшую
                min_r = min(r_prime[j] for j in Q)
                candidates = [j for j in Q if r_prime[j] == min_r]
                k = max(candidates, key=lambda j: self.jobs[j].q_i)
                # Обновляем текущее время до ближайшей готовности
                tau = min_r

            # Определяем время начала
            s_k = max(tau, r_prime[k])

            # Обновляем модифицированные времена поступления последователей
            for j in succ[k]:
                required_start = s_k + self.l_matrix[k][j]
                if r_prime[j] < required_start:
                    r_prime[j] = required_start

            # Добавляем работу в расписание
            S.append(k)
            tau = s_k + self.jobs[k].d_i

            # Обновляем makespan
            completion_with_delivery = tau + self.jobs[k].q_i
            if completion_with_delivery > C:
                C = completion_with_delivery

        return S, C

    def _initialize_release_times(self) -> Dict[int, float]:
        """
        Инициализация модифицированных времён поступления.
        Учитывает все DPC (включая с l_ij = 0) из self.dpc_pairs.
        Выполняет итеративное обновление до достижения фиксированной точки.
        """
        r_prime = {job_id: job.r_i for job_id, job in self.jobs.items()}

        updated = True
        while updated:
            updated = False

            # Обновление на основе всех DPC-дуг
            for (i, j) in self.dpc_pairs:
                if i in r_prime and j in r_prime:
                    required_start = r_prime[i] + self.l_matrix[i][j]
                    if r_prime[j] < required_start:
                        r_prime[j] = required_start
                        updated = True

        return r_prime