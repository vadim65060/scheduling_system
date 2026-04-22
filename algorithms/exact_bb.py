"""
Точное решение методом ветвей и границ
"""

import time
from typing import List, Dict, Tuple, Optional, Set

from core.utils import Timer
from core.Algorithm import Algorithm

debug_print = False

class ExactBranchAndBound(Algorithm):
    """Точное решение методом ветвей и границ"""

    def solve(self, timeout: float = 60.0, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу методом ветвей и границ.

        Args:
            timeout: Максимальное время выполнения

        Returns:
            Кортеж (расписание, makespan, статистика)
        """

        with Timer() as timer:
            # Если нет заданий, сразу возвращаем
            if self.n == 0:
                stats = {
                    'algorithm': 'ExactBranchAndBound',
                    'execution_time': 0.0,
                    'nodes_explored': 0,
                    'pruned_by_bound': 0,
                    'timeout_reached': False,
                    'C_max': 0
                }
                return [], 0, stats

            best_schedule, best_C, stats = self._branch_and_bound(timeout)

        stats['algorithm'] = 'ExactBranchAndBound'
        stats['execution_time'] = timer.get_elapsed()

        return best_schedule, best_C, stats

    def _branch_and_bound(self, timeout: float) -> Tuple[Optional[List[int]], float, Dict]:
        """Основной алгоритм ветвей и границ"""
        # Статистика
        stats = {
            'nodes_explored': 0,
            'pruned_by_bound': 0,
            'timeout_reached': False,
            'C_max': float('inf')
        }

        # Начальные значения
        best_schedule = None
        best_C = float('inf')
        start_time = time.time()

        # Стек для DFS
        stack = []
        initial_node = {
            'schedule': [],
            'remaining': set(self.jobs.keys()),
            'lower_bound': self._calculate_lower_bound([], set(self.jobs.keys())),
            'current_time': 0
        }
        stack.append(initial_node)

        while stack:
            # Проверка таймаута
            if time.time() - start_time > timeout:
                stats['timeout_reached'] = True
                break

            # Извлекаем узел
            node = stack.pop()
            stats['nodes_explored'] += 1

            # Периодический вывод прогресса
            if debug_print and stats['nodes_explored'] % 10000 == 0:
                print(f"\rУзлов: {stats['nodes_explored']:,}, "
                      f"Лучший C_max: {best_C:.2f}", end="")

            # Если все задания запланированы
            if not node['remaining']:
                C, _ = self.calculate_makespan(node['schedule'])

                if C < best_C:
                    best_C = C
                    best_schedule = node['schedule'].copy()
                    stats['C_max'] = best_C
                    if debug_print:
                        print(f"\n🔥 Новый оптимум: C_max = {best_C:.2f}")
                        print(f"   Расписание: {' → '.join(map(str, best_schedule))}")

                continue

            # Отсечение по нижней границе
            if node['lower_bound'] >= best_C:
                stats['pruned_by_bound'] += 1
                continue

            # Генерация дочерних узлов
            for job_id in node['remaining']:
                # Проверка возможности добавления
                if not self._can_add_job(job_id, node['schedule']):
                    continue

                # Создание нового узла
                new_schedule = node['schedule'] + [job_id]
                new_remaining = node['remaining'].copy()
                new_remaining.remove(job_id)

                # Вычисление времени начала
                job = self.jobs[job_id]
                new_current_time = max(node['current_time'], job.r_i)

                # Учет ограничений предшествования
                for prev_id in new_schedule[:-1]:
                    if self.l_matrix[prev_id][job_id] > 0:
                        prev_completion = self._get_completion_time(
                            new_schedule, prev_id, node['current_time']
                        )
                        required_start = prev_completion + self.l_matrix[prev_id][job_id]
                        new_current_time = max(new_current_time, required_start)

                # Вычисление нижней границы
                lower_bound = self._calculate_lower_bound(
                    new_schedule, new_remaining, new_current_time
                )

                # Создание нового узла
                new_node = {
                    'schedule': new_schedule,
                    'remaining': new_remaining,
                    'lower_bound': lower_bound,
                    'current_time': new_current_time + job.d_i
                }

                stack.append(new_node)
        if debug_print:
            print(f"\rУзлов исследовано: {stats['nodes_explored']:,}                    ")

        return best_schedule, best_C, stats

    def _can_add_job(self, job_id: int, partial_schedule: List[int]) -> bool:
        """Проверяет, можно ли добавить задание"""
        for pred in self.jobs:
            if self.l_matrix[pred][job_id] > 0 and pred not in partial_schedule:
                return False
        return True

    def _calculate_lower_bound(self, partial_schedule: List[int],
                               remaining: Set[int],
                               current_time: float = 0) -> float:
        """Вычисляет нижнюю границу"""
        if not remaining:
            C, _ = self.calculate_makespan(partial_schedule)
            return C

        min_processing = sum(self.jobs[j].d_i for j in remaining)
        min_delivery = min((self.jobs[j].q_i for j in remaining), default=0)

        return current_time + min_processing + min_delivery

    def _get_completion_time(self, schedule: List[int], job_id: int,
                             start_time: float = 0) -> float:
        """Вычисляет время завершения задания"""
        current_time = start_time
        for j in schedule:
            job = self.jobs[j]
            s_j = max(current_time, job.r_i)

            if j == job_id:
                return s_j + job.d_i

            current_time = s_j + job.d_i

        return current_time