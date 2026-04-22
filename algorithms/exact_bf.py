"""
Точное решение полным перебором
"""

import itertools
import time
from typing import List, Dict, Tuple, Optional

from core.utils import Timer, ProgressBar
from core.Algorithm import Algorithm


class ExactBruteForce(Algorithm):
    """Точное решение полным перебором"""

    def solve(self, timeout: float = 30.0, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу полным перебором.

        Args:
            timeout: Максимальное время выполнения

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        with Timer() as timer:
            # Если нет заданий, сразу возвращаем
            if self.n == 0:
                stats = {
                    'algorithm': 'ExactBruteForce',
                    'execution_time': 0.0,
                    'total_permutations': 0,
                    'checked_permutations': 0,
                    'valid_permutations': 0,
                    'timeout_reached': False,
                    'C_max': 0
                }
                return [], 0, stats

            best_schedule, best_C, stats = self._brute_force(timeout)

        stats['algorithm'] = 'ExactBruteForce'
        stats['execution_time'] = timer.get_elapsed()

        return best_schedule, best_C, stats

    def _brute_force(self, timeout: float) -> Tuple[Optional[List[int]], float, Dict]:
        """Основной алгоритм полного перебора"""
        job_ids = list(self.jobs.keys())
        total_perms = self._factorial(self.n)

        # Статистика
        stats = {
            'total_permutations': total_perms,
            'checked_permutations': 0,
            'valid_permutations': 0,
            'timeout_reached': False,
            'C_max': float('inf')
        }

        # Проверка сложности
        if total_perms > 10000000:  # 10 миллионов
            print(f"⚠️  ВНИМАНИЕ: {total_perms:,} перестановок!")
            print("   Рекомендуется использовать Branch and Bound")

        # Начальные значения
        best_schedule = None
        best_C = float('inf')
        start_time = time.time()

        # Прогресс-бар
        progress = ProgressBar(total_perms, desc="Полный перебор")

        try:
            for i, permutation in enumerate(itertools.permutations(job_ids)):
                # Проверка таймаута
                if time.time() - start_time > timeout:
                    stats['timeout_reached'] = True
                    break

                # Обновление прогресса
                if i % max(1, total_perms // 100) == 0:
                    progress.update(max(1, total_perms // 100))

                stats['checked_permutations'] += 1

                # Проверка ограничений предшествования
                if not self._is_valid_schedule(permutation):
                    continue

                stats['valid_permutations'] += 1

                # Вычисление makespan
                C, _ = self.calculate_makespan(list(permutation))

                # Обновление лучшего решения
                if C < best_C:
                    best_C = C
                    best_schedule = list(permutation)
                    stats['C_max'] = best_C

                    # Быстрый вывод улучшения
                    print(f"\n🔥 Новый оптимум: C_max = {best_C:.2f}")
                    print(f"   Расписание: {' → '.join(map(str, best_schedule))}")

        except KeyboardInterrupt:
            print("\n\n⏹️  Прервано пользователем")

        progress.finish()
        return best_schedule, best_C, stats

    def _factorial(self, n: int) -> int:
        """Вычисляет факториал"""
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def _is_valid_schedule(self, schedule: List[int]) -> bool:
        """Проверяет, удовлетворяет ли расписание ограничениям предшествования"""
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    try:
                        pos_i = schedule.index(i)
                        pos_j = schedule.index(j)
                        if pos_i > pos_j:
                            return False
                    except ValueError:
                        pass
        return True