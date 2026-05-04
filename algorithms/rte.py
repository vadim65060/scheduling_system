import copy
from typing import List, Dict, Tuple, Optional, Type, Any

from core.Algorithm import Algorithm
from core.job import Job
from core.utils import Timer


class RTE(Algorithm):
    """
    RelaxationThenEnforce
    Эвристический алгоритм, который:
      1. Строит расписание с помощью переданной эвристики (без учета DPC)
      2. Вычисляет C_max с учетом DPC через calculate_makespan

    Это позволяет оценить, насколько хорошо эвристика работает
    в присутствии отложенных ограничений предшествования.
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 heuristic_class: Optional[Type[Algorithm]] = None,
                 heuristic_kwargs: Optional[Dict[str, Any]] = None):
        """
        Args:
            jobs: Список заданий
            precedence_constraints: Ограничения предшествования (DPC)
            heuristic_class: Класс эвристики для построения расписания.
                            Если None, используется BestOfHeuristics
            heuristic_kwargs: Дополнительные аргументы для эвристики
        """
        super().__init__(jobs, precedence_constraints)

        # Импортируем эвристику по умолчанию, если не указана
        if heuristic_class is None:
            from algorithms.BestOfHeuristics import BestOfHeuristics
            self.heuristic_class = BestOfHeuristics
        else:
            self.heuristic_class = heuristic_class

        self.heuristic_kwargs = heuristic_kwargs or {}
        self.name = f"RTE ({self.heuristic_class.__name__})"

        self.precedence_constraints_noDPC = copy.deepcopy(precedence_constraints)

        for (i, j), l_ij in precedence_constraints.items():
            self.precedence_constraints_noDPC[(i, j)] = self.jobs[i].d_i

        # Результаты
        self.heuristic_schedule: Optional[List[int]] = None
        self.heuristic_makespan: float = float('inf')
        self.final_makespan: float = float('inf')

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу:
          1. Запускает эвристику без учета DPC
          2. Вычисляет C_max с учетом DPC

        Returns:
            Кортеж (расписание, C_max, статистика)
        """
        with Timer() as timer:
            # 1. Создаем эвристику БЕЗ DPC
            heuristic = self.heuristic_class(
                jobs=list(self.jobs.values()),
                precedence_constraints=self.precedence_constraints_noDPC,
                **self.heuristic_kwargs
            )

            # 2. Получаем расписание от эвристики
            self.heuristic_schedule, self.heuristic_makespan, heuristic_stats = heuristic.solve(**kwargs)

            if self.heuristic_schedule is None:
                return None, float('inf'), {
                    'algorithm': self.name,
                    'execution_time': timer.get_elapsed(),
                    'error': 'Эвристика не вернула расписание'
                }

            # 3. Вычисляем C_max с учетом DPC
            self.final_makespan, start_times = self.calculate_makespan(self.heuristic_schedule)

        stats = {
            'algorithm': self.name,
            'execution_time': timer.get_elapsed(),
            'heuristic_used': self.heuristic_class.__name__,
            'heuristic_makespan': self.heuristic_makespan,
            'difference': self.final_makespan - self.heuristic_makespan,
        }

        # Добавляем статистику эвристики, если она есть
        if heuristic_stats:
            for key, value in heuristic_stats.items():
                if key not in stats:
                    stats[f'heuristic_{key}'] = value

        return self.heuristic_schedule, self.final_makespan, stats