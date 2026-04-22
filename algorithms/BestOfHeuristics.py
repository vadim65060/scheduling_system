from typing import List, Dict, Tuple

from algorithms.iltf import ILTF
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from core.Algorithm import Algorithm


class BestOfHeuristics(Algorithm):
    """
    Класс-обёртка, который запускает LTH, MLTH и ILTF
    и выбирает лучшее расписание по C_max.
    """

    def __init__(self, jobs, precedence):
        super().__init__(jobs, precedence)

        # Инициализируем три алгоритма на тех же данных
        self.algorithms = {
            "LTH": LTH(jobs, precedence),
            "MLTH": MLTH(jobs, precedence),
            "ILTF": ILTF(jobs, precedence),
        }

    def solve(self, **kwargs) -> Tuple[List[int], float, Dict]:
        best_schedule = None
        best_C = float("inf")
        best_stats = None
        best_name = None

        for name, algo in self.algorithms.items():
            schedule, C_max, stats = algo.solve(**kwargs)

            if C_max < best_C:
                best_C = C_max
                best_schedule = schedule
                best_stats = stats
                best_name = name

        # Добавляем информацию о том, какой алгоритм победил
        best_stats = dict(best_stats)
        best_stats["best_algorithm"] = best_name

        return best_schedule, best_C, best_stats
