"""
Базовый класс для алгоритмов
"""

from abc import abstractmethod
from typing import List, Dict, Tuple, Optional
from core.scheduler import Scheduler
from core.job import Job


class Algorithm(Scheduler):
    """Базовый класс для всех алгоритмов"""

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None):
        super().__init__(jobs, precedence_constraints)

    @abstractmethod
    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу планирования.

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        pass

    def get_name(self) -> str:
        """Возвращает имя алгоритма"""
        return self.name