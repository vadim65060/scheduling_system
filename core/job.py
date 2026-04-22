"""
Класс Job для представления задания
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Job:
    """Класс для представления задания"""

    id: int
    r_i: float  # release time (время появления)
    d_i: float  # processing time (время выполнения)
    q_i: float  # delivery time (время доставки)

    def __repr__(self):
        return f"Job({self.id}: r={self.r_i:.1f}, d={self.d_i:.1f}, q={self.q_i:.1f})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Job):
            return False
        return self.id == other.id

    def copy(self) -> 'Job':
        """Создает копию задания"""
        return Job(self.id, self.r_i, self.d_i, self.q_i)