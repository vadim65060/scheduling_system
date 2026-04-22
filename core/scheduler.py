"""
Базовый класс для всех планировщиков
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from .job import Job
from .utils import Visualizer


class Scheduler(ABC):
    """Абстрактный базовый класс для всех алгоритмов планирования"""

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None):
        """
        Инициализация планировщика.

        Args:
            jobs: Список заданий
            precedence_constraints: Ограничения предшествования
        """
        self.jobs = {job.id: job for job in jobs}
        self.n = len(jobs)
        self.name = self.__class__.__name__

        # Матрица задержек предшествования
        self.l_matrix = defaultdict(lambda: defaultdict(float))

        if precedence_constraints:
            for (i, j), l_ij in precedence_constraints.items():
                # Убеждаемся, что l_ij ≥ d_i
                job_i = self.jobs[i]
                self.l_matrix[i][j] = max(l_ij, job_i.d_i)

    @abstractmethod
    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу планирования.

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        pass

    # scheduler.py
    def calculate_makespan(self, schedule: List[int], sigma: Optional[Dict[int, Set[int]]] = None) -> Tuple[
        float, Dict[int, float]]:
        """
        ОДНА МАШИНА + DPC + опционально SIGMA:
          - работы выполняются последовательно в порядке `schedule`;
          - учтены r_i и L(i,j) из self.l_matrix;
          - если передана sigma, учитываются добавленные отношения предшествования.
        """
        start_times: Dict[int, float] = {}
        current_time = 0.0

        for j in schedule:
            job = self.jobs[j]
            start = job.r_i
            start = max(start, current_time)

            # DPC из l_matrix
            for i in schedule:
                lij = self.l_matrix.get(i, {}).get(j, 0.0)
                if lij > 0 and i in start_times:
                    required = start_times[i] + lij
                    start = max(start, required)

            # Отношения из sigma (если переданы)
            if sigma:
                for i, next_ids in sigma.items():
                    if j in next_ids and i in start_times:
                        required = start_times[i] + self.jobs[i].d_i
                        start = max(start, required)

            start_times[j] = start
            current_time = start + job.d_i

        if not schedule:
            return 0.0, {}

        C_max = max(
            start_times[j] + self.jobs[j].d_i + self.jobs[j].q_i
            for j in schedule
        )
        return C_max, start_times

    def visualize(self, schedule: List[int], C_max: float, stats, title: str = None):
        """Визуализирует решение"""
        if not schedule:
            Visualizer.print_error("Нет расписания для визуализации")
            return

        _, start_times = self.calculate_makespan(schedule)

        if title is None:
            title = f"РЕЗУЛЬТАТ: {self.name}"

        Visualizer.print_header(title)

        # Основная информация
        Visualizer.print_info("Алгоритм", self.name)
        Visualizer.print_info("C_max", f"{C_max:.2f}")
        Visualizer.print_info("Количество заданий", self.n)
        Visualizer.print_info("Расписание", " → ".join(map(str, schedule)))

        # Гант-диаграма
        print()
        gantt = Visualizer.create_gantt_chart(
            schedule, start_times, self.jobs, C_max,
            title=f"ГАНТ-ДИАГРАММА - {self.name}"
        )
        print(gantt)

        # Статистика
        Visualizer.print_section("СТАТИСТИКА РАСПИСАНИЯ")

        total_processing = sum(job.d_i for job in self.jobs.values())
        total_idle = 0
        current_time = 0

        for job_id in schedule:
            start = start_times[job_id]
            idle_time = max(0, start - current_time)
            total_idle += idle_time
            current_time = start + self.jobs[job_id].d_i

        efficiency = (total_processing / (total_processing + total_idle)) * 100

        Visualizer.print_info("Общее время обработки", f"{total_processing:.2f}")
        Visualizer.print_info("Общее время простоя", f"{total_idle:.2f}")
        Visualizer.print_info("Эффективность машины", f"{efficiency:.1f}%")

        # Критические задания
        critical_jobs = []
        for job_id in schedule:
            completion = start_times[job_id] + self.jobs[job_id].d_i
            delivery_completion = completion + self.jobs[job_id].q_i
            if abs(delivery_completion - C_max) < 1e-6:
                critical_jobs.append(str(job_id))

        if critical_jobs:
            Visualizer.print_info("Критические задания", ", ".join(critical_jobs))

        Visualizer.print_section("СТАТИСТИКА ВЫПОЛНЕНИЯ")
        for key, value in stats.items():
            if key == "execution_time":
                Visualizer.print_info(key, Visualizer.format_time(value))
            elif isinstance(value, float):
                Visualizer.print_info(key, f"{value:.4f}")
            else:
                Visualizer.print_info(key, value)