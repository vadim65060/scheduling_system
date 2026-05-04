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
        self.jobs: dict[int, Job] = {job.id: job for job in jobs}
        self.n = len(jobs)
        self.name = self.__class__.__name__

        # Матрица задержек предшествования
        self.l_matrix = defaultdict(lambda: defaultdict(float))

        # Множество всех DPC-дуг, включая с l_ij = 0
        # Это принципиально важно: l_ij = 0 допустимо, когда d_i = 0
        self.dpc_pairs: Set[Tuple[int, int]] = set()

        if precedence_constraints:
            self.dpc_pairs = set(precedence_constraints.keys())
            for (i, j), l_ij in precedence_constraints.items():
                # Убеждаемся, что l_ij >= d_i
                job_i = self.jobs[i]
                self.l_matrix[i][j] = max(l_ij, job_i.d_i)

    def _is_dpc_arc(self, i: int, j: int) -> bool:
        """
        Проверяет, существует ли DPC-дуга от i к j.
        Учитывает как дуги из l_matrix (l_ij > 0), так и дуги с l_ij = 0.
        """
        return (i, j) in self.dpc_pairs or self.l_matrix[i][j] > 0

    def _get_predecessors(self) -> Dict[int, set]:
        """
        Возвращает словарь предшественников: pred[j] = {i: i -> j}
        Учитывает все DPC, включая дуги с l_ij = 0.
        """
        pred = defaultdict(set)

        # Добавляем все DPC из dpc_pairs (включая l_ij = 0)
        for (i, j) in self.dpc_pairs:
            if i in self.jobs and j in self.jobs:
                pred[j].add(i)

        # Добавляем DPC из l_matrix (l_ij > 0)
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0 and (i, j) not in self.dpc_pairs:
                    pred[j].add(i)

        return pred

    def _get_successors(self) -> Dict[int, set]:
        """
        Возвращает словарь последователей: succ[i] = {j: i -> j}
        Учитывает все DPC, включая дуги с l_ij = 0.
        """
        succ = defaultdict(set)

        # Добавляем все DPC из dpc_pairs (включая l_ij = 0)
        for (i, j) in self.dpc_pairs:
            if i in self.jobs and j in self.jobs:
                succ[i].add(j)

        # Добавляем DPC из l_matrix (l_ij > 0)
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0 and (i, j) not in self.dpc_pairs:
                    succ[i].add(j)

        return succ

    def _compute_transitive_dpc(self):
        """
        Вычисляет транзитивное замыкание DPC.
        Использует алгоритм Флойда-Уоршелла для нахождения максимальных путей.
        """
        if self.n == 0:
            return

        job_ids = list(self.jobs.keys())
        n = len(job_ids)
        idx = {jid: i for i, jid in enumerate(job_ids)}

        # Инициализация матрицы расстояний
        dist = [[-float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0

        # Заполняем прямые DPC (все дуги из dpc_pairs)
        for (i, j) in self.dpc_pairs:
            if i in idx and j in idx:
                dist[idx[i]][idx[j]] = max(dist[idx[i]][idx[j]], self.l_matrix[i][j])

        # Также учитываем дуги только из l_matrix
        for i in job_ids:
            for j in job_ids:
                if self.l_matrix[i][j] > 0 and (i, j) not in self.dpc_pairs:
                    dist[idx[i]][idx[j]] = max(dist[idx[i]][idx[j]], self.l_matrix[i][j])

        # Алгоритм Флойда-Уоршелла для максимальных путей
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] > -float('inf') and dist[k][j] > -float('inf'):
                        dist[i][j] = max(dist[i][j], dist[i][k] + dist[k][j])

        # Обновляем l_matrix и dpc_pairs с учетом транзитивности
        for i in job_ids:
            for j in job_ids:
                if dist[idx[i]][idx[j]] > 0:
                    new_lij = max(self.l_matrix[i][j], dist[idx[i]][idx[j]])
                    self.l_matrix[i][j] = new_lij
                    self.dpc_pairs.add((i, j))

        # Проверка на циклы с положительной длиной
        for i in job_ids:
            if self.l_matrix[i][i] > 0:
                print(f"WARNING: Cycle detected with positive length {self.l_matrix[i][i]}")

    def _has_cycle(self, sigma: Optional[Dict[int, Set[int]]] = None) -> bool:
        """
        Проверяет наличие циклов в графе предшествования.

        Args:
            sigma: Дополнительные отношения предшествования

        Returns:
            True если есть цикл, False иначе
        """
        # Строим граф
        graph = defaultdict(list)

        # Добавляем DPC
        for (i, j) in self.dpc_pairs:
            graph[i].append(j)

        # Добавляем sigma
        if sigma:
            for i, next_ids in sigma.items():
                for j in next_ids:
                    graph[i].append(j)

        # DFS для обнаружения циклов
        visited = set()
        rec_stack = set()

        def has_cycle_util(v):
            visited.add(v)
            rec_stack.add(v)

            for neighbor in graph.get(v, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(v)
            return False

        # Создаем копию ключей для безопасной итерации
        for node in list(graph.keys()):
            if node not in visited:
                if has_cycle_util(node):
                    return True

        return False

    @abstractmethod
    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу планирования.

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        pass

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

            # DPC из l_matrix (уже транзитивные)
            for i in schedule:
                if i == j:
                    break
                lij = self.l_matrix.get(i, {}).get(j, 0.0)
                if lij > 0 and i in start_times:
                    required = start_times[i] + lij
                    start = max(start, required)

            # Отношения из sigma (если переданы)
            if sigma:
                for i, next_ids in sigma.items():
                    if j in next_ids and i in start_times:
                        # Используем L(i,j) если есть DPC, иначе d_i
                        lij = self.l_matrix.get(i, {}).get(j, self.jobs[i].d_i)
                        required = start_times[i] + max(lij, self.jobs[i].d_i)
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

    def calculate_makespan_with_transitive(self, schedule: List[int],
                                           sigma: Optional[Dict[int, Set[int]]] = None) -> Tuple[
        float, Dict[int, float]]:
        """
        Версия calculate_makespan, которая сначала вычисляет транзитивное замыкание.
        Используйте этот метод, если не уверены, что l_matrix транзитивна.
        """
        # Сохраняем исходную матрицу
        original_l_matrix = self.l_matrix.copy()

        # Создаем временную матрицу для вычислений
        temp_l_matrix = defaultdict(lambda: defaultdict(float))
        for i in self.jobs:
            for j in self.jobs:
                temp_l_matrix[i][j] = original_l_matrix[i][j]

        # Добавляем DPC из sigma
        if sigma:
            for i, next_ids in sigma.items():
                for j in next_ids:
                    if temp_l_matrix[i][j] < self.jobs[i].d_i:
                        temp_l_matrix[i][j] = self.jobs[i].d_i

        # Вычисляем транзитивное замыкание
        job_ids = list(self.jobs.keys())
        n = len(job_ids)
        idx = {jid: i for i, jid in enumerate(job_ids)}

        dist = [[-float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0

        for i in job_ids:
            for j in job_ids:
                if temp_l_matrix[i][j] > 0:
                    dist[idx[i]][idx[j]] = max(dist[idx[i]][idx[j]], temp_l_matrix[i][j])

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] > -float('inf') and dist[k][j] > -float('inf'):
                        dist[i][j] = max(dist[i][j], dist[i][k] + dist[k][j])

        # Временная функция для получения транзитивной задержки
        def get_transitive_lij(i, j):
            if dist[idx[i]][idx[j]] > 0:
                return dist[idx[i]][idx[j]]
            return 0.0

        # Вычисляем makespan с транзитивными DPC
        start_times = {}
        current_time = 0.0

        for j in schedule:
            job = self.jobs[j]
            start = job.r_i
            start = max(start, current_time)

            for i in schedule:
                if i == j:
                    break
                if i in start_times:
                    lij = get_transitive_lij(i, j)
                    if lij > 0:
                        start = max(start, start_times[i] + lij)

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
        Visualizer.print_info("Расписание", " -> ".join(map(str, schedule)))

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
