"""
LDepth Algorithm Implementation - Optimized Version
Based on: "An Improved Branch-and-Bound Algorithm for the One-Machine Scheduling Problem
with Delayed Precedence Constraints" (Zhang, Sauppe, Jacobson, 2020)
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import time

from core.Algorithm import Algorithm
from core.job import Job


class LDepthNode:
    """Узел дерева поиска для LDepth алгоритма"""
    __slots__ = ['lower_bound', 'precedence', 'depth', 'parent_id', 'id']

    def __init__(self,
                 lower_bound: float,
                 precedence: Dict[int, Set[int]],
                 depth: int = 0,
                 parent_id: int = None):
        self.lower_bound = lower_bound
        self.precedence = precedence
        self.depth = depth
        self.parent_id = parent_id
        self.id = id(self)

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound


class LDepth(Algorithm):
    """
    LDepth - улучшенный Branch-and-Bound алгоритм для задачи 1|r_i,q_i,dpc|C_max
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 time_limit: float = 60.0,
                 max_iterations: int = 100000):
        super().__init__(jobs, precedence_constraints)
        self.max_time = time_limit
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_makespan = float('inf')
        self.iterations = 0
        self.nodes_explored = 0

        # Предварительно вычисляемые структуры данных
        self._job_ids = list(self.jobs.keys())
        self._n = len(self._job_ids)

        # Кэш для ускорения вычислений
        self._l_matrix_cache = {}
        self._setup_precedence_cache()

    def _setup_precedence_cache(self):
        """Предварительно вычисляет и кэширует отношения предшествования"""
        # Прямые предшественники и последователи из DPC
        self._pi_dpc = defaultdict(set)
        self._sigma_dpc = defaultdict(set)

        for i in self._job_ids:
            for j in self._job_ids:
                if self.l_matrix[i][j] > 0:
                    self._pi_dpc[j].add(i)
                    self._sigma_dpc[i].add(j)
                    self._l_matrix_cache[(i, j)] = self.l_matrix[i][j]

        # Кэш для значений работ
        self._r_i = {j: self.jobs[j].r_i for j in self._job_ids}
        self._d_i = {j: self.jobs[j].d_i for j in self._job_ids}
        self._q_i = {j: self.jobs[j].q_i for j in self._job_ids}

        # Сумма всех времен обработки (константа)
        self._sum_d = sum(self._d_i.values())

        # Минимальный и максимальный хвосты
        self._min_q = min(self._q_i.values())
        self._max_q = max(self._q_i.values())

        # Максимальное r_i + d_i + q_i
        self._max_rdq = max(self._r_i[j] + self._d_i[j] + self._q_i[j] for j in self._job_ids)

        # Максимальное r_i
        self._max_r = max(self._r_i.values())

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """Решает задачу планирования используя LDepth алгоритм."""
        start_time = time.time()

        # Инициализация
        self.best_solution = None
        self.best_makespan = float('inf')
        self.iterations = 0
        self.nodes_explored = 0

        # Корневой узел
        root = LDepthNode(
            lower_bound=self._calculate_lower_bound_fast(),
            precedence={},
            depth=0
        )

        # CBFS структуры данных - динамический список контуров
        contours = [[root]]  # Будет расширяться по мере необходимости
        current_contour = 0
        max_contour = 0

        # Основной цикл
        while True:
            # Проверка ограничений (каждые 100 итераций для производительности)
            if self.iterations % 100 == 0:
                if time.time() - start_time > self.max_time:
                    break

            if self.iterations >= self.max_iterations:
                break

            # Найти непустой контур
            found = False
            start_contour = current_contour

            # Проверяем, есть ли вообще непустые контуры
            if not any(contours):
                break

            while True:
                if current_contour < len(contours) and contours[current_contour]:
                    found = True
                    break
                current_contour += 1
                if current_contour > max_contour:
                    current_contour = 0
                if current_contour == start_contour:
                    break

            if not found:
                break

            # Выбрать лучший узел
            contour = contours[current_contour]
            if not contour:
                current_contour = (current_contour + 1) % (max_contour + 1)
                continue

            best_idx = 0
            best_lb = contour[0].lower_bound
            for i in range(1, len(contour)):
                lb = contour[i].lower_bound
                if lb < best_lb:
                    best_lb = lb
                    best_idx = i

            node = contour.pop(best_idx)

            # Если контур опустел, можно его очистить (опционально)
            # if not contour:
            #     contours[current_contour] = []

            # Отсечение
            if node.lower_bound >= self.best_makespan:
                continue

            self.nodes_explored += 1
            self.iterations += 1

            # Обновить головы
            updated_heads = self._update_heads_fast(node.precedence)

            # Получить расписания
            schedule_lth, makespan_lth = self._lth_fast(node.precedence, updated_heads)
            schedule_mlth, makespan_mlth = self._mlth_fast(node.precedence, updated_heads)

            # Выбрать лучшее
            if makespan_lth <= makespan_mlth:
                schedule = schedule_lth
                makespan = makespan_lth
            else:
                schedule, makespan = self._reschedule_delayed_jobs_fast(
                    schedule_mlth, node.precedence, updated_heads
                )
                if makespan > makespan_lth:
                    schedule = schedule_lth
                    makespan = makespan_lth

            # Обновить лучшее решение
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_solution = schedule

            # Найти критический путь
            critical_path, start_times = self._find_critical_path_fast(schedule)

            if not critical_path:
                continue

            # Ветвление
            if self._can_branch_fast(critical_path, start_times, updated_heads):
                child1_prec, child2_prec = self._strong_branch_fast(
                    critical_path, start_times, node.precedence, updated_heads
                )
            else:
                child1_prec, child2_prec = self._weak_branch_fast(
                    critical_path, node.precedence
                )

            # Создать дочерние узлы
            new_depth = node.depth + 1

            # Расширяем список контуров при необходимости
            while len(contours) <= new_depth:
                contours.append([])

            if child1_prec is not None:
                lb1 = self._calculate_lower_bound_with_precedence_fast(child1_prec)
                if lb1 < self.best_makespan:
                    contours[new_depth].append(
                        LDepthNode(lb1, child1_prec, new_depth, node.id)
                    )
                    max_contour = max(max_contour, new_depth)

            if child2_prec is not None:
                lb2 = self._calculate_lower_bound_with_precedence_fast(child2_prec)
                if lb2 < self.best_makespan:
                    contours[new_depth].append(
                        LDepthNode(lb2, child2_prec, new_depth, node.id)
                    )
                    max_contour = max(max_contour, new_depth)

            # Перейти к следующему контуру
            current_contour += 1
            if current_contour > max_contour:
                current_contour = 0

        execution_time = time.time() - start_time

        stats = {
            "execution_time": execution_time,
            "iterations": self.iterations,
            "nodes_explored": self.nodes_explored,
            "best_makespan": self.best_makespan,
            "contours_used": max_contour + 1
        }

        return self.best_solution, self.best_makespan, stats

    def _update_heads_fast(self, precedence: Dict[int, Set[int]]) -> Dict[int, float]:
        """Быстрое обновление голов (только головы, без хвостов)"""
        heads = {}

        for j in self._job_ids:
            r = self._r_i[j]
            # Проверяем предшественников в порядке DPC
            for i in self._pi_dpc.get(j, ()):
                r = max(r, self._r_i[i] + self._l_matrix_cache[(i, j)])
            # Проверяем переданные предшествования
            for i, followers in precedence.items():
                if j in followers:
                    lij = self._l_matrix_cache.get((i, j), self._d_i[i])
                    r = max(r, self._r_i[i] + lij)
            heads[j] = r

        return heads

    def _lth_fast(self, precedence: Dict[int, Set[int]],
                  heads: Dict[int, float]) -> Tuple[List[int], float]:
        """Оптимизированная версия LTH"""
        schedule = []
        tau = 0.0
        r_prime = heads.copy()
        scheduled = set()

        # Объединяем предшественников
        pi = defaultdict(set)
        for j in self._job_ids:
            pi[j].update(self._pi_dpc.get(j, set()))
        for i, followers in precedence.items():
            for j in followers:
                pi[j].add(i)

        while len(scheduled) < self._n:
            # Найти доступные работы (оптимизированная проверка)
            available = []
            for j in self._job_ids:
                if j not in scheduled and pi[j].issubset(scheduled):
                    available.append(j)

            if not available:
                break

            # Выбор работы (оптимизированный)
            candidates = [j for j in available if r_prime[j] <= tau]

            if candidates:
                k = max(candidates, key=lambda j: self._q_i[j])
            else:
                min_r = min(r_prime[j] for j in available)
                min_r_candidates = [j for j in available if r_prime[j] == min_r]
                k = max(min_r_candidates, key=lambda j: self._q_i[j])

            s_k = max(tau, r_prime[k])
            schedule.append(k)
            scheduled.add(k)
            tau = s_k + self._d_i[k]

            # Обновить головы последователей (только если есть DPC)
            for j in self._sigma_dpc.get(k, ()):
                if j not in scheduled:
                    r_prime[j] = max(r_prime[j], s_k + self._l_matrix_cache[(k, j)])

        if len(schedule) < self._n:
            return schedule, float('inf')

        makespan = self._calculate_makespan_fast(schedule)
        return schedule, makespan

    def _mlth_fast(self, precedence: Dict[int, Set[int]],
                   heads: Dict[int, float]) -> Tuple[List[int], float]:
        """Оптимизированная версия MLTH"""
        schedule = []
        tau = 0.0
        r_prime = heads.copy()
        scheduled = set()

        pi = defaultdict(set)
        for j in self._job_ids:
            pi[j].update(self._pi_dpc.get(j, set()))
        for i, followers in precedence.items():
            for j in followers:
                pi[j].add(i)

        while len(scheduled) < self._n:
            available = []
            for j in self._job_ids:
                if j not in scheduled and pi[j].issubset(scheduled):
                    available.append(j)

            if not available:
                break

            # Быстрый выбор с учетом MLTH логики
            released = [j for j in available if r_prime[j] <= tau]

            if released:
                k = max(released, key=lambda j: self._q_i[j])
            else:
                k = max(available, key=lambda j: (self._q_i[j], -r_prime[j]))

            # Поиск следующей работы для возможного ожидания
            l = None
            next_release = float('inf')
            for j in available:
                if r_prime[j] > tau and r_prime[j] < next_release:
                    next_release = r_prime[j]
                    l = j

            if l is not None and self._q_i[l] > self._q_i[k]:
                tau = max(tau, r_prime[l])
                continue

            s_k = max(tau, r_prime[k])
            schedule.append(k)
            scheduled.add(k)
            tau = s_k + self._d_i[k]

            for j in self._sigma_dpc.get(k, ()):
                if j not in scheduled:
                    r_prime[j] = max(r_prime[j], s_k + self._l_matrix_cache[(k, j)])

        if len(schedule) < self._n:
            return schedule, float('inf')

        makespan = self._calculate_makespan_fast(schedule)
        return schedule, makespan

    def _calculate_makespan_fast(self, schedule: List[int]) -> float:
        """Быстрое вычисление makespan без проверок на циклы"""
        start_times = {}
        current_time = 0.0

        for j in schedule:
            start = max(self._r_i[j], current_time)

            # Учитываем DPC от предыдущих работ
            for i in schedule:
                if i == j:
                    break
                lij = self._l_matrix_cache.get((i, j), 0.0)
                if lij > 0:
                    start = max(start, start_times[i] + lij)

            start_times[j] = start
            current_time = start + self._d_i[j]

        if not schedule:
            return 0.0

        return max(start_times[j] + self._d_i[j] + self._q_i[j] for j in schedule)

    def _find_critical_path_fast(self, schedule: List[int]) -> Tuple[List[int], Dict[int, float]]:
        """Быстрый поиск критического пути"""
        makespan, start_times = self.calculate_makespan(schedule)  # Оставляем для start_times

        if makespan == float('inf'):
            return [], {}

        # Найти последнюю работу в критическом пути
        critical_path = []
        for job_id in schedule:
            completion = start_times[job_id] + self._d_i[job_id] + self._q_i[job_id]
            if abs(completion - makespan) < 1e-6:
                critical_path.append(job_id)
                break

        if not critical_path:
            return [], start_times

        # Построить путь назад (оптимизированно)
        current = critical_path[0]
        while True:
            current_idx = schedule.index(current)
            if current_idx == 0:
                break

            prev_candidate = schedule[current_idx - 1]
            prev_start = start_times[prev_candidate]

            # Проверка связи
            if (prev_start + self._d_i[prev_candidate] >= start_times[current] - 1e-6 or
                self._l_matrix_cache.get((prev_candidate, current), 0) > 0):
                if prev_candidate not in critical_path:
                    critical_path.insert(0, prev_candidate)
                    current = prev_candidate
                else:
                    break
            else:
                break

        return critical_path, start_times

    def _can_branch_fast(self, critical_path: List[int],
                         start_times: Dict[int, float],
                         heads: Dict[int, float]) -> bool:
        """Быстрая проверка возможности сильного ветвления"""
        if len(critical_path) < 2:
            return False

        # Проверить работы не в критическом пути
        cp_set = set(critical_path)

        for job in self._job_ids:
            if job in cp_set:
                continue

            r_prime = heads.get(job, self._r_i[job])

            for i in range(len(critical_path) - 1):
                j1 = critical_path[i]
                j2 = critical_path[i + 1]

                gap_start = start_times[j1] + self._d_i[j1]
                gap_end = start_times[j2]

                if gap_start < gap_end and r_prime < gap_end:
                    return True

        return False

    def _strong_branch_fast(self,
                           critical_path: List[int],
                           start_times: Dict[int, float],
                           precedence: Dict[int, Set[int]],
                           heads: Dict[int, float]) -> Tuple[Optional[Dict[int, Set[int]]],
                                                            Optional[Dict[int, Set[int]]]]:
        """Быстрое сильное ветвление"""
        cp_set = set(critical_path)

        for job in self._job_ids:
            if job in cp_set:
                continue

            r_prime = heads.get(job, self._r_i[job])

            for i in range(len(critical_path) - 1):
                j1 = critical_path[i]
                j2 = critical_path[i + 1]

                gap_start = start_times[j1] + self._d_i[j1]
                gap_end = start_times[j2]

                if gap_start < gap_end and r_prime < gap_end:
                    # Быстрая проверка DPC
                    can_insert = True
                    for k in range(len(self._job_ids)):
                        if self._job_ids[k] == job:
                            break
                        k_id = self._job_ids[k]
                        if (k_id, job) in self._l_matrix_cache:
                            if start_times.get(k_id, 0) + self._l_matrix_cache[(k_id, job)] > gap_end:
                                can_insert = False
                                break

                    if not can_insert:
                        continue

                    # Создать новые отношения предшествования
                    child1_prec = {k: v.copy() for k, v in precedence.items()}
                    child1_prec.setdefault(j1, set()).add(job)

                    child2_prec = {k: v.copy() for k, v in precedence.items()}
                    child2_prec.setdefault(job, set()).add(j1)

                    return child1_prec, child2_prec

        return None, None

    def _weak_branch_fast(self,
                         critical_path: List[int],
                         precedence: Dict[int, Set[int]]) -> Tuple[Optional[Dict[int, Set[int]]],
                                                                  Optional[Dict[int, Set[int]]]]:
        """Быстрое слабое ветвление"""
        if len(critical_path) < 2:
            return None, None

        i = critical_path[0]
        j = critical_path[1]

        child1_prec = {k: v.copy() for k, v in precedence.items()}
        child1_prec.setdefault(i, set()).add(j)

        child2_prec = {k: v.copy() for k, v in precedence.items()}
        child2_prec.setdefault(j, set()).add(i)

        return child1_prec, child2_prec

    def _calculate_lower_bound_fast(self) -> float:
        """Быстрое вычисление нижней границы"""
        return max(self._max_rdq, self._max_r + self._sum_d, self._sum_d + self._min_q)

    def _calculate_lower_bound_with_precedence_fast(self,
                                                    precedence: Dict[int, Set[int]]) -> float:
        """Быстрое вычисление нижней границы с учетом предшествования"""
        # Простая нижняя граница
        lb = self._calculate_lower_bound_fast()

        # Улучшенная оценка с учетом предшествования
        max_completion = 0
        for j in self._job_ids:
            r = self._r_i[j]
            for i, followers in precedence.items():
                if j in followers:
                    lij = self._l_matrix_cache.get((i, j), self._d_i[i])
                    r = max(r, self._r_i[i] + lij)
            completion = r + self._d_i[j] + self._q_i[j]
            max_completion = max(max_completion, completion)

        return max(lb, max_completion)

    def _reschedule_delayed_jobs_fast(self,
                                      schedule: List[int],
                                      precedence: Dict[int, Set[int]],
                                      heads: Dict[int, float]) -> Tuple[List[int], float]:
        """Быстрая процедура решедулинга"""
        max_iterations = len(schedule) ** 2
        current_schedule = schedule.copy()

        for _ in range(max_iterations):
            critical_path, start_times = self._find_critical_path_fast(current_schedule)

            if not critical_path:
                break

            # Поиск отложенной работы
            delayed_job = None
            first_job = critical_path[0]
            first_start = start_times[first_job]

            for job in critical_path:
                r_prime = heads.get(job, self._r_i[job])
                if r_prime < first_start:
                    delayed_job = job
                    break

            if delayed_job is None:
                break

            # Вставка работы
            i1_idx = current_schedule.index(first_job)

            if i1_idx == 0:
                # Вставка в начало
                current_schedule.remove(delayed_job)
                current_schedule.insert(0, delayed_job)
            else:
                # Вставка перед first_job
                current_schedule.remove(delayed_job)
                new_idx = current_schedule.index(first_job)
                current_schedule.insert(new_idx, delayed_job)

        makespan = self._calculate_makespan_fast(current_schedule)
        return current_schedule, makespan

    def get_name(self) -> str:
        return "LDepth-Optimized"