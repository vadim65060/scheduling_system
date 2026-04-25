"""
LDepth Algorithm Implementation
Based on: "An Improved Branch-and-Bound Algorithm for the One-Machine Scheduling Problem
with Delayed Precedence Constraints" (Zhang, Sauppe, Jacobson, 2020)
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from core.Algorithm import Algorithm
from core.job import Job


class LDepthNode:
    """Узел дерева поиска для LDepth алгоритма"""

    def __init__(self,
                 lower_bound: float,
                 precedence: Dict[int, Set[int]],
                 depth: int = 0,
                 parent_id: int = None):
        self.lower_bound = lower_bound
        self.precedence = precedence  # pi(i) - predecessors
        self.depth = depth
        self.parent_id = parent_id
        self.id = id(self)

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound


class LDepth(Algorithm):
    """
    LDepth - улучшенный Branch-and-Bound алгоритм для задачи 1|r_i,q_i,dpc|C_max

    Особенности:
    1. MLTH (Modified Longest Tail Heuristic) для генерации расписаний
    2. CBFS (Cyclic Best-First Search) стратегия поиска
    3. Процедура решедулинга для отложенных работ
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

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Решает задачу планирования используя LDepth алгоритм.

        Returns:
            Кортеж (расписание, makespan, статистика)
        """
        import time
        start_time = time.time()

        # Инициализация
        self.best_solution = None
        self.best_makespan = float('inf')
        self.iterations = 0
        self.nodes_explored = 0

        # Корневой узел
        root = LDepthNode(
            lower_bound=self._calculate_lower_bound(),
            precedence=defaultdict(set),
            depth=0
        )

        # CBFS структуры данных - контуры по глубине
        contours = defaultdict(list)
        current_contour = 0
        contours[0].append(root)
        max_contour = 0

        # Основной цикл
        while True:
            # Проверка ограничений
            if time.time() - start_time > self.max_time:
                break
            if self.iterations >= self.max_iterations:
                break

            # Найти непустой контур
            found = False
            for _ in range(max_contour + 1):
                if contours[current_contour]:
                    found = True
                    break
                current_contour = (current_contour + 1) % (max_contour + 1)

            if not found:
                break

            # Выбрать лучший узел из текущего контура (BFS с минимальной нижней границей)
            best_idx = 0
            best_lb = float('inf')
            for i, node in enumerate(contours[current_contour]):
                if node.lower_bound < best_lb:
                    best_lb = node.lower_bound
                    best_idx = i

            node = contours[current_contour].pop(best_idx)

            # Отсечение
            if node.lower_bound >= self.best_makespan:
                current_contour = (current_contour + 1) % (max_contour + 1)
                continue

            self.nodes_explored += 1
            self.iterations += 1

            # Обновить головы и хвосты
            updated_heads, updated_tails = self._update_heads_and_tails(node.precedence)

            # Получить расписание используя LTH и MLTH
            schedule_lth, makespan_lth = self._lth(node.precedence, updated_heads)
            schedule_mlth, makespan_mlth = self._mlth(node.precedence, updated_heads)

            # Выбрать лучшее расписание
            if makespan_lth <= makespan_mlth:
                schedule = schedule_lth
                makespan = makespan_lth
            else:
                # Применить процедуру решедулинга для MLTH
                schedule, makespan = self._reschedule_delayed_jobs(
                    schedule_mlth, node.precedence, updated_heads
                )
                # Сравнить с LTH снова
                if makespan > makespan_lth:
                    schedule = schedule_lth
                    makespan = makespan_lth

            # Обновить лучшее решение
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_solution = schedule.copy()

            # Найти критический путь
            critical_path = self._find_critical_path(schedule, updated_heads)

            if not critical_path:
                # Нет критического пути - оптимальное решение найдено
                current_contour = (current_contour + 1) % (max_contour + 1)
                continue

            # Проверить условие ветвления
            if self._can_branch(critical_path, schedule, updated_heads):
                # Сильное ветвление
                child1_prec, child2_prec = self._strong_branch(
                    critical_path, schedule, node.precedence
                )
            else:
                # Слабое ветвление
                child1_prec, child2_prec = self._weak_branch(
                    critical_path, schedule, node.precedence
                )

            # Создать дочерние узлы
            if child1_prec is not None:
                child1 = LDepthNode(
                    lower_bound=self._calculate_lower_bound_with_precedence(child1_prec),
                    precedence=child1_prec,
                    depth=node.depth + 1,
                    parent_id=node.id
                )
                contours[node.depth + 1].append(child1)
                max_contour = max(max_contour, node.depth + 1)

            if child2_prec is not None:
                child2 = LDepthNode(
                    lower_bound=self._calculate_lower_bound_with_precedence(child2_prec),
                    precedence=child2_prec,
                    depth=node.depth + 1,
                    parent_id=node.id
                )
                contours[node.depth + 1].append(child2)
                max_contour = max(max_contour, node.depth + 1)

            # Перейти к следующему контуру
            current_contour = (current_contour + 1) % (max_contour + 1)

        execution_time = time.time() - start_time

        stats = {
            "execution_time": execution_time,
            "iterations": self.iterations,
            "nodes_explored": self.nodes_explored,
            "best_makespan": self.best_makespan,
            "contours_used": max_contour + 1 if contours else 0
        }

        return self.best_solution, self.best_makespan, stats

    def _lth(self,
             precedence: Dict[int, Set[int]],
             heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Longest Tail Heuristic (LTH) - оригинальная эвристика из статьи.

        На каждом шаге выбирает доступную работу с максимальным q_i.
        """
        n = len(self.jobs)
        schedule = []
        tau = 0.0  # текущее время
        r_prime = heads.copy()  # обновленные головы с учетом предшественников

        # Множества предшественников и последователей
        pi = defaultdict(set)
        sigma = defaultdict(set)

        # Построить отношения предшествования из DPC
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    pi[j].add(i)
                    sigma[i].add(j)

        # Добавить переданные отношения предшествования
        for i in precedence:
            for j in precedence[i]:
                pi[j].add(i)
                sigma[i].add(j)

        scheduled = set()

        while len(scheduled) < n:
            # Найти доступные работы (все предшественники запланированы)
            available = set()
            for j in self.jobs:
                if j not in scheduled:
                    if pi[j].issubset(scheduled):
                        available.add(j)

            # Выбрать работу с максимальным хвостом среди доступных
            if not available:
                break

            # Найти работу с минимальным r'_i среди доступных
            min_r = min(r_prime[j] for j in available)

            # Выбрать работу: если есть доступная с r'_i <= tau, взять с макс q_i
            # Иначе взять работу с макс q_i среди имеющих минимальное r'_i
            candidates = [j for j in available if r_prime[j] <= tau]

            if candidates:
                # Выбрать работу с максимальным хвостом
                k = max(candidates, key=lambda j: self.jobs[j].q_i)
            else:
                # Выбрать работу с макс хвостом среди имеющих мин голову
                min_r_candidates = [j for j in available if r_prime[j] == min_r]
                k = max(min_r_candidates, key=lambda j: self.jobs[j].q_i)

            # Запланировать работу
            s_k = max(tau, r_prime[k])
            schedule.append(k)
            scheduled.add(k)
            tau = s_k + self.jobs[k].d_i

            # Обновить головы последователей
            for j in sigma[k]:
                if j not in scheduled:
                    r_prime[j] = max(r_prime[j], s_k + self.l_matrix[k][j])

        # Вычислить makespan
        if len(schedule) < n:
            return schedule, float('inf')

        makespan, _ = self.calculate_makespan(schedule,
                                              {i: set(precedence[i]) for i in precedence})
        return schedule, makespan

    def _mlth(self,
              precedence: Dict[int, Set[int]],
              heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Modified Longest Tail Heuristic (MLTH) - модифицированная эвристика.

        Отличие от LTH: может пропускать время вперед, чтобы включить
        недоступные работы с большими хвостами.
        """
        n = len(self.jobs)
        schedule = []
        tau = 0.0
        r_prime = heads.copy()

        # Построить отношения предшествования
        pi = defaultdict(set)
        sigma = defaultdict(set)

        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    pi[j].add(i)
                    sigma[i].add(j)

        for i in precedence:
            for j in precedence[i]:
                pi[j].add(i)
                sigma[i].add(j)

        scheduled = set()

        while len(scheduled) < n:
            # Найти доступные работы
            available = set()
            for j in self.jobs:
                if j not in scheduled:
                    if pi[j].issubset(scheduled):
                        available.add(j)

            if not available:
                break

            # Работа с максимальным хвостом среди уже доступных
            released = [j for j in available if r_prime[j] <= tau]

            if released:
                k = max(released, key=lambda j: self.jobs[j].q_i)
            else:
                # Выбрать работу с максимальным хвостом среди всех доступных
                k = max(available, key=lambda j: (self.jobs[j].q_i, -r_prime[j]))

            # Следующая работа, которая будет доступна
            next_release = float('inf')
            l = None
            for j in available:
                if r_prime[j] > tau and r_prime[j] < next_release:
                    next_release = r_prime[j]
                    l = j

            # КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ LTH:
            # Если следующая работа имеет больший хвост, ждем её
            if l is not None and self.jobs[l].q_i > self.jobs[k].q_i:
                tau = max(tau, r_prime[l])
                continue

            # Запланировать выбранную работу
            s_k = max(tau, r_prime[k])
            schedule.append(k)
            scheduled.add(k)
            tau = s_k + self.jobs[k].d_i

            # Обновить головы последователей
            for j in sigma[k]:
                if j not in scheduled:
                    lij = self.l_matrix[k][j]
                    if lij > 0:
                        r_prime[j] = max(r_prime[j], s_k + lij)

        if len(schedule) < n:
            return schedule, float('inf')

        makespan, _ = self.calculate_makespan(schedule,
                                              {i: set(precedence[i]) for i in precedence})
        return schedule, makespan

    def _reschedule_delayed_jobs(self,
                                 schedule: List[int],
                                 precedence: Dict[int, Set[int]],
                                 heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Процедура решедулинга отложенных работ (Algorithm 4 в статье).

        Обеспечивает выполнение условия (6) для критических путей.
        """
        max_iterations = len(schedule) ** 2  # O(n^2) гарантия завершения
        iteration = 0

        current_schedule = schedule.copy()

        while iteration < max_iterations:
            iteration += 1

            # Проверить наличие отложенных работ
            critical_path, start_times = self._find_critical_path_with_starts(
                current_schedule, heads
            )

            if not critical_path:
                break

            # Найти отложенную работу в критическом пути
            delayed_job = None
            first_job = critical_path[0]

            for job in critical_path:
                r_prime = heads.get(job, self.jobs[job].r_i)
                if r_prime < start_times[first_job]:
                    delayed_job = job
                    break

            if delayed_job is None:
                break

            # Найти позицию для вставки (перед первой работой критического пути)
            i1 = first_job
            i1_idx = current_schedule.index(i1)

            # Найти предшествующую работу
            if i1_idx > 0:
                j1 = current_schedule[i1_idx - 1]
            else:
                # Вставить в начало
                new_schedule = [delayed_job]
                for j in current_schedule:
                    if j != delayed_job:
                        new_schedule.append(j)
                current_schedule = new_schedule
                continue

            # Удалить отложенную работу и вставить перед i1
            new_schedule = []
            delayed_inserted = False
            for j in current_schedule:
                if j == delayed_job and not delayed_inserted:
                    continue  # пропускаем отложенную работу
                if j == i1 and not delayed_inserted:
                    new_schedule.append(delayed_job)
                    delayed_inserted = True
                new_schedule.append(j)

            if not delayed_inserted:
                new_schedule.append(delayed_job)

            current_schedule = new_schedule

        makespan, _ = self.calculate_makespan(current_schedule,
                                              {i: set(precedence[i]) for i in precedence})
        return current_schedule, makespan

    def _find_critical_path(self,
                            schedule: List[int],
                            heads: Dict[int, float]) -> List[int]:
        """Находит критический путь в расписании."""
        makespan, start_times = self.calculate_makespan(schedule)

        if makespan == float('inf'):
            return []

        # Найти работы, определяющие makespan
        critical_path = []
        for job_id in schedule:
            completion = start_times[job_id] + self.jobs[job_id].d_i
            delivery_completion = completion + self.jobs[job_id].q_i
            if abs(delivery_completion - makespan) < 1e-6:
                critical_path.append(job_id)
                break  # берем первую (последнюю в смысле завершения)

        if not critical_path:
            return []

        # Построить путь назад
        current = critical_path[0]
        while True:
            prev_job = None
            current_start = start_times[current]

            # Найти предшественника в расписании
            current_idx = schedule.index(current)
            if current_idx > 0:
                prev_candidate = schedule[current_idx - 1]
                prev_start = start_times[prev_candidate]

                # Проверить, связаны ли работы
                if (prev_start + self.jobs[prev_candidate].d_i >= current_start - 1e-6 or
                        self.l_matrix[prev_candidate][current] > 0):
                    prev_job = prev_candidate

            if prev_job is None or prev_job in critical_path:
                break

            critical_path.insert(0, prev_job)
            current = prev_job

        return critical_path

    def _find_critical_path_with_starts(self,
                                        schedule: List[int],
                                        heads: Dict[int, float]) -> Tuple[List[int], Dict[int, float]]:
        """Находит критический путь и времена начала."""
        makespan, start_times = self.calculate_makespan(schedule)
        critical_path = self._find_critical_path(schedule, heads)
        return critical_path, start_times

    def _can_branch(self,
                    critical_path: List[int],
                    schedule: List[int],
                    heads: Dict[int, float]) -> bool:
        """
        Проверяет условие сильного ветвления.

        Сильное ветвление применяется, когда существует работа j не в критическом пути,
        которая может быть вставлена в критический путь для улучшения решения.
        """
        if not critical_path or len(critical_path) < 2:
            return False

        makespan, start_times = self.calculate_makespan(schedule)

        # Проверить работы не в критическом пути
        for job in schedule:
            if job in critical_path:
                continue

            r_prime = heads.get(job, self.jobs[job].r_i)

            # Проверить, может ли работа быть вставлена в критический путь
            for i in range(len(critical_path) - 1):
                j1 = critical_path[i]
                j2 = critical_path[i + 1]

                gap_start = start_times[j1] + self.jobs[j1].d_i
                gap_end = start_times[j2]

                if gap_start < gap_end and r_prime < gap_end:
                    return True

        return False

    def _strong_branch(self,
                       critical_path: List[int],
                       schedule: List[int],
                       precedence: Dict[int, Set[int]]) -> Tuple[Optional[Dict[int, Set[int]]],
    Optional[Dict[int, Set[int]]]]:
        """
        Сильное ветвление: создает два подузла.

        Узел 1: j1 -> job для некоторой работы job не в критическом пути
        Узел 2: job -> j1 (обратное отношение)
        """
        makespan, start_times = self.calculate_makespan(schedule)

        # Получить обновленные головы для текущего узла
        heads, _ = self._update_heads_and_tails(precedence)

        # Найти подходящую работу для ветвления
        for job in schedule:
            if job in critical_path:
                continue

            r_prime = heads.get(job, self.jobs[job].r_i)

            for i in range(len(critical_path) - 1):
                j1 = critical_path[i]
                j2 = critical_path[i + 1]

                gap_start = start_times[j1] + self.jobs[j1].d_i
                gap_end = start_times[j2]

                # Проверяем, есть ли промежуток в критическом пути
                # и может ли работа job быть вставлена в этот промежуток
                if gap_start < gap_end and r_prime < gap_end:
                    # Проверяем, не нарушит ли вставка другие ограничения
                    can_insert = True

                    # Проверяем DPC ограничения
                    for k in schedule:
                        if k == job:
                            break
                        # Если есть DPC от k к job, проверяем время
                        lij = self.l_matrix.get(k, {}).get(job, 0.0)
                        if lij > 0 and k in start_times:
                            required = start_times[k] + lij
                            if required > gap_end:
                                can_insert = False
                                break

                    if not can_insert:
                        continue

                    # Создать отношения предшествования
                    # Узел 1: j1 должен предшествовать job
                    child1_prec = defaultdict(set)
                    for k, v in precedence.items():
                        child1_prec[k] = v.copy()
                    child1_prec[j1].add(job)

                    # Узел 2: job должен предшествовать j1
                    child2_prec = defaultdict(set)
                    for k, v in precedence.items():
                        child2_prec[k] = v.copy()
                    child2_prec[job].add(j1)

                    return child1_prec, child2_prec

        return None, None

    def _weak_branch(self,
                     critical_path: List[int],
                     schedule: List[int],
                     precedence: Dict[int, Set[int]]) -> Tuple[Optional[Dict[int, Set[int]]],
    Optional[Dict[int, Set[int]]]]:
        """
        Слабое ветвление: разделяет критический путь.

        Узел 1: i -> j для некоторых i,j в критическом пути
        Узел 2: j -> i
        """
        if len(critical_path) < 2:
            return None, None

        # Взять первые две работы в критическом пути
        i = critical_path[0]
        j = critical_path[1]

        child1_prec = defaultdict(set)
        for k, v in precedence.items():
            child1_prec[k] = v.copy()
        child1_prec[i].add(j)

        child2_prec = defaultdict(set)
        for k, v in precedence.items():
            child2_prec[k] = v.copy()
        child2_prec[j].add(i)

        return child1_prec, child2_prec

    def _calculate_lower_bound(self) -> float:
        """
        Вычисляет нижнюю границу для корневого узла.

        Использует preemptive relaxation: max(r_i + d_i + q_i, max r_i + sum d_i, sum d_i + min q_i)
        """
        if not self.jobs:
            return 0.0

        # Максимальное r_i + d_i + q_i
        lb1 = max(self.jobs[j].r_i + self.jobs[j].d_i + self.jobs[j].q_i
                  for j in self.jobs)

        # Максимальное r_i + сумма всех d_i
        lb2 = max(self.jobs[j].r_i for j in self.jobs) + sum(self.jobs[j].d_i
                                                             for j in self.jobs)

        # Сумма всех d_i + минимальное q_i
        lb3 = sum(self.jobs[j].d_i for j in self.jobs) + min(self.jobs[j].q_i
                                                             for j in self.jobs)

        return max(lb1, lb2, lb3)

    def _calculate_lower_bound_with_precedence(self,
                                               precedence: Dict[int, Set[int]]) -> float:
        """
        Вычисляет нижнюю границу с учетом ограничений предшествования.
        """
        # Обновить головы с учетом предшествования
        updated_heads = {}
        for j in self.jobs:
            r = self.jobs[j].r_i
            for i in self.jobs:
                if i in precedence and j in precedence[i]:
                    r = max(r, self.jobs[i].r_i + self.l_matrix[i][j])
            updated_heads[j] = r

        # Базовая нижняя граница
        lb = self._calculate_lower_bound()

        # Улучшить с учетом критического пути в preemptive решении
        # Найти работу с максимальным r_i + d_i + q_i
        max_completion = 0
        for j in self.jobs:
            completion = updated_heads[j] + self.jobs[j].d_i + self.jobs[j].q_i
            max_completion = max(max_completion, completion)

        lb = max(lb, max_completion)

        return lb

    def _update_heads_and_tails(self,
                                precedence: Dict[int, Set[int]]) -> Tuple[Dict[int, float],
    Dict[int, float]]:
        """
        Обновляет головы и хвосты работ с учетом ограничений предшествования.
        """
        heads = {}
        tails = {}

        for j in self.jobs:
            # Обновить голову
            r = self.jobs[j].r_i
            for i in self.jobs:
                if i in precedence and j in precedence[i]:
                    r = max(r, self.jobs[i].r_i + self.l_matrix[i][j])
            heads[j] = r

            # Обновить хвост
            q = self.jobs[j].q_i
            # Учесть обратные ограничения
            for k in self.jobs:
                if j in precedence and k in precedence[j]:
                    q = max(q, self.l_matrix[j][k] + self.jobs[k].q_i)
            tails[j] = q

        return heads, tails

    def get_name(self) -> str:
        return "LDepth"
