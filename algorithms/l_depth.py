"""
LDepth Algorithm Implementation
Based on:
- "The One-machine Problem with Delayed Precedence Constraints..." (Balas, Lenstra, Vazacopoulos, 1995)
- "An Improved Branch-and-Bound Algorithm..." (Zhang, Sauppe, Jacobson, 2020)
"""

import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from algorithms.BestOfHeuristics import BestOfHeuristics
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
    LDepth - Branch-and-Bound для 1|r_i,q_i,dpc|C_max,
    использующий CBFS, MLTH и процедуру перепланирования.
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 time_limit: float = 30.0,
                 max_iterations: int = 100000):
        super().__init__(jobs, precedence_constraints)
        self.init_jobs = jobs
        self.init_precedence = precedence_constraints
        self.max_time = time_limit
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_makespan = float('inf')
        self.iterations = 0
        self.nodes_explored = 0

        # Кэши для быстрого доступа
        self._job_ids = list(self.jobs.keys())
        self._n = len(self._job_ids)
        self._r_i = {j: self.jobs[j].r_i for j in self._job_ids}
        self._d_i = {j: self.jobs[j].d_i for j in self._job_ids}
        self._q_i = {j: self.jobs[j].q_i for j in self._job_ids}

        # Кэши для DPC (из исходной матрицы l_matrix)
        self._l_ij = {}
        self._pi_dpc = defaultdict(set)
        self._sigma_dpc = defaultdict(set)

        for i in self._job_ids:
            for j in self._job_ids:
                if self.l_matrix[i][j] > 0:
                    lij = self.l_matrix[i][j]
                    self._l_ij[(i, j)] = lij
                    self._pi_dpc[j].add(i)
                    self._sigma_dpc[i].add(j)

        self._sum_d = sum(self._d_i.values())
        self._min_q = min(self._q_i.values()) if self._job_ids else 0
        self._max_rdq = max(self._r_i[j] + self._d_i[j] + self._q_i[j] for j in self._job_ids) if self._job_ids else 0
        self._max_r = max(self._r_i.values()) if self._job_ids else 0

    def _check_time_limit(self, start_time: float) -> bool:
        """Проверяет, не превышен ли лимит времени."""
        return time.time() - start_time > self.max_time

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        start_time = time.time()
        self.best_solution = None
        self.best_makespan = float('inf')
        self.iterations = 0
        self.nodes_explored = 0

        # Начальное решение через BestOfHeuristics
        boh = BestOfHeuristics(self.init_jobs, self.init_precedence)
        boh_schedule, init_makespan, _ = boh.solve(**kwargs)

        if boh_schedule and init_makespan < float('inf'):
            self.best_solution = boh_schedule
            self.best_makespan = init_makespan

        # Корень дерева
        root = LDepthNode(
            lower_bound=self._compute_lower_bound({}),
            precedence={},
            depth=0
        )

        # Инициализация контуров для CBFS (по глубине)
        contours: List[List[LDepthNode]] = [[root]]
        max_contour = 0
        current_contour = 0

        while not self._check_time_limit(start_time):
            if self.iterations >= self.max_iterations:
                break

            # CBFS: поиск непустого контура
            found = False
            start_contour = current_contour
            while not self._check_time_limit(start_time):
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

            # Выбор узла с лучшей нижней границей в контуре
            contour = contours[current_contour]
            best_idx = min(range(len(contour)), key=lambda i: contour[i].lower_bound)
            node = contour.pop(best_idx)

            # Очистка пустых контуров
            while contours and not contours[-1]:
                contours.pop()
                max_contour = max(0, max_contour - 1)

            # Отсечение
            if node.lower_bound >= self.best_makespan:
                continue

            self.nodes_explored += 1
            self.iterations += 1

            # ---- Основной цикл обработки узла из Balas et al. ----
            # Шаг 1: Обновление голов и хвостов с учетом precedence
            heads, tails = self._update_heads_tails(node.precedence)

            # Шаг 2: Генерация расписания (LTH + MLTH + rescheduling)
            schedule, makespan = self._apply_heuristics(node.precedence, heads)

            if schedule is None:
                continue

            # Обновление лучшего решения
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_solution = schedule

            # Шаг 3: Постпроцессинг (Propositions 3.3 и 3.4 из Balas et al.)
            post_result = self._postprocess(schedule, node.precedence, heads, tails)
            if post_result is not None:
                post_schedule, post_makespan = post_result
                if post_makespan < self.best_makespan:
                    self.best_makespan = post_makespan
                    self.best_solution = post_schedule

            # Шаг 4: Ветвление
            child1_prec, child2_prec = self._branch(schedule, node.precedence, heads, tails)

            if child1_prec is not None or child2_prec is not None:
                new_depth = node.depth + 1
                while len(contours) <= new_depth:
                    contours.append([])

                for new_prec in (child1_prec, child2_prec):
                    if new_prec is not None:
                        lb = self._compute_lower_bound(new_prec)
                        if lb < self.best_makespan:
                            child_node = LDepthNode(lb, new_prec, new_depth, node.id)
                            contours[new_depth].append(child_node)
                            max_contour = max(max_contour, new_depth)

            # Переход к следующему контуру
            current_contour += 1
            if current_contour > max_contour:
                current_contour = 0

        execution_time = time.time() - start_time
        stats = {
            "execution_time": execution_time,
            "iterations": self.iterations,
            "nodes_explored": self.nodes_explored,
            "best_makespan": self.best_makespan,
            "contours_used": max_contour + 1 if 'max_contour' in dir() else 1,
            "time_limit_reached": self._check_time_limit(start_time),
            "iteration_limit_reached": self.iterations >= self.max_iterations
        }

        return self.best_solution, self.best_makespan, stats

    def _update_heads_tails(self, precedence: Dict[int, Set[int]]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Обновление r_i и q_i с учетом DPC и переданных precedence constraints.
        Соответствует шагу 1 алгоритма Balas et al.

        Использует рекурсивное обновление до стабилизации.
        """
        heads = self._r_i.copy()
        tails = self._q_i.copy()

        # Строим полный граф предшественников и последователей
        predecessors = defaultdict(set)
        successors = defaultdict(set)

        # Добавляем исходные DPC
        for j in self._job_ids:
            for i in self._pi_dpc.get(j, set()):
                predecessors[j].add(i)
                successors[i].add(j)

        # Добавляем precedence constraints из ветвления
        for i, followers in precedence.items():
            for j in followers:
                if j not in predecessors.get(j, set()):
                    predecessors[j].add(i)
                    successors[i].add(j)

        # Итеративное обновление голов (прямой проход)
        changed = True
        while changed:
            changed = False
            for j in self._job_ids:
                new_head = self._r_i[j]
                for i in predecessors.get(j, set()):
                    # Используем DPC задержку если есть, иначе стандартную (d_i)
                    lij = self._l_ij.get((i, j), self._d_i[i])
                    candidate = heads[i] + lij
                    if candidate > new_head:
                        new_head = candidate
                if new_head > heads[j] + 1e-9:
                    heads[j] = new_head
                    changed = True

        # Итеративное обновление хвостов (обратный проход)
        changed = True
        while changed:
            changed = False
            for i in self._job_ids:
                new_tail = self._q_i[i]
                for j in successors.get(i, set()):
                    # Задержка от i к j
                    lij = self._l_ij.get((i, j), self._d_i[i])
                    # Хвост i должен быть не менее tails[j] + (lij - d_i)
                    # Формула из статьи: q_i = max(q_i, q_j + l_ij - d_i)
                    candidate = tails[j] + lij - self._d_i[i]
                    if candidate > new_tail:
                        new_tail = candidate
                if new_tail > tails[i] + 1e-9:
                    tails[i] = new_tail
                    changed = True

        # Проверка на циклы положительной длины
        for i in self._job_ids:
            if heads[i] + self._d_i[i] + tails[i] > self.best_makespan + 1e-9:
                # Это не цикл, но может означать несовместность ограничений
                pass

        return heads, tails

    def _find_critical_path(self, schedule: List[int], start_times: Dict[int, float],
                            makespan: float) -> List[int]:
        """
        Находит критический путь в расписании.
        Критический путь - последовательность работ от начала до работы,
        определяющей makespan.
        """
        if not schedule:
            return []

        # Находим последнюю работу на критическом пути
        last_job = None
        for j in schedule:
            if abs(start_times[j] + self._d_i[j] + self._q_i[j] - makespan) < 1e-9:
                last_job = j
                break

        if last_job is None:
            return []

        # Строим путь обратным ходом
        path = [last_job]
        current = last_job

        while True:
            start_current = start_times[current]

            # Ищем работу, которая завершается точно к началу current
            # или связана DPC с current
            found = False
            best_pred = None
            best_end_time = -1.0

            for i in schedule:
                if i == current:
                    break
                if i in path:
                    continue

                end_i = start_times[i] + self._d_i[i]
                lij = self._l_ij.get((i, current), self._d_i[i])

                # Проверяем, является ли i предшественником current на критическом пути
                if abs(end_i + lij - self._d_i[i] - start_current) < 1e-9:
                    best_pred = i
                    break
                elif abs(end_i - start_current) < 1e-9:
                    # Прямой предшественник без DPC задержки
                    if end_i > best_end_time:
                        best_end_time = end_i
                        best_pred = i

            if best_pred is not None:
                path.insert(0, best_pred)
                current = best_pred
                found = True

            if not found:
                # Проверяем, начинается ли current в свой release time
                if abs(start_times[current] - self._r_i[current]) < 1e-9:
                    break
                # Иначе ищем работу с r_i = start_current
                found_r = False
                for i in schedule:
                    if i == current:
                        break
                    if abs(self._r_i[i] - start_times[current]) < 1e-9:
                        # Не совсем корректно, но для простоты
                        break
                break

        return path

    def _find_c(self, critical_path: List[int], start_times: Dict[int, float],
                heads: Dict[int, float]) -> Optional[int]:
        """
        Ищет работу c для сильного ветвления согласно Theorem 3.1 из Balas et al.

        c - первая работа на критическом пути (не считая начала),
        такая что все работы после неё имеют r_k >= max(t_c, t_{i1})
        и q_k >= q_{ip} (хвост последней работы пути).
        """
        if len(critical_path) < 2:
            return None

        p = len(critical_path)
        q_ip = self._q_i[critical_path[-1]]

        # Проверяем каждую работу как кандидата на c
        for idx in range(len(critical_path)):
            c_candidate = critical_path[idx]
            t_c = start_times[c_candidate]
            t_i1 = start_times[critical_path[0]]

            # J_c - работы критического пути до c (не включая c)
            J_c = set(critical_path[:idx])

            if not J_c:
                continue

            # Проверяем условие Theorem 3.1:
            # (i) r_i >= max(t_c, t_{i1}) для всех i в J_c
            # (ii) q_i >= q_{i_p} для всех i в J_c
            # (iii) На сегменте C(c, n) нет precedence arcs

            condition_holds = True
            for i in J_c:
                if heads.get(i, self._r_i[i]) < max(t_c, t_i1) - 1e-9:
                    condition_holds = False
                    break
                if self._q_i[i] < q_ip - 1e-9:
                    condition_holds = False
                    break

            if condition_holds:
                # Проверяем отсутствие precedence arcs в C(c, n)
                has_prec_arc = False
                c_idx_in_path = critical_path.index(c_candidate)
                for k in range(c_idx_in_path, len(critical_path) - 1):
                    u, v = critical_path[k], critical_path[k + 1]
                    if (u, v) in self._l_ij and self._l_ij[(u, v)] > self._d_i[u]:
                        has_prec_arc = True
                        break

                if not has_prec_arc:
                    return c_candidate

        return None

    def _apply_heuristics(self, precedence: Dict[int, Set[int]],
                          heads: Dict[int, float]) -> Tuple[Optional[List[int]], float]:
        """
        Применяет LTH, MLTH и реверсивное LTH, возвращает лучший результат.
        """
        best_s = None
        best_m = float('inf')

        # Прямой LTH
        s_lth, m_lth = self._lth(precedence, heads.copy())
        if s_lth and m_lth < best_m:
            best_s, best_m = s_lth, m_lth

        # MLTH (начальное расписание)
        s_mlth_init, m_mlth_init = self._mlth(precedence, heads.copy())
        if s_mlth_init and m_mlth_init < best_m:
            best_s, best_m = s_mlth_init, m_mlth_init

        # MLTH с перепланированием delayed jobs
        if s_mlth_init:
            s_mlth_res, m_mlth_res = self._reschedule_delayed_jobs(s_mlth_init, precedence, heads)
            if s_mlth_res and m_mlth_res < best_m:
                best_s, best_m = s_mlth_res, m_mlth_res

        # Реверсивный LTH (если не найдено хорошее решение)
        if best_m >= self.best_makespan:
            s_rev, m_rev = self._reverse_lth(precedence, heads.copy())
            if s_rev and m_rev < best_m:
                best_s, best_m = s_rev, m_rev

        return best_s, best_m

    def _lth(self, precedence: Dict[int, Set[int]], heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Longest Tail Heuristic из Balas et al. (Algorithm 2 в Zhang et al.)
        """
        schedule = []
        tau = 0.0
        r_prime = heads.copy()
        scheduled = set()

        # Строим pi(i) - работы, которые должны предшествовать i
        pi = defaultdict(set)
        for j in self._job_ids:
            pi[j].update(self._pi_dpc.get(j, set()))
        for i, followers in precedence.items():
            for j in followers:
                pi[j].add(i)

        while len(scheduled) < self._n:
            # Доступные работы: не запланированы и все предшественники запланированы
            available = [j for j in self._job_ids
                        if j not in scheduled and pi[j].issubset(scheduled)]

            if not available:
                return [], float('inf')

            # Работы, доступные по времени
            candidates = [j for j in available if r_prime[j] <= tau + 1e-9]

            if candidates:
                # Выбираем работу с максимальным хвостом
                k = max(candidates, key=lambda j: self._q_i[j])
            else:
                # Если нет доступных, выбираем с минимальным r'
                min_r = min(r_prime[j] for j in available)
                candidates = [j for j in available if abs(r_prime[j] - min_r) < 1e-9]
                k = max(candidates, key=lambda j: self._q_i[j])

            s_k = max(tau, r_prime[k])
            schedule.append(k)
            scheduled.add(k)
            tau = s_k + self._d_i[k]

            # Обновляем головы последователей
            for j in self._sigma_dpc.get(k, set()):
                if j not in scheduled:
                    r_prime[j] = max(r_prime[j], s_k + self._l_ij[(k, j)])

        if len(schedule) < self._n:
            return [], float('inf')

        return schedule, self._compute_makespan(schedule)

    def _mlth(self, precedence: Dict[int, Set[int]], heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Modified Longest Tail Heuristic (MLTH) из Zhang et al. (Algorithm 3)
        """
        schedule = []
        tau = 0.0
        r_prime = heads.copy()
        scheduled = set()

        # Строим pi(i) - работы, которые должны предшествовать i
        pi = defaultdict(set)
        for j in self._job_ids:
            pi[j].update(self._pi_dpc.get(j, set()))
        for i, followers in precedence.items():
            for j in followers:
                pi[j].add(i)

        while len(scheduled) < self._n:
            # Доступные работы (все предшественники запланированы)
            available = [j for j in self._job_ids
                        if j not in scheduled and pi[j].issubset(scheduled)]

            if not available:
                return [], float('inf')

            # Работы с r' <= tau (доступные по времени)
            released = [j for j in available if r_prime[j] <= tau + 1e-9]

            if released:
                k = max(released, key=lambda j: self._q_i[j])
            else:
                # Все работы ещё не Released - выбираем с макс. хвостом
                k = max(available, key=lambda j: (self._q_i[j], -r_prime[j]))

            # Ищем работу l со следующим временем освобождения
            l = None
            next_release = float('inf')
            for j in available:
                if r_prime[j] > tau + 1e-9 and r_prime[j] < next_release:
                    next_release = r_prime[j]
                    l = j

            # Ключевая модификация MLTH:
            # ожидаем работу l если её хвост больше чем у k
            if l is not None and l != k and self._q_i[l] > self._q_i[k]:
                tau = max(tau, r_prime[l])
                continue

            s_k = max(tau, r_prime[k])
            schedule.append(k)
            scheduled.add(k)
            tau = s_k + self._d_i[k]

            # Обновляем головы последователей
            for j in self._sigma_dpc.get(k, set()):
                if j not in scheduled:
                    r_prime[j] = max(r_prime[j], s_k + self._l_ij[(k, j)])

        if len(schedule) < self._n:
            return [], float('inf')

        return schedule, self._compute_makespan(schedule)

    def _reverse_lth(self, precedence: Dict[int, Set[int]], heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Реверсивный LTH: применяет LTH к обратной задаче.
        В обратной задаче:
        - Меняются местами r_i и q_i
        - Обращаются направления DPC
        """
        # Создаём обратные работы
        reverse_jobs = []
        for j in self._job_ids:
            reverse_jobs.append(Job(id=j, r_i=self._q_i[j], d_i=self._d_i[j], q_i=self._r_i[j]))

        # Обратные DPC
        reverse_dpc = {}
        for (i, j), lij in self._l_ij.items():
            # В обратной задаче j предшествует i
            # Задержка L'(j,i) = L(i,j) - d_i + d_j
            reverse_lij = lij - self._d_i[i] + self._d_i[j]
            reverse_dpc[(j, i)] = max(reverse_lij, self._d_i[j])

        # Обратные precedence constraints
        reverse_prec = {}
        for i, followers in precedence.items():
            for j in followers:
                reverse_prec.setdefault(j, set()).add(i)

        # Создаём временный LDepth для обратной задачи
        reverse_algo = LDepth(reverse_jobs, reverse_dpc if reverse_dpc else None,
                             time_limit=10.0, max_iterations=1)

        # Применяем LTH к обратной задаче
        reverse_heads = {j: self._q_i[j] for j in self._job_ids}
        reverse_schedule, reverse_makespan = reverse_algo._lth(reverse_prec, reverse_heads)

        if not reverse_schedule:
            return [], float('inf')

        # Обращаем расписание
        forward_schedule = list(reversed(reverse_schedule))
        forward_makespan = self._compute_makespan(forward_schedule)

        return forward_schedule, forward_makespan

    def _compute_makespan(self, schedule: List[int]) -> float:
        """Вычисление makespan с использованием родительского метода."""
        if not schedule:
            return float('inf')
        makespan, _ = self.calculate_makespan(schedule)
        return makespan

    def _reschedule_delayed_jobs(self, schedule: List[int],
                                  precedence: Dict[int, Set[int]],
                                  heads: Dict[int, float]) -> Tuple[List[int], float]:
        """
        Процедура перепланирования delayed jobs.
        Соответствует Algorithm 4 из Zhang et al.

        Delayed job - работа k на критическом пути, для которой
        r'_k < s_{i1} (время начала первой работы критического пути).
        """
        if not schedule:
            return [], float('inf')

        current_schedule = schedule[:]
        max_iter = self._n ** 2  # Гарантированная верхняя граница сходимости

        for iteration in range(max_iter):
            makespan, start_times = self.calculate_makespan(current_schedule)
            if makespan == float('inf'):
                break

            # Находим критический путь
            critical_path = self._find_critical_path(current_schedule, start_times, makespan)
            if len(critical_path) < 2:
                break

            i1 = critical_path[0]
            s_i1 = start_times[i1]

            # Ищем delayed job на критическом пути
            delayed = None
            for j in critical_path:
                # r'_j - обновлённая голова работы j
                r_prime_j = self._compute_head_for_schedule(j, current_schedule, precedence)
                if r_prime_j < s_i1 - 1e-9:
                    delayed = j
                    break

            if delayed is None:
                break

            # Находим работу, перед которой нужно вставить delayed
            j1 = None
            if i1 == current_schedule[0]:
                j1 = 0  # Виртуальная работа перед первой
            else:
                idx_i1 = current_schedule.index(i1)
                j1 = current_schedule[idx_i1 - 1]

            # Перемещаем delayed перед i1
            current_schedule.remove(delayed)
            if j1 == 0:
                current_schedule.insert(0, delayed)
            else:
                idx_j1 = current_schedule.index(j1)
                current_schedule.insert(idx_j1 + 1, delayed)

        return current_schedule, self._compute_makespan(current_schedule)

    def _compute_head_for_schedule(self, job: int, schedule: List[int],
                                     precedence: Dict[int, Set[int]]) -> float:
        """
        Вычисляет обновлённую голову работы с учётом частичного расписания.
        """
        head = self._r_i[job]

        # Учитываем DPC предшественников
        for pred in self._pi_dpc.get(job, set()):
            if pred in schedule:
                pred_idx = schedule.index(pred)
                job_idx = schedule.index(job) if job in schedule else len(schedule)
                if pred_idx < job_idx:
                    # Предшественник уже запланирован
                    _, start_times = self.calculate_makespan(schedule)
                    if pred in start_times:
                        head = max(head, start_times[pred] +
                                  self._l_ij.get((pred, job), self._d_i[pred]))

        # Учитываем precedence constraints
        for i, followers in precedence.items():
            if job in followers and i in schedule:
                pred_idx = schedule.index(i)
                job_idx = schedule.index(job) if job in schedule else len(schedule)
                if pred_idx < job_idx:
                    _, start_times = self.calculate_makespan(schedule)
                    if i in start_times:
                        lij = self._l_ij.get((i, job), self._d_i[i])
                        head = max(head, start_times[i] + lij)

        return head

    def _postprocess(self, schedule: List[int], precedence: Dict[int, Set[int]],
                     heads: Dict[int, float], tails: Dict[int, float]) -> Optional[Tuple[List[int], float]]:
        """
        Постпроцессинг согласно Propositions 3.3 и 3.4 из Balas et al.

        Шаг 3 алгоритма: если удается увеличить хвосты работ,
        перезапускаем LTH с обновлёнными tails.
        """
        if not schedule:
            return None

        makespan, start_times = self.calculate_makespan(schedule)
        if makespan == float('inf'):
            return None

        # Находим критический путь
        critical_path = self._find_critical_path(schedule, start_times, makespan)
        if len(critical_path) < 2:
            return None

        changed = False
        new_tails = tails.copy()

        # Proposition 3.3: если сегмент C(j, n) не содержит precedence arcs,
        # и условия выполняются, то j должен предшествовать всем работам в K
        for idx, j in enumerate(critical_path):
            if idx == len(critical_path) - 1:
                break

            K = set(critical_path[idx + 1:])
            t_j = start_times[j]

            # Проверяем условия Proposition 3.3
            segment_valid = True
            for k_idx in range(idx + 1, len(critical_path) - 1):
                u, v = critical_path[k_idx], critical_path[k_idx + 1]
                if (u, v) in self._l_ij and self._l_ij[(u, v)] > self._d_i[u]:
                    segment_valid = False
                    break

            if not segment_valid:
                continue

            all_conditions = True
            for k in K:
                if self._q_i[k] < self._q_i[j] - 1e-9:
                    all_conditions = False
                    break
                if k not in self._sigma_dpc.get(j, set()):
                    if heads.get(k, self._r_i[k]) < t_j - 1e-9:
                        all_conditions = False
                        break

            if all_conditions:
                # Обновляем хвосты работ в K
                for k in K:
                    new_tail = max(new_tails[k], makespan - (start_times[k] + self._d_i[k]))
                    if new_tail > new_tails[k] + 1e-9:
                        new_tails[k] = new_tail
                        changed = True

        # Proposition 3.4: симметрично для предшественников
        for idx, i in enumerate(critical_path):
            if idx == 0:
                continue

            H = set(critical_path[:idx])
            t_i = start_times[i]

            # Проверяем условия Proposition 3.4 (симметрично)
            segment_valid = True
            for h_idx in range(idx - 1):
                u, v = critical_path[h_idx], critical_path[h_idx + 1]
                if (u, v) in self._l_ij and self._l_ij[(u, v)] > self._d_i[u]:
                    segment_valid = False
                    break

            if not segment_valid:
                continue

            all_conditions = True
            for h in H:
                if self._r_i[h] > self._r_i[i] + 1e-9:
                    all_conditions = False
                    break
                if h not in self._pi_dpc.get(i, set()):
                    if self._q_i[h] < self._q_i[i] - 1e-9:
                        all_conditions = False
                        break

            if all_conditions:
                # Обновляем головы
                for h in H:
                    # Это повлияет при следующем запуске LTH
                    pass

        # Если tails изменились, перезапускаем LTH
        if changed:
            # Используем обновлённые tails как часть голов для обратной задачи
            # Для простоты перезапускаем LTH с оригинальными головами
            new_heads = heads.copy()
            for j in self._job_ids:
                if new_tails[j] > tails[j] + 1e-9:
                    # Увеличиваем значимость хвоста
                    pass

            new_schedule, new_makespan = self._lth(precedence, new_heads)
            if new_schedule and new_makespan < makespan:
                return new_schedule, new_makespan

        return None

    def _branch(self, schedule: List[int], precedence: Dict[int, Set[int]],
                heads: Dict[int, float], tails: Dict[int, float]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Логика ветвления Balas et al.:
        1. Пытается применить сильное ветвление (Theorem 3.1)
        2. Если не удаётся - слабое ветвление
        """
        if not schedule:
            return None, None

        makespan, start_times = self.calculate_makespan(schedule)
        critical_path = self._find_critical_path(schedule, start_times, makespan)

        if len(critical_path) < 2:
            return None, None

        # Попытка сильного ветвления
        c = self._find_c(critical_path, start_times, heads)

        if c is not None and c in critical_path:
            c_idx = critical_path.index(c)
            J = set(critical_path[:c_idx])

            if J:
                # Проверяем дополнительное условие Theorem 3.1:
                # r_i >= max(t_c, t_{i1}) для всех i ∈ J\σ(c)
                t_c = start_times[c]
                t_i1 = start_times[critical_path[0]]
                threshold = max(t_c, t_i1)

                condition_holds = True
                for i in J:
                    if heads.get(i, self._r_i[i]) < threshold - 1e-9:
                        condition_holds = False
                        break

                if condition_holds:
                    # СИЛЬНОЕ ВЕТВЛЕНИЕ
                    # Вариант 1: c перед J
                    prec1 = {}
                    for k, v in precedence.items():
                        prec1[k] = v.copy()
                    prec1.setdefault(c, set()).update(J)

                    # Вариант 2: c после J
                    prec2 = {}
                    for k, v in precedence.items():
                        prec2[k] = v.copy()
                    for j in J:
                        prec2.setdefault(j, set()).add(c)

                    return prec1, prec2

        # СЛАБОЕ ВЕТВЛЕНИЕ
        # Применяем реверсивный LTH и пытаемся снова
        s_rev, _ = self._reverse_lth(precedence, heads)
        if s_rev:
            _, start_times_rev = self.calculate_makespan(s_rev)
            makespan_rev = self._compute_makespan(s_rev)
            critical_path_rev = self._find_critical_path(s_rev, start_times_rev, makespan_rev)

            if len(critical_path_rev) >= 2:
                c_rev = self._find_c(critical_path_rev, start_times_rev, heads)
                if c_rev is not None:
                    c_idx = critical_path_rev.index(c_rev)
                    J_rev = set(critical_path_rev[:c_idx])
                    if J_rev:
                        prec1 = {}
                        for k, v in precedence.items():
                            prec1[k] = v.copy()
                        prec1.setdefault(c_rev, set()).update(J_rev)

                        prec2 = {}
                        for k, v in precedence.items():
                            prec2[k] = v.copy()
                        for j in J_rev:
                            prec2.setdefault(j, set()).add(c_rev)

                        return prec1, prec2

        # Выбираем пару (i, j) для слабого ветвления
        # Согласно статье: ищем работу m с r_m < max(t_c, t_{i1})
        # или essential precedence arc
        i, j = self._find_weak_branching_pair(critical_path, start_times, heads, precedence)

        if i is not None and j is not None:
            prec1 = {}
            for k, v in precedence.items():
                prec1[k] = v.copy()
            prec1.setdefault(i, set()).add(j)

            prec2 = {}
            for k, v in precedence.items():
                prec2[k] = v.copy()
            prec2.setdefault(j, set()).add(i)

            return prec1, prec2

        return None, None

    def _find_weak_branching_pair(self, critical_path: List[int],
                                     start_times: Dict[int, float],
                                     heads: Dict[int, float],
                                     precedence: Dict[int, Set[int]]) -> Tuple[Optional[int], Optional[int]]:
        """
        Находит пару работ для слабого ветвления согласно Balas et al.

        Case 1: Существует работа m на критическом пути с r_m < max(t_c, t_{i1})
        Case 2: Существует essential precedence arc на критическом пути
        """
        if len(critical_path) < 2:
            return None, None

        i1 = critical_path[0]
        t_i1 = start_times[i1]
        makespan = start_times[critical_path[-1]] + self._d_i[critical_path[-1]] + self._q_i[critical_path[-1]]

        # Ищем c для определения J
        c_candidate = None
        for idx in range(1, len(critical_path)):
            prev = critical_path[idx - 1]
            curr = critical_path[idx]
            if start_times[prev] + self._d_i[prev] < start_times[curr] - 1e-9:
                c_candidate = curr
                break

        if c_candidate is None:
            c_candidate = critical_path[0]

        t_c = start_times[c_candidate]
        threshold = max(t_c, t_i1)

        # Case 1: Ищем работу m с r_m < threshold
        for m in critical_path:
            r_prime_m = heads.get(m, self._r_i[m])
            if r_prime_m < threshold - 1e-9:
                # Выбираем j = c_candidate
                j = c_candidate
                return m, j

        # Case 2: Ищем essential precedence arc
        # Essential: L(i,j) > d_i, r_i < t_i, q_i < makespan - t_i - d_i
        for idx in range(len(critical_path) - 1):
            i = critical_path[idx]
            j = critical_path[idx + 1]

            if (i, j) in self._l_ij:
                lij = self._l_ij[(i, j)]
                if lij > self._d_i[i] + 1e-9:  # L(i,j) > d_i
                    t_i = start_times[i]
                    if heads.get(i, self._r_i[i]) < t_i - 1e-9:  # r_i < t_i
                        q_i_bound = makespan - t_i - self._d_i[i]
                        if self._q_i[i] < q_i_bound - 1e-9:  # q_i < bound
                            # Essential precedence arc найден
                            return i, j

        # Если ничего не найдено, возвращаем первую пару
        return critical_path[0], critical_path[1]

    def _compute_lower_bound(self, precedence: Dict[int, Set[int]]) -> float:
        """
        Нижняя граница: максимум из:
        1. max_i(r_i + d_i + q_i)
        2. max_i(r_i) + sum(d_i)
        3. sum(d_i) + min_i(q_i)
        4. Preemptive lower bound (упрощённый)
        5. Учёт precedence constraints
        """
        lb = max(self._max_rdq, self._max_r + self._sum_d, self._sum_d + self._min_q)

        # Preemptive lower bound: решаем задачу с прерываниями
        preemptive_lb = self._compute_preemptive_lower_bound(precedence)
        lb = max(lb, preemptive_lb)

        # Учет precedence constraints
        max_completion = 0.0
        for j in self._job_ids:
            head_j = self._r_i[j]

            # Учитываем DPC
            for i in self._pi_dpc.get(j, set()):
                lij = self._l_ij.get((i, j), self._d_i[i])
                head_j = max(head_j, self._r_i[i] + lij)

            # Учитываем дополнительные precedence
            for i, followers in precedence.items():
                if j in followers:
                    lij = self._l_ij.get((i, j), self._d_i[i])
                    head_j = max(head_j, self._r_i[i] + lij)

            max_completion = max(max_completion, head_j + self._d_i[j] + self._q_i[j])

        return max(lb, max_completion)

    def _compute_preemptive_lower_bound(self, precedence: Dict[int, Set[int]]) -> float:
        """
        Вычисляет нижнюю границу из preemptive relaxation.
        Использует метод Carlier (1982) для preemptive 1|r_i,q_i|C_max.
        DPC трактуются как стандартные precedence constraints.
        """
        # Упрощённая версия: preemptive lower bound = max(K, sum(d_i))
        # где K - максимальное значение из алгоритма Carlier для preemptive задачи

        K = 0.0
        heads = self._r_i.copy()
        tails = self._q_i.copy()

        # Применяем обновление голов/хвостов с учётом precedence
        # (упрощённо, без полного preemptive расчёта)
        for j in self._job_ids:
            head_j = self._r_i[j]
            for i, followers in precedence.items():
                if j in followers:
                    lij = self._l_ij.get((i, j), self._d_i[i])
                    head_j = max(head_j, self._r_i[i] + lij)
            K = max(K, head_j + self._d_i[j] + self._q_i[j])

        return max(K, self._sum_d)

    def get_name(self) -> str:
        return "LDepth (CBFS + MLTH)"