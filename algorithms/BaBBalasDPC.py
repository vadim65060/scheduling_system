"""
Точный алгоритм ветвей и границ для задачи 1|r_j, q_j, DPC|C_max
Основан на статье Balas, Lenstra, Vazacopoulos (1995)
Полностью исправленная версия с учётом транзитивности DPC, корректной обратной задачей
и улучшенным контролем повторных состояний.
"""

import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set, FrozenSet

from core.Algorithm import Algorithm
from core.job import Job


# ============================================================
# Флаг отладки - установите False для отключения вывода
# ============================================================
DEBUG = True


def debug_print(*args, **kwargs):
    """Условный вывод отладочной информации."""
    if DEBUG:
        print(*args, **kwargs)


class BalasBaBDPC(Algorithm):
    """
    Алгоритм ветвей и границ для задачи одного станка с отложенными ограничениями предшествования.
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 time_limit: float = 60.0,
                 debug: bool = False):
        super().__init__(jobs, precedence_constraints)
        self.time_limit = time_limit
        self.start_time = 0
        self.debug = debug or DEBUG  # можно переопределить через параметр

        self.best_schedule: Optional[List[int]] = None
        self.best_makespan: float = float('inf')
        self.timed_out: bool = False

        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0
        self.best_sigma = None
        self.best_pi = None

    def _debug_print(self, *args, **kwargs):
        """Условный вывод отладочной информации."""
        if self.debug:
            print(*args, **kwargs)

    def _build_incoming_dpc(self, l_matrix=None) -> defaultdict:
        """Строит список входящих DPC на основе текущей l_matrix."""
        if l_matrix is None:
            l_matrix = self.l_matrix
        incoming = defaultdict(list)
        for i in self.jobs:
            for j in self.jobs:
                lij = l_matrix[i][j]
                if lij > 0:
                    incoming[j].append((i, lij))
        return incoming

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        self.start_time = time.time()
        self.timed_out = False

        # Получаем флаг отладки из kwargs, если передан
        if 'debug' in kwargs:
            self.debug = kwargs['debug']

        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0

        jobs_list = list(self.jobs.values())

        # Проверка на взаимные DPC (только если DEBUG включен)
        if self.debug:
            for i in self.jobs:
                for j in self.jobs:
                    if self.l_matrix[i][j] > 0 and self.l_matrix[j][i] > 0:
                        print(f"⚠️ WARNING: Mutual DPC between {i} and {j}")

        initial_data = {
            'r': {j.id: j.r_i for j in jobs_list},
            'q': {j.id: j.q_i for j in jobs_list},
            'sigma': defaultdict(set),
            'pi': defaultdict(set),
        }

        # Инициализируем sigma/pi из заданных DPC
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    self._add_precedence(initial_data['sigma'], initial_data['pi'], i, j)

        # Используем внутренний LTH для начального решения
        initial_schedule, initial_makespan, _ = self._longest_tail_heuristic(initial_data, self.l_matrix)

        self.best_makespan = initial_makespan
        self.best_schedule = initial_schedule.copy() if initial_schedule else None
        self.best_sigma = {k: set(v) for k, v in initial_data['sigma'].items()}
        self.best_pi = {k: set(v) for k, v in initial_data['pi'].items()}
        initial_upper_bound = initial_makespan

        self._debug_print(f"Initial LTH makespan: {initial_makespan:.2f}")

        # Запускаем B&B
        self._branch_and_bound(initial_data, initial_upper_bound, depth=0, history=set())

        elapsed = time.time() - self.start_time

        if self.timed_out and self.best_schedule is None:
            self.best_makespan = float('inf')

        stats = {
            'execution_time': elapsed,
            'nodes_explored': self.nodes_explored,
            'strong_branches': self.strong_branches,
            'weak_branches': self.weak_branches,
            'pruned_by_bound': self.pruned_by_bound,
            'pruned_by_test': self.pruned_by_test,
            'timed_out': self.timed_out,
            'initial_makespan': initial_makespan,
            'improvement': initial_makespan - self.best_makespan if self.best_makespan < float('inf') else 0,
            'optimal': self.best_schedule is not None and not self.timed_out
        }

        return self.best_schedule, self.best_makespan, stats

    def _update_best(self, schedule: List[int], makespan: float, data: Dict, upper_bound: float) -> float:
        """Обновляет лучшее найденное решение."""
        if makespan < self.best_makespan - 1e-6:
            self.best_makespan = makespan
            self.best_schedule = schedule.copy()
            self.best_sigma = {k: set(v) for k, v in data['sigma'].items()}
            self.best_pi = {k: set(v) for k, v in data['pi'].items()}
            self._debug_print(f"  -> New best makespan: {makespan:.2f}")
            return min(upper_bound, makespan)
        return upper_bound

    def _add_precedence(self, sigma: Dict[int, Set[int]], pi: Dict[int, Set[int]],
                        i: int, j: int) -> bool:
        """
        Добавляет отношение i -> j с транзитивным замыканием.
        Возвращает False, если создаётся цикл.
        """
        if i == j:
            return False

        if j in sigma.get(i, set()):
            return True  # уже есть

        if i in sigma.get(j, set()):
            return False

        sigma[i].add(j)
        pi[j].add(i)

        # Транзитивность: все, кто должны быть после j, теперь должны быть после i
        for k in list(sigma.get(j, set())):
            if k != i:
                if not self._add_precedence(sigma, pi, i, k):
                    return False

        # Транзитивность: все, кто должны быть перед i, теперь должны быть перед j
        for k in list(pi.get(i, set())):
            if k != j:
                if not self._add_precedence(sigma, pi, k, j):
                    return False

        return True

    def _is_essential_precedence_arc(self, u: int, v: int, data: Dict,
                                      start_times: Dict[int, float], C_max: float) -> bool:
        """Проверяет, является ли дуга (u, v) essential согласно статье."""
        lij = self.l_matrix[u][v]
        if lij <= self.jobs[u].d_i:
            return False

        r_u = data['r'][u]
        t_u = start_times.get(u, 0)
        q_u = data['q'][u]

        if r_u >= t_u - 1e-6:
            return False

        if q_u >= C_max - t_u - self.jobs[u].d_i - 1e-6:
            return False

        return True

    def _is_precedence_arc(self, u: int, v: int, data: Dict,
                           start_times: Dict[int, float], C_max: float,
                           l_matrix=None) -> bool:
        """Проверяет, является ли дуга (u,v) precedence arc любого типа (essential или нет)."""
        if l_matrix is None:
            l_matrix = self.l_matrix

        if l_matrix[u][v] > 0:
            return True

        t_u = start_times.get(u, -float('inf'))
        t_v = start_times.get(v, -float('inf'))
        if t_u > -float('inf') and t_v > -float('inf'):
            if abs(t_v - (t_u + self.jobs[u].d_i)) < 1e-6:
                return True

        return False

    def _find_critical_path_via_graph(self, schedule: List[int], start_times: Dict[int, float],
                                      C_max: float, data: Dict, l_matrix=None) -> Optional[Dict]:
        """Находит критический путь, игнорируя фиктивные работы."""
        if l_matrix is None:
            l_matrix = self.l_matrix

        if not schedule:
            return None

        max_job_id = max(self.jobs.keys())

        # Убираем фиктивные работы из расписания для анализа
        real_schedule = [j for j in schedule if j not in (0, max_job_id)]
        if not real_schedule:
            return None

        # Быстрая проверка через fallback
        result = self._find_critical_path_fallback(schedule, start_times, C_max, data, l_matrix)
        if result is not None and result['c'] not in (0, max_job_id):
            return result

        # Если fallback не сработал или вернул фиктивную работу — строим граф только на реальных работах
        n = len(self.jobs)
        job_ids = list(self.jobs.keys())
        idx_map = {jid: i for i, jid in enumerate(job_ids)}

        N = n + 2
        o = n
        t_node = n + 1

        dist = [[-float('inf')] * N for _ in range(N)]

        for i, jid in enumerate(job_ids):
            dist[o][i] = data['r'][jid]

        # Строим граф только для реальных работ
        for idx_i, i in enumerate(real_schedule):
            for idx_j, j in enumerate(real_schedule):
                if idx_i < idx_j:
                    lij = l_matrix[i][j]
                    if lij > 0:
                        dist[idx_map[i]][idx_map[j]] = max(dist[idx_map[i]][idx_map[j]], lij)
                    else:
                        dist[idx_map[i]][idx_map[j]] = max(dist[idx_map[i]][idx_map[j]],
                                                           self.jobs[i].d_i)

        for i, jid in enumerate(job_ids):
            dist[i][t_node] = self.jobs[jid].d_i + data['q'][jid]

        # Топологический порядок
        topo = [o] + [idx_map[job] for job in real_schedule] + [t_node]

        longest = [-float('inf')] * N
        longest[o] = 0
        pred = [-1] * N

        for u in topo:
            if longest[u] == -float('inf'):
                continue
            for v in range(N):
                if dist[u][v] > -float('inf'):
                    if longest[u] + dist[u][v] > longest[v] + 1e-6:
                        longest[v] = longest[u] + dist[u][v]
                        pred[v] = u

        if longest[t_node] < C_max - 1e-6:
            # Ищем реальную работу с максимальным completion + q
            best_c = None
            best_val = 0
            for j in real_schedule:
                val = start_times[j] + self.jobs[j].d_i + data['q'][j]
                if val > best_val + 1e-6:
                    best_val = val
                    best_c = j
            if best_c is None:
                return None
            # Находим все работы после best_c в расписании
            c_index = real_schedule.index(best_c) if best_c in real_schedule else 0
            J = set(real_schedule[c_index + 1:])
            return {
                'critical_path': real_schedule,
                'c': best_c,
                'J': J,
                'start_times': start_times
            }

        # Восстанавливаем путь
        path = []
        v = t_node
        while v != o:
            u = pred[v]
            if u == -1:
                break
            if u != o and u != t_node:
                path.insert(0, job_ids[u])
            v = u

        if not path:
            return None

        # Ищем первую реальную работу в пути
        real_c = None
        for jid in path:
            if jid not in (0, max_job_id):
                real_c = jid
                break

        if real_c is None:
            return None

        c_index = path.index(real_c) if real_c in path else 0
        J = set(path[c_index + 1:])
        J = {j for j in J if j not in (0, max_job_id)}

        return {
            'critical_path': path,
            'c': real_c,
            'J': J,
            'start_times': start_times
        }

    def _find_critical_path_fallback(self, schedule: List[int], start_times: Dict[int, float],
                                     C_max: float, data: Dict, l_matrix=None) -> Optional[Dict]:
        if l_matrix is None:
            l_matrix = self.l_matrix

        if not schedule:
            return None

        q = data['q']
        r = data['r']

        pos_in_schedule = {job: idx for idx, job in enumerate(schedule)}

        current = None
        for j in reversed(schedule):
            completion = start_times[j] + self.jobs[j].d_i
            if abs(completion + q[j] - C_max) < 1e-6:
                current = j
                break

        if current is None:
            return None

        critical_path = [current]
        while True:
            found = False
            curr_idx = pos_in_schedule[current]

            for pred in schedule[:curr_idx]:
                pred_end = start_times[pred] + self.jobs[pred].d_i
                if abs(pred_end - start_times[current]) < 1e-6:
                    current = pred
                    critical_path.insert(0, current)
                    found = True
                    break
                lij = l_matrix[pred][current]
                if lij > 0:
                    if abs(start_times[pred] + lij - start_times[current]) < 1e-6:
                        current = pred
                        critical_path.insert(0, current)
                        found = True
                        break
            if not found:
                break

        c = None
        for job_id in critical_path:
            if r[job_id] < start_times[job_id] - 1e-6:
                c = job_id
                break
        if c is None:
            c = critical_path[0] if critical_path else None

        if c is None:
            return None

        c_index = critical_path.index(c)
        J = set(critical_path[c_index + 1:])

        return {
            'critical_path': critical_path,
            'c': c,
            'J': J,
            'start_times': start_times
        }

    def _check_strong_branching_conditions(self, critical_info: Dict, data: Dict,
                                           start_times: Dict, C_max: float,
                                           l_matrix=None) -> bool:
        """Проверяет условия теоремы 3.1 для всего сегмента C(c,n)."""
        if l_matrix is None:
            l_matrix = self.l_matrix

        critical_path = critical_info['critical_path']
        c = critical_info['c']
        J = critical_info['J']
        r = data['r']

        try:
            c_index = critical_path.index(c)
        except ValueError:
            return False

        segment = critical_path[c_index:]

        for i in range(len(segment) - 1):
            u, v = segment[i], segment[i + 1]
            if self._is_precedence_arc(u, v, data, start_times, C_max, l_matrix):
                return False

        t_c = start_times[c]
        for i in J:
            t_i = start_times[i]
            if r[i] < max(t_i, t_c) - 1e-6:
                return False

        return True

    def _branch_and_bound(self, data: Dict, upper_bound: float, depth: int, history: Set[FrozenSet]) -> None:
        self.nodes_explored += 1

        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return

        if depth > 3 * self.n:
            return

        # Проверка на повторяющиеся состояния
        state_key = self._get_state_key(data)
        if state_key in history:
            return
        history.add(state_key)

        lb = self._calculate_lower_bound(data)

        self._debug_print(f"\n{'=' * 60}")
        self._debug_print(f"Depth {depth:2}: nodes={self.nodes_explored:5}, LB={lb:8.2f}, UB={upper_bound:8.2f}")

        if lb >= upper_bound - 1e-6:
            self._debug_print(f"  -> Pruned by bound")
            self.pruned_by_bound += 1
            return

        schedule, makespan, start_times = self._longest_tail_heuristic(data, self.l_matrix)
        self._debug_print(f"  -> LTH makespan: {makespan:.2f}")

        upper_bound = self._update_best(schedule, makespan, data, upper_bound)

        if lb >= makespan - 1e-6:
            self._debug_print(f"  -> LB >= makespan, optimal for this branch")
            return

        # Постпроцессинг
        while True:
            changed = self._postprocess(data, schedule, start_times, makespan, self.l_matrix)
            if not changed:
                break
            schedule, makespan, start_times = self._longest_tail_heuristic(data, self.l_matrix)
            upper_bound = self._update_best(schedule, makespan, data, upper_bound)

        critical_info = self._find_critical_path_via_graph(schedule, start_times, makespan, data, self.l_matrix)
        max_job_id = max(self.jobs.keys())

        self._debug_print(f"  -> critical_info: {critical_info is not None}")
        if critical_info is not None and self.debug:
            self._debug_print(f"  -> c={critical_info.get('c')}, J_size={len(critical_info.get('J', set()))}")

        # Если нет критического пути или c фиктивный - возвращаем
        if critical_info is None:
            self._debug_print(f"  -> No critical path found, returning")
            return

        c = critical_info['c']
        J = critical_info['J']

        if c == 0 or c == max_job_id:
            self._debug_print(f"  -> c is dummy job, returning")
            return

        if not J:
            self._debug_print(f"  -> J is empty, returning")
            return

        self._debug_print(f"  -> final c={c}, J={sorted(J)[:5]}...")

        can_use_strong = self._check_strong_branching_conditions(
            critical_info, data, start_times, makespan, self.l_matrix
        )
        self._debug_print(f"  -> can_use_strong: {can_use_strong}")

        if can_use_strong:
            self.strong_branches += 1
            self._apply_strong_branching(data, c, J, upper_bound, depth + 1, history)
        else:
            self._debug_print(f"  -> trying reverse problem...")
            rev_schedule, rev_makespan, rev_starts, rev_critical = self._solve_reverse_problem(
                data, upper_bound, depth, history
            )

            if rev_critical and rev_critical.get('c') not in (0, max_job_id):
                if self._check_strong_branching_conditions(
                        rev_critical, data, rev_starts, rev_makespan, self.l_matrix
                ):
                    self.strong_branches += 1
                    rev_c = rev_critical['c']
                    rev_J = rev_critical['J']
                    self._debug_print(f"  -> reverse strong branching: c={rev_c}")
                    self._apply_strong_branching(data, rev_c, rev_J, upper_bound, depth + 1, history)
                    return

            # Weak branching
            self.weak_branches += 1
            self._debug_print(f"  -> weak branching")
            self._apply_weak_branching(data, critical_info, upper_bound, depth + 1, history)

    def _get_state_key(self, data: Dict) -> FrozenSet:
        """Создает уникальный ключ для состояния."""
        # Используем frozenset из пар (i, j) для sigma
        sigma_pairs = frozenset((i, j) for i, next_set in data['sigma'].items() for j in next_set)
        # Также учитываем r и q (округляем для стабильности)
        r = data['r']
        q = data['q']
        r_tuples = frozenset((i, round(r[i], 2)) for i in sorted(r.keys()))
        q_tuples = frozenset((i, round(q[i], 2)) for i in sorted(q.keys()))
        return frozenset([sigma_pairs, r_tuples, q_tuples])

    def _solve_reverse_problem(self, data: Dict, upper_bound: float, depth: int, history: Set[FrozenSet]):
        """
        Решает обратную задачу: инвертирует DPC и направления предшествования,
        затем вызывает LTH, разворачивает расписание и возвращает его в исходной ориентации.
        """
        job_ids = list(self.jobs.keys())

        # Строим инвертированную матрицу DPC
        rev_l_matrix = self._get_reverse_l_matrix()

        # Создаём обратные данные: меняем r и q местами
        rev_data = {
            'r': {j: data['q'][j] for j in job_ids},
            'q': {j: data['r'][j] for j in job_ids},
            'sigma': defaultdict(set),
            'pi': defaultdict(set)
        }
        # Инвертируем предшествования из data['sigma']/['pi']
        for i in job_ids:
            for j in data['sigma'].get(i, set()):
                rev_data['sigma'][j].add(i)
                rev_data['pi'][i].add(j)

        # Также добавляем в rev_data все DPC как жёсткие предшествования
        for i in job_ids:
            for j in job_ids:
                if rev_l_matrix[i][j] > 0:
                    rev_data['sigma'][i].add(j)
                    rev_data['pi'][j].add(i)

        # Запускаем LTH на обратной задаче
        rev_schedule, rev_makespan, rev_starts = self._longest_tail_heuristic(rev_data, rev_l_matrix)

        # Разворачиваем расписание, чтобы получить прямой порядок
        rev_schedule.reverse()

        # Пересчитываем start_times и C_max для прямого расписания
        forward_start = {}
        current = 0.0
        for job in rev_schedule:
            start = max(data['r'][job], current)
            # Учитываем DPC от уже запланированных (в прямом порядке)
            for pred in rev_schedule:
                if pred == job:
                    break
                lij = self.l_matrix[pred][job]
                if lij > 0:
                    start = max(start, forward_start[pred] + self.jobs[pred].d_i + lij)
            forward_start[job] = start
            current = start + self.jobs[job].d_i

        forward_makespan = max(forward_start[j] + self.jobs[j].d_i + data['q'][j] for j in rev_schedule)

        # Анализируем критический путь уже на прямом расписании
        critical_info = self._find_critical_path_via_graph(rev_schedule, forward_start, forward_makespan, data)

        return rev_schedule, forward_makespan, forward_start, critical_info

    def _apply_weak_branching(self, data: Dict, critical_info: Dict, upper_bound: float,
                              depth: int, history: Set[FrozenSet]) -> None:
        i, j = self._select_weak_branching_pair(data, critical_info)
        if i is None or j is None:
            return

        sigma = data['sigma']
        pi = data['pi']

        i_before_j_possible = True
        j_before_i_possible = True

        if j in sigma.get(i, set()) or i in pi.get(j, set()) or self.l_matrix[i][j] > 0:
            j_before_i_possible = False
        if i in sigma.get(j, set()) or j in pi.get(i, set()) or self.l_matrix[j][i] > 0:
            i_before_j_possible = False

        if not i_before_j_possible and not j_before_i_possible:
            return

        critical_path = critical_info['critical_path']
        try:
            idx_i = critical_path.index(i)
            idx_j = critical_path.index(j)
            i_before_j_in_path = idx_i < idx_j
        except ValueError:
            i_before_j_in_path = False

        # ВАЖНО: сначала пробуем более перспективную ветвь (соответствующую критическому пути)
        if i_before_j_possible and i_before_j_in_path:
            self._branch_on_relation(data, i, j, upper_bound, depth, history)
            if j_before_i_possible:
                self._branch_on_relation(data, j, i, upper_bound, depth, history)
        elif j_before_i_possible:
            self._branch_on_relation(data, j, i, upper_bound, depth, history)
            if i_before_j_possible:
                self._branch_on_relation(data, i, j, upper_bound, depth, history)
        elif i_before_j_possible:
            self._branch_on_relation(data, i, j, upper_bound, depth, history)

    def _branch_on_relation(self, data: Dict, first: int, second: int,
                            upper_bound: float, depth: int, history: Set[FrozenSet]) -> None:
        """Создаёт новую ветвь с добавленным отношением first -> second."""
        new_data = self._copy_data(data)
        if not self._add_precedence(new_data['sigma'], new_data['pi'], first, second):
            return

        lij = self.l_matrix[first][second]
        # Исправление: lij уже включает d_i, не добавляем его повторно
        new_data['r'][second] = max(new_data['r'][second],
                                    new_data['r'][first] + lij)
        new_data['q'][first] = max(new_data['q'][first],
                                   new_data['q'][second] + lij - self.jobs[first].d_i)

        self._branch_and_bound(new_data, upper_bound, depth, history)

    def _apply_strong_branching(self, data: Dict, c: int, J: Set[int], upper_bound: float,
                                depth: int, history: Set[FrozenSet]) -> None:
        """Применяет сильное ветвление: c -> J и J -> c."""
        max_job_id = max(self.jobs.keys())

        if c == 0 or c == max_job_id:
            return

        if not J:
            return

        if self._carlier_tests(data, c, J, upper_bound):
            self.pruned_by_test += 1
            return

        # Ветвь 1: c -> J
        data1 = self._copy_data(data)
        valid_branch1 = True
        for j in J:
            if not self._add_precedence(data1['sigma'], data1['pi'], c, j):
                valid_branch1 = False
                break
            lij = self.l_matrix[c][j]
            # Исправление: lij уже включает d_i
            data1['r'][j] = max(data1['r'][j], data1['r'][c] + lij)

        if valid_branch1:
            self._branch_and_bound(data1, upper_bound, depth, history)

        # Ветвь 2: J -> c
        data2 = self._copy_data(data)
        valid_branch2 = True
        for j in J:
            if not self._add_precedence(data2['sigma'], data2['pi'], j, c):
                valid_branch2 = False
                break
            lij = self.l_matrix[j][c]
            # Исправление: lij уже включает d_i
            data2['q'][j] = max(data2['q'][j],
                                data2['q'][c] + lij - self.jobs[j].d_i)

        if valid_branch2:
            self._branch_and_bound(data2, upper_bound, depth, history)

    def _carlier_tests(self, data: Dict, c: int, J: Set[int], upper_bound: float) -> bool:
        """Тесты Карлье для отсечения ветвей."""
        r = data['r']
        q = data['q']

        if not J:
            return False

        min_r_J = min(r[j] for j in J)
        sum_d_J = sum(self.jobs[j].d_i for j in J)
        min_q_J = min(q[j] for j in J)
        h_J = min_r_J + sum_d_J + min_q_J

        all_jobs = set(self.jobs.keys())
        K = {k for k in all_jobs - J - {c}
             if self.jobs[k].d_i > upper_bound - h_J}

        # Тест 1: k должен быть после J
        for k in K:
            if r[c] + self.jobs[c].d_i + sum_d_J + q[k] >= upper_bound - 1e-6:
                return True

        # Тест 2: k должен быть перед J
        for k in K:
            if min_r_J + sum_d_J + self.jobs[k].d_i + q[k] >= upper_bound - 1e-6:
                return True

        return False

    def _select_weak_branching_pair(self, data: Dict, critical_info: Dict) -> Tuple[Optional[int], Optional[int]]:
        critical_path = critical_info['critical_path']
        r = data['r']
        start_times = critical_info['start_times']
        c = critical_info['c']
        C_max = self.best_makespan
        max_job_id = max(self.jobs.keys())

        try:
            c_index = critical_path.index(c)
        except ValueError:
            return None, None

        segment = critical_path[c_index:]

        essential_arc = None
        for idx in range(len(segment) - 1, 0, -1):
            u, v = segment[idx - 1], segment[idx]
            if u not in (0, max_job_id) and v not in (0, max_job_id):
                if self._is_essential_precedence_arc(u, v, data, start_times, C_max):
                    essential_arc = (u, v)
                    break

        if essential_arc is None:
            j = c if c not in (0, max_job_id) else critical_path[0]
            # Пропускаем фиктивные работы
            if j in (0, max_job_id):
                for node in segment:
                    if node not in (0, max_job_id):
                        j = node
                        break
                else:
                    return None, None

            t_j = start_times[j]
            for node in segment:
                if node not in (0, max_job_id) and r[node] < max(start_times[node], t_j) - 1e-6:
                    if not self._are_ordered(node, j, data) and not self._are_ordered(j, node, data):
                        return node, j
            return c if c not in (0, max_job_id) else j, j
        else:
            k, l = essential_arc
            j = l
            try:
                l_index = critical_path.index(l)
            except ValueError:
                return None, None
            segment_l = critical_path[l_index:]
            for node in segment_l:
                if node not in (0, max_job_id) and node != l and r[node] < start_times[l] - 1e-6:
                    if not self._are_ordered(node, j, data) and not self._are_ordered(j, node, data):
                        return node, j
            if len(segment_l) > 1:
                for node in segment_l:
                    if node not in (0, max_job_id):
                        return node, j
            return None, None

    def _are_ordered(self, i: int, j: int, data: Dict) -> bool:
        sigma = data['sigma']
        pi = data['pi']
        if j in sigma.get(i, set()) or i in pi.get(j, set()):
            return True
        if self.l_matrix[i][j] > 0:
            return True
        return False

    def _postprocess(self, data: Dict, schedule: List[int],
                     start_times: Dict[int, float], C_max: float, l_matrix=None) -> bool:
        if l_matrix is None:
            l_matrix = self.l_matrix

        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']
        pi = data['pi']

        # Proposition 3.3: обновление r
        for idx, j in enumerate(schedule):
            segment_has_prec = False
            for k in range(idx, len(schedule) - 1):
                u, v = schedule[k], schedule[k + 1]
                if self._is_essential_precedence_arc(u, v, data, start_times, C_max):
                    segment_has_prec = True
                    break
            if segment_has_prec:
                continue

            K = schedule[idx + 1:]
            if not K:
                continue

            q_j = q[j]
            t_j = start_times[j]
            conditions_hold = True
            for k in K:
                if q[k] < q_j - 1e-6 or r[k] < t_j - 1e-6:
                    conditions_hold = False
                    break

            if conditions_hold:
                for k in K:
                    new_r = t_j + self.jobs[j].d_i + l_matrix[j][k]
                    if new_r > r[k] + 1e-6:
                        r[k] = new_r
                        changed = True
                    if j not in sigma.get(k, set()):
                        self._add_precedence(sigma, pi, j, k)
                        changed = True

        # Proposition 3.4: обновление q
        for idx in range(len(schedule) - 1, -1, -1):
            i = schedule[idx]
            segment_has_prec = False
            for k in range(idx):
                u, v = schedule[k], schedule[k + 1]
                if self._is_essential_precedence_arc(u, v, data, start_times, C_max):
                    segment_has_prec = True
                    break
            if segment_has_prec:
                continue

            K = schedule[:idx]
            if not K:
                continue

            r_i = r[i]
            t_i = start_times[i]
            conditions_hold = True
            for k in K:
                if r[k] < r_i - 1e-6 or q[k] < q[i] - 1e-6:
                    conditions_hold = False
                    break

            if conditions_hold:
                for k in K:
                    new_q = q[i] + self.jobs[i].d_i + l_matrix[k][i] - self.jobs[k].d_i
                    if new_q > q[k] + 1e-6:
                        q[k] = new_q
                        changed = True
                    if k not in sigma.get(i, set()):
                        self._add_precedence(sigma, pi, k, i)
                        changed = True

        return changed

    def _calculate_lower_bound(self, data: Dict) -> float:
        r = data['r']
        q = data['q']

        if not self.jobs:
            return 0.0

        if not hasattr(self, '_total_d'):
            self._total_d = sum(j.d_i for j in self.jobs.values())

        min_r = min(r.values())
        min_q = min(q.values())
        lb1 = min_r + self._total_d + min_q

        lb2 = 0.0
        if self.l_matrix:
            for i in self.jobs:
                for j in self.jobs:
                    lij = self.l_matrix[i][j]
                    if lij > 0:
                        path_length = r[i] + self.jobs[i].d_i + lij + q[j]
                        if path_length > lb2:
                            lb2 = path_length

        return max(lb1, lb2)

    def _longest_tail_heuristic(self, data: Dict, l_matrix=None) -> Tuple[List[int], float, Dict[int, float]]:
        if l_matrix is None:
            l_matrix = self.l_matrix

        r = data['r'].copy()
        q = data['q'].copy()
        pi = data['pi']

        unscheduled = set(self.jobs.keys())
        schedule = []
        start_times = {}
        current_time = 0.0

        incoming_dpc = self._build_incoming_dpc(l_matrix)

        while unscheduled:
            best_job = None
            best_q = -float('inf')
            best_start = float('inf')

            for job_id in unscheduled:
                preds = pi.get(job_id)
                if preds and any(p in unscheduled for p in preds):
                    continue

                # Вычисляем earliest start с учётом уже запланированных работ и DPC
                start = r[job_id]
                dpc_list = incoming_dpc.get(job_id)
                if dpc_list:
                    for s_id, lij in dpc_list:
                        if s_id in start_times:
                            start = max(start, start_times[s_id] + lij)
                start = max(start, current_time)

                q_val = q[job_id]
                if q_val > best_q or (abs(q_val - best_q) < 1e-6 and start < best_start):
                    best_q = q_val
                    best_job = job_id
                    best_start = start

            if best_job is None:
                # Fallback: берём любую доступную
                for job_id in unscheduled:
                    if not pi.get(job_id) or all(p not in unscheduled for p in pi[job_id]):
                        best_job = job_id
                        best_start = max(r[job_id], current_time)
                        break

            start_times[best_job] = best_start
            schedule.append(best_job)
            current_time = best_start + self.jobs[best_job].d_i
            unscheduled.remove(best_job)

            # Обновляем r для оставшихся работ с учётом новой запланированной
            for other in unscheduled:
                lij = l_matrix[best_job][other]
                if lij > 0:
                    # Исправление: lij уже включает d_i, не добавляем его повторно
                    r[other] = max(r[other], best_start + lij)

        C_max = 0.0
        for j in schedule:
            delivery_end = start_times[j] + self.jobs[j].d_i + q[j]
            C_max = max(C_max, delivery_end)

        return schedule, C_max, start_times

    def _create_reverse_problem(self, data: Dict) -> Dict:
        r = data['r']
        q = data['q']
        return {
            'r': {j: q[j] for j in self.jobs.keys()},
            'q': {j: r[j] for j in self.jobs.keys()},
            'sigma': defaultdict(set, {k: v.copy() for k, v in data['pi'].items()}),
            'pi': defaultdict(set, {k: v.copy() for k, v in data['sigma'].items()}),
        }

    def _get_reverse_l_matrix(self):
        """Создаёт обратную матрицу DPC."""
        rev = defaultdict(lambda: defaultdict(float))
        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0:
                    rev[j][i] = self.l_matrix[i][j] - self.jobs[i].d_i + self.jobs[j].d_i
        return rev

    def _copy_data(self, data: Dict) -> Dict:
        return {
            'r': data['r'].copy(),
            'q': data['q'].copy(),
            'sigma': defaultdict(set, {k: v.copy() for k, v in data['sigma'].items()}),
            'pi': defaultdict(set, {k: v.copy() for k, v in data['pi'].items()}),
        }