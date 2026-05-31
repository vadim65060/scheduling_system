"""
Точный алгоритм ветвей и границ для задачи 1|r_j, q_j, DPC|C_max
Основан на статье Balas, Lenstra, Vazacopoulos (1995).
Строгая реализация Algorithm 1 (Bal) с исправлениями.
"""

import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy

from algorithms.BestOfHeuristics import BestOfHeuristics
from core.Algorithm import Algorithm
from core.job import Job


class BalasBaBDPC(Algorithm):
    """
    Алгоритм ветвей и границ для задачи одного станка
    с отложенными ограничениями предшествования (DPC).
    Реализует Algorithm 1 (Bal) из статьи Balas et al. (1995).
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 time_limit: float = 60.0):
        super().__init__(jobs, precedence_constraints)
        self._debug_count = 0
        self._debug_limit = 0
        self.init_jobs = jobs
        self.init_precedence = precedence_constraints
        self.time_limit = time_limit
        self.start_time = 0.0

        self.best_schedule: Optional[List[int]] = None
        self.best_makespan: float = float('inf')
        self.timed_out: bool = False

        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0

        # Вычисляем транзитивное замыкание DPC.2
        self._compute_transitive_dpc()

        # Списки входящих/исходящих DPC для быстрого доступа
        self._incoming_dpc = defaultdict(list)
        self._outgoing_dpc = defaultdict(list)
        for (i, j) in self.dpc_pairs:
            if i in self.jobs and j in self.jobs:
                lij = self.l_matrix[i][j]
                self._incoming_dpc[j].append((i, lij))
                self._outgoing_dpc[i].append((j, lij))

        # Кэшируем сумму длительностей
        self._total_d = sum(j.d_i for j in self.jobs.values())

    # =====================================================================
    # ГЛАВНЫЙ МЕТОД
    # =====================================================================

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Запускает алгоритм ветвей и границ.
        Возвращает (расписание, C_max, статистика).
        """
        self.start_time = time.time()
        self.timed_out = False

        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0

        # --- Шаг 1. Вычисление начальной верхней границы ---
        boh = BestOfHeuristics(deepcopy(self.init_jobs), deepcopy(self.init_precedence))
        boh_schedule, init_makespan, _ = boh.solve(**kwargs)

        self.best_makespan = init_makespan
        self.best_schedule = boh_schedule.copy() if boh_schedule else None
        upper_bound = init_makespan

        # --- Шаг 2. Начальное усиление голов и хвостов ---
        initial_data = self._initialize_data()
        self._update_heads_and_tails(initial_data)

        # --- Шаг 3. Запуск ветвей и границ ---
        self._branch_and_bound(initial_data, upper_bound, depth=0)

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
            'initial_makespan': init_makespan,
            'improvement': init_makespan - self.best_makespan
            if self.best_makespan < float('inf') else 0,
            'optimal': (self.best_schedule is not None and not self.timed_out)
        }

        return self.best_schedule, self.best_makespan, stats

    # =====================================================================
    # ИНИЦИАЛИЗАЦИЯ И УСИЛЕНИЕ
    # =====================================================================

    def _initialize_data(self) -> Dict:
        """Создаёт начальные данные для корневой вершины."""
        return {
            'r': {j.id: j.r_i for j in self.init_jobs},
            'q': {j.id: j.q_i for j in self.init_jobs},
            'sigma': {},
            'pi': {},
        }

    def _update_heads_and_tails(self, data: Dict) -> bool:
        """Итеративно усиливает головы (r) и хвосты (q) до фиксированной точки."""
        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']

        # Создаем множество всех известных отношений
        sigma_set = set()
        for i, next_set in sigma.items():
            for j in next_set:
                sigma_set.add((i, j))

        for _ in range(self.n):
            local_changed = False

            # Обновление на основе DPC
            for (i, j) in self.dpc_pairs:
                lij = self.l_matrix[i][j]
                new_rj = r[i] + lij
                if new_rj > r[j] + 1e-9:
                    r[j] = new_rj
                    local_changed = True

                new_qi = q[j] + lij
                if new_qi > q[i] + 1e-9:
                    q[i] = new_qi
                    local_changed = True

            # Обновление на основе sigma
            for (i, j) in sigma_set:
                delay = self.jobs[i].d_i
                if (i, j) in self.dpc_pairs:
                    delay = self.l_matrix[i][j]

                new_rj = r[i] + delay
                if new_rj > r[j] + 1e-9:
                    r[j] = new_rj
                    local_changed = True

                new_qi = q[j] + delay
                if new_qi > q[i] + 1e-9:
                    q[i] = new_qi
                    local_changed = True

            if local_changed:
                changed = True
            else:
                break

        return changed

    # =====================================================================
    # ОСНОВНОЙ ЦИКЛ ВЕТВЕЙ И ГРАНИЦ (Algorithm 1)
    # =====================================================================

    def _branch_and_bound(self, data: Dict, upper_bound: float, depth: int) -> None:
        """Основная рекурсивная процедура Algorithm 1 (Bal)."""
        self.nodes_explored += 1

        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return

        if depth > 3 * self.n:
            return

        # Шаг 1: Вычисление нижней границы
        lb = self._calculate_lower_bound(data)
        if lb >= upper_bound - 1e-6:
            self.pruned_by_bound += 1
            return

        # Шаг 2: Построение расписания эвристикой LTH
        schedule, makespan, start_times = self._longest_tail_heuristic(data)

        if makespan < self.best_makespan - 1e-9:
            self.best_makespan = makespan
            self.best_schedule = schedule.copy()
            upper_bound = min(upper_bound, makespan)
            if lb >= makespan - 1e-6:
                return

        # Шаг 3: Постобработка
        max_pp_iter = self.n
        for _ in range(max_pp_iter):
            changed = self._postprocess(data, schedule, start_times, makespan)
            if not changed:
                break
            schedule, makespan, start_times = self._longest_tail_heuristic(data)
            if makespan < self.best_makespan - 1e-9:
                self.best_makespan = makespan
                self.best_schedule = schedule.copy()
                upper_bound = min(upper_bound, makespan)

        # Отладка
        if self._debug_count < self._debug_limit:
            self._debug_count += 1
            print(f"\n{'='*70}")
            print(f"ОТЛАДКА УЗЛА #{self._debug_count}")
            print(f"Depth: {depth}, UB: {upper_bound:.2f}")
            critical_info = self._find_critical_path(schedule, start_times, data)

            if critical_info is None:
                print("❌ critical_info is None!")
            else:
                cp = critical_info['critical_path']
                c = critical_info['c']
                J = critical_info['J']
                c_index = critical_info.get('c_index', -1)

                print(f"\n📊 КРИТИЧЕСКИЙ ПУТЬ:")
                print(f"  Путь: {cp}")
                print(f"  c = {c} (индекс {c_index})")
                print(f"  J = {J}")

                if c != 0:
                    print(f"\n🔍 ПРОВЕРКА УСЛОВИЙ СИЛЬНОГО ВЕТВЛЕНИЯ:")
                    segment = cp[c_index:]
                    print(f"  Сегмент C(c,n): {segment}")
                    has_prec = False
                    for k in range(len(segment) - 1):
                        u, v = segment[k], segment[k + 1]
                        is_dpc = (u, v) in self.dpc_pairs
                        is_sigma = v in data['sigma'].get(u, set())
                        if is_dpc or is_sigma:
                            has_prec = True
                            print(f"  ❌ Найдена дуга: ({u},{v}) DPC={is_dpc} SIGMA={is_sigma}")
                    if not has_prec:
                        print(f"  ✅ Условие 1 выполнено: нет precedence дуг")

                    t_c = start_times[c]
                    cond2_ok = True
                    for i in J:
                        t_i = start_times[i]
                        r_i = data['r'][i]
                        if r_i < max(t_i, t_c) - 1e-9:
                            cond2_ok = False
                            print(f"  ❌ Условие 2 нарушено для {i}: r_{i}={r_i} < max(t_{i}={t_i}, t_{c}={t_c})")
                    if cond2_ok:
                        print(f"  ✅ Условие 2 выполнено")

                    if not has_prec and cond2_ok:
                        print(f"\n🎯 СИЛЬНОЕ ВЕТВЛЕНИЕ ДОЛЖНО СРАБОТАТЬ!")
                else:
                    print(f"\n⚠️ c=0 — все работы на критическом пути начинаются вовремя")

        # Шаг 4: Ветвление
        critical_info = self._find_critical_path(schedule, start_times, data)
        if critical_info is None:
            return

        c = critical_info['c']
        J = critical_info['J']

        if c == 0:
            return

        if self._check_strong_branching_conditions(critical_info, data, start_times):
            self.strong_branches += 1
            self._apply_strong_branching(data, c, J, upper_bound, depth + 1)
        else:
            if self._try_reverse_strong_branching(data, upper_bound, depth + 1):
                self.strong_branches += 1
            else:
                self.weak_branches += 1
                self._apply_weak_branching(data, critical_info, upper_bound, depth + 1)

    # =====================================================================
    # НИЖНЯЯ ГРАНИЦА
    # =====================================================================

    def _calculate_lower_bound(self, data: Dict) -> float:
        """
        Вычисляет прерываемую нижнюю границу (preemptive bound).
        Реализует max_{K ⊆ U} (min_{i∈K} r_i + sum_{i∈K} d_i + min_{i∈K} q_i)
        за O(n log n) через решение прерываемой задачи (Carlier 1982).
        """
        r = data['r']
        q = data['q']

        # Получаем список всех работ
        jobs_list = list(self.jobs.values())
        n = len(jobs_list)

        if n == 0:
            return 0.0

        # Сортируем работы по убыванию q (tail)
        jobs_sorted = sorted(jobs_list, key=lambda j: q[j.id], reverse=True)

        # Перебираем все возможные "хвостовые" подмножества
        # Оптимальное подмножество для прерываемой границы имеет вид:
        # K* = {j : q_j >= q_threshold} для некоторого порога
        best_lb = 0.0

        # Вычисляем префиксные суммы d и поддерживаем min_r
        sum_d = 0.0
        min_r_sofar = float('inf')

        for j in jobs_sorted:
            j_id = j.id
            sum_d += j.d_i
            min_r_sofar = min(min_r_sofar, r[j_id])

            # Текущий кандидат: K = первые k работ (с наибольшими q)
            current_lb = min_r_sofar + sum_d + q[j_id]

            if current_lb > best_lb:
                best_lb = current_lb

        # Дополнительная проверка: рассмотреть только min_r + sum_d (без q)
        # Это соответствует K, где min q = 0 (нижняя граница по времени обработки)
        min_r_all = min(r.values())
        sum_d_all = sum(j.d_i for j in self.jobs.values())
        lb_processing = min_r_all + sum_d_all
        if lb_processing > best_lb:
            best_lb = lb_processing

        # Учёт DPC через усиление r и q уже сделано в _update_heads_and_tails
        # Поэтому дополнительный перебор пар не требуется — он слабее, чем max по K

        return best_lb

    # =====================================================================
    # ЭВРИСТИКА LTH (Algorithm 2)
    # =====================================================================

    def _longest_tail_heuristic(self, data: Dict) -> Tuple[List[int], float, Dict[int, float]]:
        r = data['r'].copy()
        q = data['q'].copy()
        sigma = data['sigma']

        unscheduled = set(self.jobs.keys())
        schedule = []
        start_times = {}
        current_time = 0.0

        # Счётчики невыполненных предшественников
        remaining_preds = {}
        for j in self.jobs:
            count = 0
            for (i, _) in self._incoming_dpc.get(j, []):
                if i in self.jobs:
                    count += 1
            for pred in data['pi'].get(j, set()):
                if pred in self.jobs:
                    count += 1
            remaining_preds[j] = count

        ready = {j for j in unscheduled if remaining_preds[j] == 0}

        while unscheduled:
            if not ready:
                break

            released = {j for j in ready if r[j] <= current_time + 1e-9}

            if released:
                k = max(released, key=lambda j: q[j])
                s_k = max(current_time, r[k])
            else:
                min_r = min(r[j] for j in ready)
                candidates = {j for j in ready if abs(r[j] - min_r) < 1e-9}
                k = max(candidates, key=lambda j: q[j])
                s_k = max(current_time, r[k])

            start_times[k] = s_k
            schedule.append(k)
            current_time = s_k + self.jobs[k].d_i
            unscheduled.remove(k)
            ready.remove(k)

            # Обновляем последователей
            for (succ_id, lij) in self._outgoing_dpc.get(k, []):
                if succ_id in unscheduled:
                    remaining_preds[succ_id] -= 1
                    if remaining_preds[succ_id] == 0:
                        ready.add(succ_id)
                    new_r = s_k + lij
                    if new_r > r[succ_id]:
                        r[succ_id] = new_r

            for succ_id in sigma.get(k, set()):
                if succ_id in unscheduled:
                    remaining_preds[succ_id] -= 1
                    if remaining_preds[succ_id] == 0:
                        ready.add(succ_id)
                    new_r = s_k + self.jobs[k].d_i
                    if new_r > r[succ_id]:
                        r[succ_id] = new_r

        C_max = 0.0
        for j in schedule:
            completion_with_delivery = start_times[j] + self.jobs[j].d_i + q[j]
            if completion_with_delivery > C_max:
                C_max = completion_with_delivery

        return schedule, C_max, start_times

    # =====================================================================
    # ПОИСК КРИТИЧЕСКОГО ПУТИ (ИСПРАВЛЕНО)
    # =====================================================================

    def _find_critical_path(self, schedule: List[int], start_times: Dict[int, float],
                            data: Dict) -> Optional[Dict]:
        if not schedule:
            return None

        n_jobs = len(schedule)
        SOURCE = n_jobs
        SINK = n_jobs + 1

        dist = [-float('inf')] * (n_jobs + 2)
        parent = [-1] * (n_jobs + 2)
        dist[SOURCE] = 0.0

        # --- Шаг 1: Дуги от истока ко всем работам (вес = r_i) ---
        for idx, j in enumerate(schedule):
            w = data['r'][j]
            if w > dist[idx]:
                dist[idx] = w
                parent[idx] = SOURCE

        # --- Шаг 2: Обработка работ в порядке расписания ---
        for idx in range(n_jobs):
            if dist[idx] == -float('inf'):
                continue
            i = schedule[idx]

            # 2a: Дуги КО ВСЕМ последующим работам в расписании
            for jdx in range(idx + 1, n_jobs):
                j = schedule[jdx]
                # Определяем вес дуги: если есть DPC, используем l_ij, иначе d_i
                if (i, j) in self.dpc_pairs:
                    weight = self.l_matrix[i][j]
                else:
                    weight = self.jobs[i].d_i
                w = dist[idx] + weight
                if w > dist[jdx] + 1e-9:
                    dist[jdx] = w
                    parent[jdx] = idx

            # 2b: Дуга от работы к стоку (вес = d_i + q_i)
            w = dist[idx] + self.jobs[i].d_i + data['q'][i]
            if w > dist[SINK] + 1e-9:
                dist[SINK] = w
                parent[SINK] = idx

        # --- Шаг 3: Восстановление критического пути ---
        if dist[SINK] == -float('inf'):
            return None

        critical_path = []
        cur = SINK
        while cur != SOURCE:
            p = parent[cur]
            if p < 0 or p == cur:
                break
            if p < n_jobs:
                critical_path.append(schedule[p])
            cur = p
        critical_path.reverse()

        if not critical_path:
            return {
                'critical_path': [],
                'c': 0,
                'J': set(),
                'start_times': start_times,
                'makespan': dist[SINK]
            }

        # --- Шаг 4: Поиск работы c ---
        c = 0
        c_index = -1
        for idx, j in enumerate(critical_path):
            if data['r'][j] < start_times.get(j, 0) - 1e-9:
                c = j
                c_index = idx
                break

        if c == 0:
            return {
                'critical_path': critical_path,
                'c': 0,
                'J': set(),
                'start_times': start_times,
                'makespan': dist[SINK]
            }

        J = set(critical_path[c_index + 1:])

        return {
            'critical_path': critical_path,
            'c': c,
            'J': J,
            'c_index': c_index,
            'start_times': start_times,
            'makespan': dist[SINK]
        }

    # =====================================================================
    # ПРОВЕРКА УСЛОВИЙ СИЛЬНОГО ВЕТВЛЕНИЯ
    # =====================================================================

    def _check_strong_branching_conditions(self, critical_info: Dict, data: Dict,
                                           start_times: Dict[int, float]) -> bool:
        critical_path = critical_info['critical_path']
        c = critical_info['c']
        J = critical_info['J']

        if c == 0 or not J:
            return False

        try:
            c_index = critical_path.index(c)
        except ValueError:
            return False

        segment = critical_path[c_index:]
        for k in range(len(segment) - 1):
            u = segment[k]
            v = segment[k + 1]
            is_dpc = (u, v) in self.dpc_pairs
            is_sigma = v in data['sigma'].get(u, set())
            if is_dpc or is_sigma:
                return False

        t_c = start_times[c]
        for i in J:
            t_i = start_times[i]
            r_i = data['r'][i]
            if r_i < max(t_i, t_c) - 1e-9:
                return False

        return True

    # =====================================================================
    # СИЛЬНОЕ ВЕТВЛЕНИЕ
    # =====================================================================

    def _apply_strong_branching(self, data: Dict, c: int, J: Set[int],
                                upper_bound: float, depth: int) -> None:
        if self._carlier_tests(data, c, J, upper_bound):
            self._update_heads_and_tails(data)
            self._branch_and_bound(data, upper_bound, depth)
            return

        # Ветвь 1: c предшествует J
        data1 = self._copy_data(data)
        for j in J:
            self._add_precedence(data1, c, j)
            delay = self.l_matrix[c][j] if (c, j) in self.dpc_pairs else self.jobs[c].d_i
            required_rj = data1['r'][c] + delay
            if required_rj > data1['r'][j]:
                data1['r'][j] = required_rj
            required_qc = data1['q'][j] + delay
            if required_qc > data1['q'][c]:
                data1['q'][c] = required_qc
        self._update_heads_and_tails(data1)
        self._branch_and_bound(data1, upper_bound, depth)

        # Ветвь 2: J предшествует c
        data2 = self._copy_data(data)
        for j in J:
            self._add_precedence(data2, j, c)
            delay = self.l_matrix[j][c] if (j, c) in self.dpc_pairs else self.jobs[j].d_i
            required_rc = data2['r'][j] + delay
            if required_rc > data2['r'][c]:
                data2['r'][c] = required_rc
            required_qj = data2['q'][c] + delay
            if required_qj > data2['q'][j]:
                data2['q'][j] = required_qj
        self._update_heads_and_tails(data2)
        self._branch_and_bound(data2, upper_bound, depth)

    def _carlier_tests(self, data: Dict, c: int, J: Set[int], upper_bound: float) -> bool:
        r = data['r']
        q = data['q']

        if not J:
            return False

        min_r_J = min(r[j] for j in J)
        sum_d_J = sum(self.jobs[j].d_i for j in J)
        h_J = min_r_J + sum_d_J + min(q[j] for j in J)

        all_jobs = set(self.jobs.keys())
        K = {k for k in all_jobs - J - {c}
             if self.jobs[k].d_i > upper_bound - h_J}

        constraints_added = False

        for k in K:
            if r[c] + self.jobs[c].d_i + sum_d_J + q[k] >= upper_bound - 1e-6:
                for j in J:
                    if self._add_precedence(data, j, k):
                        constraints_added = True
                if constraints_added:
                    required_rk = r[c] + self.jobs[c].d_i
                    for j in J:
                        required_rk = max(required_rk, r[j] + self.jobs[j].d_i)
                    if required_rk > r[k]:
                        r[k] = required_rk

            if min_r_J + sum_d_J + self.jobs[k].d_i + q[k] >= upper_bound - 1e-6:
                for j in J:
                    if self._add_precedence(data, k, j):
                        constraints_added = True
                if constraints_added:
                    required_qk = self.jobs[c].d_i
                    for j in J:
                        required_qk = max(required_qk, q[j] + self.jobs[j].d_i)
                    if required_qk > q[k]:
                        q[k] = required_qk

        return constraints_added

    # =====================================================================
    # ОБРАТНАЯ ЗАДАЧА (ПОЛНОСТЬЮ ПЕРЕДЕЛАНО)
    # =====================================================================

    def _try_reverse_strong_branching(self, data: Dict, upper_bound: float, depth: int) -> bool:
        """
        Пытается применить сильное ветвление к обратной задаче.
        Теперь не меняет глобальные атрибуты, а передаёт обратные матрицы локально.
        """
        # 1. Создаём обратные данные
        reverse_data = self._create_reverse_data(data)
        rev_l_matrix, rev_dpc_pairs = self._create_reverse_dpc_and_matrix()

        # 2. Создаём временные incoming/outgoing списки для обратной задачи
        rev_incoming, rev_outgoing = self._build_reverse_adjacency(rev_dpc_pairs, rev_l_matrix)

        # 3. Локально запускаем LTH на обратной задаче
        rev_schedule, rev_makespan, rev_starts = self._reverse_lth(reverse_data, rev_incoming, rev_outgoing)
        if not rev_schedule:
            return False

        # 4. Ищем критический путь в обратной задаче (используем локальные матрицы)
        rev_critical = self._reverse_find_critical_path(rev_schedule, rev_starts, reverse_data, rev_l_matrix, rev_dpc_pairs)
        if rev_critical is None or rev_critical['c'] == 0:
            return False

        # 5. Проверяем условия сильного ветвления в обратной задаче (локально)
        if not self._reverse_check_strong_branching(rev_critical, reverse_data, rev_starts, rev_dpc_pairs):
            return False

        # 6. Сильное ветвление возможно!
        rev_c = rev_critical['c']
        rev_J = rev_critical['J']

        # Ветвь 1: rev_J -> rev_c в исходной задаче
        data1 = self._copy_data(data)
        for j in rev_J:
            self._add_precedence(data1, j, rev_c)
            delay = self.l_matrix[j][rev_c] if (j, rev_c) in self.dpc_pairs else self.jobs[j].d_i
            required_rc = data1['r'][j] + delay
            if required_rc > data1['r'][rev_c]:
                data1['r'][rev_c] = required_rc
            required_qj = data1['q'][rev_c] + delay
            if required_qj > data1['q'][j]:
                data1['q'][j] = required_qj
        self._update_heads_and_tails(data1)
        self._branch_and_bound(data1, upper_bound, depth)

        # Ветвь 2: rev_c -> rev_J в исходной задаче
        data2 = self._copy_data(data)
        for j in rev_J:
            self._add_precedence(data2, rev_c, j)
            delay = self.l_matrix[rev_c][j] if (rev_c, j) in self.dpc_pairs else self.jobs[rev_c].d_i
            required_rj = data2['r'][rev_c] + delay
            if required_rj > data2['r'][j]:
                data2['r'][j] = required_rj
            required_qc = data2['q'][j] + delay
            if required_qc > data2['q'][rev_c]:
                data2['q'][rev_c] = required_qc
        self._update_heads_and_tails(data2)
        self._branch_and_bound(data2, upper_bound, depth)

        return True

    def _create_reverse_data(self, data: Dict) -> Dict:
        return {
            'r': {j: data['q'][j] for j in self.jobs},
            'q': {j: data['r'][j] for j in self.jobs},
            'sigma': {k: set(v) for k, v in data.get('pi', {}).items()},
            'pi': {k: set(v) for k, v in data.get('sigma', {}).items()},
        }

    def _create_reverse_dpc_and_matrix(self):
        rev_l_matrix = defaultdict(lambda: defaultdict(float))
        rev_dpc_pairs = set()

        for (i, j) in self.dpc_pairs:
            lij = self.l_matrix[i][j]
            rev_lji = lij - self.jobs[i].d_i + self.jobs[j].d_i
            rev_lji = max(rev_lji, self.jobs[j].d_i)
            rev_l_matrix[j][i] = rev_lji
            rev_dpc_pairs.add((j, i))

        return rev_l_matrix, rev_dpc_pairs

    def _build_reverse_adjacency(self, rev_dpc_pairs, rev_l_matrix):
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for (i, j) in rev_dpc_pairs:
            if i in self.jobs and j in self.jobs:
                lij = rev_l_matrix[i][j]
                incoming[j].append((i, lij))
                outgoing[i].append((j, lij))
        return incoming, outgoing

    def _reverse_lth(self, data, rev_incoming, rev_outgoing):
        """Локальная версия LTH для обратной задачи."""
        r = data['r'].copy()
        q = data['q'].copy()
        sigma = data['sigma']

        unscheduled = set(self.jobs.keys())
        schedule = []
        start_times = {}
        current_time = 0.0

        remaining_preds = {}
        for j in self.jobs:
            count = 0
            for (i, _) in rev_incoming.get(j, []):
                if i in self.jobs:
                    count += 1
            for pred in data['pi'].get(j, set()):
                if pred in self.jobs:
                    count += 1
            remaining_preds[j] = count

        ready = {j for j in unscheduled if remaining_preds[j] == 0}

        while unscheduled:
            if not ready:
                break
            released = {j for j in ready if r[j] <= current_time + 1e-9}
            if released:
                k = max(released, key=lambda j: q[j])
                s_k = max(current_time, r[k])
            else:
                min_r = min(r[j] for j in ready)
                candidates = {j for j in ready if abs(r[j] - min_r) < 1e-9}
                k = max(candidates, key=lambda j: q[j])
                s_k = max(current_time, r[k])

            start_times[k] = s_k
            schedule.append(k)
            current_time = s_k + self.jobs[k].d_i
            unscheduled.remove(k)
            ready.remove(k)

            for (succ_id, lij) in rev_outgoing.get(k, []):
                if succ_id in unscheduled:
                    remaining_preds[succ_id] -= 1
                    if remaining_preds[succ_id] == 0:
                        ready.add(succ_id)
                    new_r = s_k + lij
                    if new_r > r[succ_id]:
                        r[succ_id] = new_r

            for succ_id in sigma.get(k, set()):
                if succ_id in unscheduled:
                    remaining_preds[succ_id] -= 1
                    if remaining_preds[succ_id] == 0:
                        ready.add(succ_id)
                    new_r = s_k + self.jobs[k].d_i
                    if new_r > r[succ_id]:
                        r[succ_id] = new_r

        C_max = 0.0
        for j in schedule:
            completion_with_delivery = start_times[j] + self.jobs[j].d_i + q[j]
            if completion_with_delivery > C_max:
                C_max = completion_with_delivery

        return schedule, C_max, start_times

    def _reverse_find_critical_path(self, schedule, start_times, data, rev_l_matrix, rev_dpc_pairs):
        """Локальная версия поиска критического пути для обратной задачи."""
        if not schedule:
            return None

        n_jobs = len(schedule)
        SOURCE = n_jobs
        SINK = n_jobs + 1

        dist = [-float('inf')] * (n_jobs + 2)
        parent = [-1] * (n_jobs + 2)
        dist[SOURCE] = 0.0

        for idx, j in enumerate(schedule):
            w = data['r'][j]
            if w > dist[idx]:
                dist[idx] = w
                parent[idx] = SOURCE

        for idx in range(n_jobs):
            if dist[idx] == -float('inf'):
                continue
            i = schedule[idx]

            # Дуги ко всем последующим
            for jdx in range(idx + 1, n_jobs):
                j = schedule[jdx]
                if (i, j) in rev_dpc_pairs:
                    weight = rev_l_matrix[i][j]
                else:
                    weight = self.jobs[i].d_i
                w = dist[idx] + weight
                if w > dist[jdx] + 1e-9:
                    dist[jdx] = w
                    parent[jdx] = idx

            w = dist[idx] + self.jobs[i].d_i + data['q'][i]
            if w > dist[SINK] + 1e-9:
                dist[SINK] = w
                parent[SINK] = idx

        if dist[SINK] == -float('inf'):
            return None

        critical_path = []
        cur = SINK
        while cur != SOURCE:
            p = parent[cur]
            if p < 0 or p == cur:
                break
            if p < n_jobs:
                critical_path.append(schedule[p])
            cur = p
        critical_path.reverse()

        c = 0
        c_index = -1
        for idx, j in enumerate(critical_path):
            if data['r'][j] < start_times.get(j, 0) - 1e-9:
                c = j
                c_index = idx
                break

        if c == 0:
            return {'critical_path': critical_path, 'c': 0, 'J': set()}

        J = set(critical_path[c_index + 1:])
        return {'critical_path': critical_path, 'c': c, 'J': J, 'c_index': c_index}

    def _reverse_check_strong_branching(self, critical_info, data, start_times, rev_dpc_pairs):
        """Локальная проверка условий сильного ветвления для обратной задачи."""
        critical_path = critical_info['critical_path']
        c = critical_info['c']
        J = critical_info['J']

        if c == 0 or not J:
            return False

        try:
            c_index = critical_path.index(c)
        except ValueError:
            return False

        segment = critical_path[c_index:]
        for k in range(len(segment) - 1):
            u = segment[k]
            v = segment[k + 1]
            if (u, v) in rev_dpc_pairs or v in data['sigma'].get(u, set()):
                return False

        t_c = start_times[c]
        for i in J:
            t_i = start_times[i]
            r_i = data['r'][i]
            if r_i < max(t_i, t_c) - 1e-9:
                return False

        return True

    # =====================================================================
    # СЛАБОЕ ВЕТВЛЕНИЕ (без изменений, но используем исправленный _add_precedence)
    # =====================================================================

    def _apply_weak_branching(self, data: Dict, critical_info: Dict,
                              upper_bound: float, depth: int) -> None:
        i, j = self._select_weak_branching_pair(data, critical_info)
        if i is None or j is None:
            return

        sigma = data['sigma']
        pi = data['pi']

        i_before_j_possible = not (
                j in sigma.get(i, set()) or
                i in pi.get(j, set()) or
                (i, j) in self.dpc_pairs
        )
        j_before_i_possible = not (
                i in sigma.get(j, set()) or
                j in pi.get(i, set()) or
                (j, i) in self.dpc_pairs
        )

        if not i_before_j_possible and not j_before_i_possible:
            return

        critical_path = critical_info['critical_path']
        try:
            idx_i = critical_path.index(i)
            idx_j = critical_path.index(j)
            i_before_j_in_path = idx_i < idx_j
        except ValueError:
            i_before_j_in_path = False

        if i_before_j_possible and i_before_j_in_path:
            self._weak_branch_try(data, i, j, upper_bound, depth)
            if j_before_i_possible:
                self._weak_branch_try(data, j, i, upper_bound, depth)
        elif j_before_i_possible:
            self._weak_branch_try(data, j, i, upper_bound, depth)
            if i_before_j_possible:
                self._weak_branch_try(data, i, j, upper_bound, depth)
        elif i_before_j_possible:
            self._weak_branch_try(data, i, j, upper_bound, depth)

    def _weak_branch_try(self, data: Dict, i: int, j: int, upper_bound: float, depth: int) -> None:
        data_copy = self._copy_data(data)
        self._add_precedence(data_copy, i, j)
        delay = self.l_matrix[i][j] if (i, j) in self.dpc_pairs else self.jobs[i].d_i
        required_rj = data_copy['r'][i] + delay
        if required_rj > data_copy['r'][j]:
            data_copy['r'][j] = required_rj
        required_qi = data_copy['q'][j] + delay
        if required_qi > data_copy['q'][i]:
            data_copy['q'][i] = required_qi
        self._update_heads_and_tails(data_copy)
        self._branch_and_bound(data_copy, upper_bound, depth)

    def _select_weak_branching_pair(self, data: Dict,
                                    critical_info: Dict) -> Tuple[Optional[int], Optional[int]]:
        critical_path = critical_info['critical_path']
        c = critical_info['c']
        start_times = critical_info['start_times']
        C_max = self.best_makespan

        if not critical_path:
            return None, None

        case1_path = self._find_critical_path_without_precedence(data, start_times)

        if case1_path is not None:
            c_case1 = case1_path['c']
            if c_case1 == 0:
                return None, None
            c_index = case1_path['c_index']
            j = c_case1
            t_j = start_times[j]
            segment = case1_path['critical_path'][c_index:]
            for node in segment:
                if data['r'][node] < max(start_times[node], t_j) - 1e-9:
                    if not self._are_ordered_pair(node, j, data):
                        return node, j
            return None, None
        else:
            critical_paths = self._find_all_critical_paths(data, start_times)

            best_path = None
            min_prec_count = float('inf')
            for path_info in critical_paths:
                if path_info['c'] == 0:
                    continue
                c_idx = path_info['c_index']
                segment = path_info['critical_path'][c_idx:]
                prec_count = 0
                for idx in range(len(segment) - 1):
                    u, v = segment[idx], segment[idx + 1]
                    if (u, v) in self.dpc_pairs or v in data['sigma'].get(u, set()):
                        prec_count += 1
                if prec_count < min_prec_count:
                    min_prec_count = prec_count
                    best_path = path_info

            if best_path is None:
                return None, None

            path = best_path['critical_path']
            c_idx = best_path['c_index']
            segment = path[c_idx:]
            l = None
            for idx in range(len(segment) - 2, -1, -1):
                u = segment[idx]
                v = segment[idx + 1]
                if self._is_essential_precedence_arc(u, v, data, start_times, C_max):
                    l = v
                    break

            if l is None:
                return None, None

            j = l
            t_j = start_times.get(j, 0)
            l_index = path.index(l)
            segment_l = path[l_index:]
            for node in segment_l:
                if node != l and data['r'][node] < t_j - 1e-9:
                    if not self._are_ordered_pair(node, j, data):
                        return node, j
            return None, None

    def _find_critical_path_without_precedence(self, data: Dict,
                                               start_times: Dict[int, float]) -> Optional[Dict]:
        schedule = sorted(start_times.keys(), key=lambda j: start_times[j])
        critical_info = self._find_critical_path(schedule, start_times, data)
        if critical_info is None or critical_info['c'] == 0:
            return critical_info

        path = critical_info['critical_path']
        c_index = critical_info.get('c_index', path.index(critical_info['c']))
        segment = path[c_index:]
        for idx in range(len(segment) - 1):
            u, v = segment[idx], segment[idx + 1]
            if (u, v) in self.dpc_pairs or v in data['sigma'].get(u, set()):
                return None

        return critical_info

    def _find_all_critical_paths(self, data: Dict,
                                 start_times: Dict[int, float]) -> List[Dict]:
        schedule = sorted(start_times.keys(), key=lambda j: start_times[j])
        critical_info = self._find_critical_path(schedule, start_times, data)
        return [critical_info] if critical_info else []

    def _is_essential_precedence_arc(self, u: int, v: int, data: Dict,
                                     start_times: Dict[int, float],
                                     C_max: float) -> bool:
        if (u, v) not in self.dpc_pairs:
            return False
        lij = self.l_matrix[u][v]
        if lij <= self.jobs[u].d_i:
            return False
        r_u = data['r'][u]
        t_u = start_times.get(u, 0)
        q_u = data['q'][u]
        if r_u >= t_u - 1e-9:
            return False
        if q_u >= C_max - t_u - self.jobs[u].d_i - 1e-9:
            return False
        return True

    def _are_ordered_pair(self, i: int, j: int, data: Dict) -> bool:
        sigma = data['sigma']
        pi = data['pi']
        if j in sigma.get(i, set()) or i in pi.get(j, set()):
            return True
        if (i, j) in self.dpc_pairs:
            return True
        return False

    # =====================================================================
    # ПОСТОБРАБОТКА
    # =====================================================================

    def _postprocess(self, data: Dict, schedule: List[int],
                     start_times: Dict[int, float], C_max: float) -> bool:
        if not schedule:
            return False

        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']
        pi = data['pi']

        critical_info = self._find_critical_path(schedule, start_times, data)
        if critical_info is None or critical_info['c'] == 0:
            return False

        critical_path = critical_info['critical_path']

        # Proposition 3.3
        for idx, j in enumerate(critical_path):
            K = critical_path[idx + 1:]
            if not K:
                continue

            has_precedence = False
            for k_idx in range(idx, len(critical_path) - 1):
                u = critical_path[k_idx]
                v = critical_path[k_idx + 1]
                if (u, v) in self.dpc_pairs or v in sigma.get(u, set()):
                    has_precedence = True
                    break
            if has_precedence:
                continue

            q_j = q[j]
            t_j = start_times[j]
            conditions_hold = True
            for k in K:
                if q[k] < q_j - 1e-9:
                    conditions_hold = False
                    break
                if k not in sigma.get(j, set()) and r[k] < t_j - 1e-9:
                    conditions_hold = False
                    break

            if conditions_hold:
                for k in K:
                    if k not in sigma.get(j, set()):
                        self._add_precedence(data, j, k)
                        delay = self.l_matrix[j][k] if (j, k) in self.dpc_pairs else self.jobs[j].d_i
                        required_rk = t_j + delay
                        if required_rk > r[k]:
                            r[k] = required_rk
                        changed = True

        if changed:
            return True

        # Proposition 3.4
        for idx in range(len(critical_path) - 1, -1, -1):
            i = critical_path[idx]
            K = critical_path[:idx]
            if not K:
                continue

            has_precedence = False
            for k_idx in range(0, idx):
                u = critical_path[k_idx]
                v = critical_path[k_idx + 1]
                if (u, v) in self.dpc_pairs or v in sigma.get(u, set()):
                    has_precedence = True
                    break
            if has_precedence:
                continue

            r_i = r[i]
            q_i = q[i]
            conditions_hold = True
            for k in K:
                if r[k] < r_i - 1e-9:
                    conditions_hold = False
                    break
                if k not in pi.get(i, set()) and q[k] < q_i - 1e-9:
                    conditions_hold = False
                    break

            if conditions_hold:
                for k in K:
                    if i not in sigma.get(k, set()):
                        self._add_precedence(data, k, i)
                        delay = self.l_matrix[k][i] if (k, i) in self.dpc_pairs else self.jobs[k].d_i
                        required_qk = q_i + delay
                        if required_qk > q[k]:
                            q[k] = required_qk
                        changed = True

        return changed

    # =====================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =====================================================================

    def _add_precedence(self, data: Dict, i: int, j: int) -> bool:
        if i == j:
            return False

        sigma = data['sigma']
        pi = data['pi']

        if i not in sigma:
            sigma[i] = set()
        if j not in pi:
            pi[j] = set()

        if j in sigma.get(i, set()):
            return True
        if i in sigma.get(j, set()):
            return False

        sigma[i].add(j)
        pi[j].add(i)

        for k in list(sigma.get(j, set())):
            if k != i:
                if not self._add_precedence(data, i, k):
                    return False
        for k in list(pi.get(i, set())):
            if k != j:
                if not self._add_precedence(data, k, j):
                    return False

        return True

    def _copy_data(self, data: Dict) -> Dict:
        return {
            'r': data['r'].copy(),
            'q': data['q'].copy(),
            'sigma': {k: set(v) for k, v in data['sigma'].items()},
            'pi': {k: set(v) for k, v in data['pi'].items()},
        }