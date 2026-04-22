"""
Точный алгоритм ветвей и границ для задачи 1|r_j, q_j, DPC|C_max
Основан на статье Balas, Lenstra, Vazacopoulos (1995)
Финальная версия с поиском критического пути через граф G(t).
"""

import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

from core.Algorithm import Algorithm
from algorithms.lth import LTH
from core.job import Job


class BalasBaBDPC(Algorithm):
    """
    Алгоритм ветвей и границ для задачи одного станка с отложенными ограничениями предшествования.
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 time_limit: float = 60.0):
        super().__init__(jobs, precedence_constraints)
        self.time_limit = time_limit
        self.start_time = 0

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

        # Строим списки входящих DPC ОДИН РАЗ
        self._incoming_dpc = defaultdict(list)
        for i in self.jobs:
            for j in self.jobs:
                lij = self.l_matrix[i][j]
                if lij > 0:
                    self._incoming_dpc[j].append((i, lij))

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        self.start_time = time.time()
        self.timed_out = False

        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0

        jobs_list = list(self.jobs.values())

        precedence_dict = None
        if self.l_matrix:
            precedence_dict = {}
            for i in self.jobs:
                for j in self.jobs:
                    if self.l_matrix[i][j] > 0:
                        precedence_dict[(i, j)] = self.l_matrix[i][j]

        for i in self.jobs:
            for j in self.jobs:
                if self.l_matrix[i][j] > 0 and self.l_matrix[j][i] > 0:
                    print(f"⚠️ WARNING: Mutual DPC between {i} and {j}")

        boh = LTH(jobs_list, precedence_dict)
        boh_schedule, boh_makespan, boh_stats = boh.solve(**kwargs)

        # Устанавливаем начальные значения из эвристики
        self.best_makespan = boh_makespan
        self.best_schedule = boh_schedule.copy() if boh_schedule else None
        initial_upper_bound = boh_makespan

        best_heuristic_name = boh_stats.get("best_algorithm", "Unknown")
        # =============================================================================

        initial_data = {
            'r': {j.id: j.r_i for j in jobs_list},
            'q': {j.id: j.q_i for j in jobs_list},
            'sigma': defaultdict(set),
            'pi': defaultdict(set),
        }

        # Запускаем B&B с хорошей верхней границей
        self._branch_and_bound(initial_data, initial_upper_bound, depth=0, history=[])

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
            'initial_heuristic': best_heuristic_name,
            'initial_makespan': boh_makespan,
            'improvement': boh_makespan - self.best_makespan if self.best_makespan < float('inf') else 0,
            'optimal': self.best_schedule is not None and not self.timed_out
        }

        return self.best_schedule, self.best_makespan, stats

    def _add_precedence(self, sigma: Dict[int, Set[int]], pi: Dict[int, Set[int]],
                        i: int, j: int) -> bool:
        """
        Добавляет отношение i -> j с транзитивным замыканием.
        Возвращает False, если создаётся цикл.
        """
        # Нельзя добавить отношение самому себе
        if i == j:
            # print(f"⚠️ WARNING: Attempt to add self-precedence {i} -> {j}")
            return False

        if j in sigma.get(i, set()):
            return True  # уже есть

        # Проверка на цикл: если j -> i уже существует, нельзя добавлять i -> j
        if i in sigma.get(j, set()):
            # print(f"⚠️ WARNING: Cycle detected! Cannot add {i} -> {j} because {j} -> {i} exists")
            return False

        sigma[i].add(j)
        pi[j].add(i)

        # Транзитивность: все, кто должны быть после j, теперь должны быть после i
        for k in list(sigma.get(j, set())):
            if k != i:  # Защита от самопетли
                if not self._add_precedence(sigma, pi, i, k):
                    return False

        # Транзитивность: все, кто должны быть перед i, теперь должны быть перед j
        for k in list(pi.get(i, set())):
            if k != j:  # Защита от самопетли
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
                           start_times: Dict[int, float], C_max: float) -> bool:
        """Проверяет, является ли дуга (u,v) precedence arc любого типа (essential или нет)."""
        if self.l_matrix[u][v] > 0:
            return True

        t_u = start_times.get(u, -float('inf'))
        t_v = start_times.get(v, -float('inf'))
        if t_u > -float('inf') and t_v > -float('inf'):
            if abs(t_v - (t_u + self.jobs[u].d_i)) < 1e-6:
                return True

        return False

    def _find_critical_path_via_graph(self, schedule: List[int], start_times: Dict[int, float],
                                      C_max: float, data: Dict) -> Optional[Dict]:
        # Быстрая проверка через fallback
        result = self._find_critical_path_fallback(schedule, start_times, C_max, data)
        if result is not None:
            return result

        # Если fallback не сработал — используем полный граф
        n = len(self.jobs)
        job_ids = list(self.jobs.keys())
        idx_map = {jid: i for i, jid in enumerate(job_ids)}

        N = n + 2
        o = n
        t = n + 1

        dist = [[-float('inf')] * N for _ in range(N)]

        for i, jid in enumerate(job_ids):
            dist[o][i] = data['r'][jid]

        for idx_i, i in enumerate(schedule):
            for idx_j, j in enumerate(schedule):
                if idx_i < idx_j:
                    lij = self.l_matrix[i][j]
                    if lij > 0:
                        dist[idx_map[i]][idx_map[j]] = max(dist[idx_map[i]][idx_map[j]], lij)
                    else:
                        dist[idx_map[i]][idx_map[j]] = max(dist[idx_map[i]][idx_map[j]],
                                                           self.jobs[i].d_i)

        for i, jid in enumerate(job_ids):
            dist[i][t] = self.jobs[jid].d_i + data['q'][jid]

        topo = [o] + list(range(n)) + [t]

        longest = [-float('inf')] * N
        longest[o] = 0
        pred = [-1] * N

        for u in topo:
            if longest[u] == -float('inf'):
                continue
            for v in range(N):
                if dist[u][v] > -float('inf'):
                    if longest[u] + dist[u][v] > longest[v]:
                        longest[v] = longest[u] + dist[u][v]
                        pred[v] = u

        if longest[t] < C_max - 1e-6:
            return None

        path = []
        v = t
        while v != o:
            u = pred[v]
            if u == -1:
                break
            if u != o and u != t:
                path.insert(0, job_ids[u])
            v = u

        if not path:
            return None

        c = None
        for jid in path:
            if data['r'][jid] < start_times.get(jid, 0) - 1e-6:
                c = jid
                break
        if c is None:
            c = path[0] if path else None

        if c is None:
            return None

        c_index = path.index(c) if c in path else 0
        J = set(path[c_index + 1:])

        return {
            'critical_path': path,
            'c': c,
            'J': J,
            'start_times': start_times
        }

    def _find_critical_path_fallback(self, schedule: List[int], start_times: Dict[int, float],
                                     C_max: float, data: Dict) -> Optional[Dict]:
        if not schedule:
            return None

        q = data['q']
        r = data['r']

        # Строим обратный индекс для быстрого поиска
        # Это позволяет найти предшественника за O(1) вместо O(n)
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

            # Проверяем только работы, идущие раньше в расписании
            for pred in schedule[:curr_idx]:
                pred_end = start_times[pred] + self.jobs[pred].d_i
                if abs(pred_end - start_times[current]) < 1e-6:
                    current = pred
                    critical_path.insert(0, current)
                    found = True
                    break
                lij = self.l_matrix[pred][current]
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
                                           schedule: List[int], start_times: Dict,
                                           C_max: float) -> bool:
        """Проверяет условия теоремы 3.1 для всего сегмента C(c,n)."""
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
            if self._is_precedence_arc(u, v, data, start_times, C_max):
                return False

        t_c = start_times[c]
        for i in J:
            t_i = start_times[i]
            if r[i] < max(t_i, t_c) - 1e-6:
                return False

        return True

    def _branch_and_bound(self, data: Dict, upper_bound: float, depth: int, history: List[str]) -> None:
        self.nodes_explored += 1

        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return

        if depth > 3 * self.n:
            return

        lb = self._calculate_lower_bound(data)
        if lb >= upper_bound - 1e-6:
            self.pruned_by_bound += 1
            return

        schedule, makespan, start_times = self._longest_tail_heuristic(data)

        if makespan < self.best_makespan:
            self.best_makespan = makespan
            self.best_schedule = schedule.copy()
            self.best_sigma = {k: set(v) for k, v in data['sigma'].items()}
            self.best_pi = {k: set(v) for k, v in data['pi'].items()}
            upper_bound = min(upper_bound, makespan)
            if lb >= makespan - 1e-6:
                return

        while True:
            changed = self._postprocess(data, schedule, start_times, makespan)
            if not changed:
                break
            schedule, makespan, start_times = self._longest_tail_heuristic(data)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = schedule.copy()
                self.best_sigma = {k: set(v) for k, v in data['sigma'].items()}
                self.best_pi = {k: set(v) for k, v in data['pi'].items()}
                upper_bound = min(upper_bound, makespan)

        critical_info = self._find_critical_path_via_graph(schedule, start_times, makespan, data)
        if critical_info is None:
            return

        c = critical_info['c']
        J = critical_info['J']

        if c == 0:
            return

        can_use_strong = self._check_strong_branching_conditions(
            critical_info, data, schedule, start_times, makespan
        )

        if can_use_strong:
            self.strong_branches += 1
            self._apply_strong_branching(data, c, J, upper_bound, depth + 1, history)
        else:
            # Пробуем обратную задачу
            reverse_data = self._create_reverse_problem(data)
            original_l_matrix = self.l_matrix
            self.l_matrix = self._get_reverse_l_matrix()
            rev_schedule, rev_makespan, rev_starts = self._longest_tail_heuristic(reverse_data)
            self.l_matrix = original_l_matrix

            rev_critical = self._find_critical_path_via_graph(rev_schedule, rev_starts, rev_makespan, reverse_data)

            if rev_critical and self._check_strong_branching_conditions(
                    rev_critical, reverse_data, rev_schedule, rev_starts, rev_makespan
            ):
                self.strong_branches += 1
                rev_c = rev_critical['c']
                rev_J = rev_critical['J']
                # ВАЖНО: инвертируем отношения при возврате из обратной задачи
                self._apply_strong_branching_reversed(data, rev_c, rev_J, upper_bound, depth + 1, history)
            else:
                self.weak_branches += 1
                self._apply_weak_branching(data, critical_info, upper_bound, depth + 1, history)

    def _apply_weak_branching(self, data: Dict, critical_info: Dict, upper_bound: float,
                              depth: int, history: List[str]) -> None:
        i, j = self._select_weak_branching_pair(data, critical_info)
        if i is None or j is None:
            return

        # Усиленная защита от зацикливания
        state_id = f"weak_{i}_{j}"
        if state_id in history:
            return

        # Ограничиваем историю
        if len(history) > 20:
            return

        new_history = history + [state_id]

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

        # ВАЖНО: сначала пробуем более перспективную ветвь
        # Обычно ветвь, которая соответствует порядку в критическом пути, более перспективна
        # Определяем, какая работа идёт раньше в critical_path
        critical_path = critical_info['critical_path']
        try:
            idx_i = critical_path.index(i)
            idx_j = critical_path.index(j)
            i_before_j_in_path = idx_i < idx_j
        except ValueError:
            i_before_j_in_path = False

        if i_before_j_possible and i_before_j_in_path:
            data1 = self._copy_data(data)
            self._add_precedence(data1['sigma'], data1['pi'], i, j)
            lij = self.l_matrix[i][j]
            data1['r'][j] = max(data1['r'][j], data1['r'][i] + self.jobs[i].d_i + lij)
            self._branch_and_bound(data1, upper_bound, depth, new_history)

            if j_before_i_possible:
                data2 = self._copy_data(data)
                self._add_precedence(data2['sigma'], data2['pi'], j, i)
                lji = self.l_matrix[j][i]
                data2['r'][i] = max(data2['r'][i], data2['r'][j] + self.jobs[j].d_i + lji)
                self._branch_and_bound(data2, upper_bound, depth, new_history)
        elif j_before_i_possible:
            data2 = self._copy_data(data)
            self._add_precedence(data2['sigma'], data2['pi'], j, i)
            lji = self.l_matrix[j][i]
            data2['r'][i] = max(data2['r'][i], data2['r'][j] + self.jobs[j].d_i + lji)
            self._branch_and_bound(data2, upper_bound, depth, new_history)

            if i_before_j_possible:
                data1 = self._copy_data(data)
                self._add_precedence(data1['sigma'], data1['pi'], i, j)
                lij = self.l_matrix[i][j]
                data1['r'][j] = max(data1['r'][j], data1['r'][i] + self.jobs[i].d_i + lij)
                self._branch_and_bound(data1, upper_bound, depth, new_history)
        elif i_before_j_possible:
            data1 = self._copy_data(data)
            self._add_precedence(data1['sigma'], data1['pi'], i, j)
            lij = self.l_matrix[i][j]
            data1['r'][j] = max(data1['r'][j], data1['r'][i] + self.jobs[i].d_i + lij)
            self._branch_and_bound(data1, upper_bound, depth, new_history)

    def _apply_strong_branching_reversed(self, data: Dict, c: int, J: Set[int],
                                          upper_bound: float, depth: int, history: List[str]) -> None:
        """Применяет сильное ветвление, полученное из обратной задачи."""
        if self._carlier_tests_reversed(data, c, J, upper_bound):
            self.pruned_by_test += 1
            return

        data1 = self._copy_data(data)
        for j in J:
            self._add_precedence(data1['sigma'], data1['pi'], j, c)
            data1['r'][c] = max(data1['r'][c], data1['r'][j] + self.jobs[j].d_i + self.l_matrix[j][c])
        self._branch_and_bound(data1, upper_bound, depth, history)

        data2 = self._copy_data(data)
        for j in J:
            self._add_precedence(data2['sigma'], data2['pi'], c, j)
            data2['r'][j] = max(data2['r'][j], data2['r'][c] + self.jobs[c].d_i + self.l_matrix[c][j])
        self._branch_and_bound(data2, upper_bound, depth, history)

    def _carlier_tests_reversed(self, data: Dict, c: int, J: Set[int], upper_bound: float) -> bool:
        r = data['r']
        q = data['q']

        min_r_J = min(r[j] for j in J) if J else float('inf')
        sum_d_J = sum(self.jobs[j].d_i for j in J)
        min_q_J = min(q[j] for j in J) if J else float('inf')
        h_J = min_r_J + sum_d_J + min_q_J

        all_jobs = set(self.jobs.keys())
        K = {k for k in all_jobs - J - {c}
             if self.jobs[k].d_i > upper_bound - h_J}

        for k in K:
            if min_r_J + sum_d_J + self.jobs[k].d_i + q[k] >= upper_bound - 1e-6:
                return True
        return False

    def _calculate_lower_bound(self, data: Dict) -> float:
        r = data['r']
        q = data['q']

        if not self.jobs:
            return 0.0

        # ОПТИМИЗАЦИЯ: кэшируем суммы
        if not hasattr(self, '_total_d'):
            self._total_d = sum(j.d_i for j in self.jobs.values())

        min_r = min(r.values())
        min_q = min(q.values())
        lb1 = min_r + self._total_d + min_q

        # ОПТИМИЗАЦИЯ: вычисляем lb2 только если есть DPC
        lb2 = 0.0
        if self.l_matrix:
            # Проверяем только пары с ненулевыми задержками
            for i in self.jobs:
                for j in self.jobs:
                    lij = self.l_matrix[i][j]
                    if lij > 0:
                        path_length = r[i] + self.jobs[i].d_i + lij + q[j]
                        if path_length > lb2:
                            lb2 = path_length

        return max(lb1, lb2)

    def _longest_tail_heuristic(self, data: Dict) -> Tuple[List[int], float, Dict[int, float]]:
        r = data['r'].copy()
        q = data['q'].copy()
        sigma = data['sigma']

        unscheduled = set(self.jobs.keys())
        schedule = []
        start_times = {}
        current_time = 0.0

        incoming_dpc = self._incoming_dpc

        # Счётчик итераций для предотвращения бесконечного цикла
        max_iterations = len(unscheduled) * 2
        iteration = 0

        while unscheduled and iteration < max_iterations:
            iteration += 1
            best_job = None
            best_q = -float('inf')
            best_start = 0

            for job_id in unscheduled:
                preds = sigma.get(job_id)
                if preds:
                    can_schedule = True
                    for pred in preds:
                        if pred in unscheduled:
                            can_schedule = False
                            break
                    if not can_schedule:
                        continue

                start = r[job_id]
                if current_time > start:
                    start = current_time

                dpc_list = incoming_dpc.get(job_id)
                if dpc_list:
                    for s_id, lij in dpc_list:
                        if s_id in start_times:
                            candidate = start_times[s_id] + lij
                            if candidate > start:
                                start = candidate

                q_val = q[job_id]
                if q_val > best_q or (q_val == best_q and start < best_start):
                    best_q = q_val
                    best_job = job_id
                    best_start = start

            if best_job is None:
                # Застряли — печатаем отладочную информацию
                print(f"⚠️ LTH stuck! Unscheduled: {len(unscheduled)} jobs")
                print(f"   Jobs with unmet predecessors:")
                for job_id in unscheduled:
                    preds = sigma.get(job_id, set())
                    unmet = [p for p in preds if p in unscheduled]
                    if unmet:
                        print(f"   Job {job_id} waiting for: {unmet}")
                break

            start_times[best_job] = best_start
            schedule.append(best_job)
            current_time = best_start + self.jobs[best_job].d_i
            unscheduled.remove(best_job)

        if unscheduled:
            # Если остались незапланированные работы — добавляем их в конец
            print(f"⚠️ Warning: {len(unscheduled)} jobs could not be scheduled normally")
            for job_id in list(unscheduled):
                start = max(r[job_id], current_time)
                for s_id, lij in incoming_dpc[job_id]:
                    if s_id in start_times:
                        candidate = start_times[s_id] + lij
                        if candidate > start:
                            start = candidate
                start_times[job_id] = start
                schedule.append(job_id)
                current_time = start + self.jobs[job_id].d_i

        # Вычисление C_max
        C_max = 0.0
        for j in schedule:
            delivery_end = start_times[j] + self.jobs[j].d_i + q[j]
            if delivery_end > C_max:
                C_max = delivery_end

        return schedule, C_max, start_times

    def _apply_strong_branching(self, data: Dict, c: int, J: Set[int], upper_bound: float,
                                depth: int, history: List[str]) -> None:
        if self._carlier_tests(data, c, J, upper_bound):
            self.pruned_by_test += 1
            return

        data1 = self._copy_data(data)
        for j in J:
            self._add_precedence(data1['sigma'], data1['pi'], c, j)
            data1['r'][j] = max(data1['r'][j], data1['r'][c] + self.jobs[c].d_i + self.l_matrix[c][j])
        self._branch_and_bound(data1, upper_bound, depth, history)

        data2 = self._copy_data(data)
        for j in J:
            self._add_precedence(data2['sigma'], data2['pi'], j, c)
        for j in J:
            data2['q'][j] = max(data2['q'][j], data2['q'][c] + self.jobs[c].d_i + self.l_matrix[j][c])
        self._branch_and_bound(data2, upper_bound, depth, history)

    def _carlier_tests(self, data: Dict, c: int, J: Set[int], upper_bound: float) -> bool:
        r = data['r']
        q = data['q']

        min_r_J = min(r[j] for j in J) if J else float('inf')
        sum_d_J = sum(self.jobs[j].d_i for j in J)
        min_q_J = min(q[j] for j in J) if J else float('inf')
        h_J = min_r_J + sum_d_J + min_q_J

        all_jobs = set(self.jobs.keys())
        K = {k for k in all_jobs - J - {c}
             if self.jobs[k].d_i > upper_bound - h_J}

        for k in K:
            if r[c] + self.jobs[c].d_i + sum_d_J + q[k] >= upper_bound - 1e-6:
                return True
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

        try:
            c_index = critical_path.index(c)
        except ValueError:
            return None, None

        segment = critical_path[c_index:]

        essential_arc = None
        for idx in range(len(segment) - 1, 0, -1):
            u, v = segment[idx - 1], segment[idx]
            if self._is_essential_precedence_arc(u, v, data, start_times, C_max):
                essential_arc = (u, v)
                break

        if essential_arc is None:
            j = c if c != 0 else critical_path[0]
            t_j = start_times[j]
            for node in segment:
                if r[node] < max(start_times[node], t_j) - 1e-6:
                    if not self._are_ordered(node, j, data) and not self._are_ordered(j, node, data):
                        return node, j
            return c, j
        else:
            k, l = essential_arc
            j = l
            try:
                l_index = critical_path.index(l)
            except ValueError:
                return None, None
            segment_l = critical_path[l_index:]
            for node in segment_l:
                if node != l and r[node] < start_times[l] - 1e-6:
                    if not self._are_ordered(node, j, data) and not self._are_ordered(j, node, data):
                        return node, j
            if len(segment_l) > 1:
                return segment_l[0], j
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
                     start_times: Dict[int, float], C_max: float) -> bool:
        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']
        pi = data['pi']

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
                    new_r = t_j + self.jobs[j].d_i + self.l_matrix[j][k]
                    if new_r > r[k] + 1e-6:
                        r[k] = new_r
                        changed = True
                    if j not in sigma.get(k, set()):
                        self._add_precedence(sigma, pi, j, k)
                        changed = True

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
                    new_q = q[i] + self.jobs[i].d_i + self.l_matrix[k][i] - self.jobs[k].d_i
                    if new_q > q[k] + 1e-6:
                        q[k] = new_q
                        changed = True
                    if k not in sigma.get(i, set()):
                        self._add_precedence(sigma, pi, k, i)
                        changed = True

        return changed

    def _create_reverse_problem(self, data: Dict) -> Dict:
        r = data['r']
        q = data['q']
        return {
            'r': {j: q[j] for j in self.jobs.keys()},
            'q': {j: r[j] for j in self.jobs.keys()},
            'sigma': data['pi'].copy(),
            'pi': data['sigma'].copy(),
        }

    def _get_reverse_l_matrix(self):
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