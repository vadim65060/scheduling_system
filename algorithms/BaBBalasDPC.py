"""
Точный алгоритм ветвей и границ для задачи 1|r_j, q_j, DPC|C_max
Основан на статье Balas, Lenstra, Vazacopoulos (1995).
Строгая реализация Algorithm 1 (Bal) с учётом всех теорем и свойств.
"""

import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

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
        self.init_jobs = jobs
        self.init_precedence = precedence_constraints
        self.time_limit = time_limit
        self.start_time = 0.0

        # Лучшее найденное решение
        self.best_schedule: Optional[List[int]] = None
        self.best_makespan: float = float('inf')
        self.timed_out: bool = False

        # Статистика поиска
        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0

        # Списки входящих/исходящих DPC для быстрого доступа
        self._incoming_dpc = defaultdict(list)
        self._outgoing_dpc = defaultdict(list)
        for (i, j) in self.dpc_pairs:
            if i in self.jobs and j in self.jobs:
                lij = self.l_matrix[i][j]
                self._incoming_dpc[j].append((i, lij))
                self._outgoing_dpc[i].append((j, lij))

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

        # Сброс статистики
        self.nodes_explored = 0
        self.strong_branches = 0
        self.weak_branches = 0
        self.pruned_by_bound = 0
        self.pruned_by_test = 0

        # --- Шаг 1. Вычисление начальной верхней границы ---
        from copy import deepcopy
        boh = BestOfHeuristics(deepcopy(self.init_jobs), deepcopy(self.init_precedence))
        boh_schedule, init_makespan, _ = boh.solve(**kwargs)

        self.best_makespan = init_makespan
        self.best_schedule = boh_schedule.copy() if boh_schedule else None
        upper_bound = init_makespan

        # --- Шаг 2. Начальное усиление голов и хвостов (Algorithm 1, строка 5) ---
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
    # ИНИЦИАЛИЗАЦИЯ И УСИЛЕНИЕ ОГРАНИЧЕНИЙ
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
        """
        Итеративно усиливает головы и хвосты до достижения фиксированной точки.
        Максимальное число итераций = n (как в Беллмане-Форде).
        """
        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']

        # Предварительно вычисляем задержки для sigma-отношений
        sigma_delays = {}
        for i, next_set in sigma.items():
            for j in next_set:
                if (i, j) in self.dpc_pairs:
                    sigma_delays[(i, j)] = self.l_matrix[i][j]
                else:
                    sigma_delays[(i, j)] = self.jobs[i].d_i

        # Не более n итераций (длина максимального пути в графе)
        for _ in range(self.n):
            local_changed = False

            # Обновление на основе всех DPC-дуг
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

            # Обновление на основе sigma-отношений (только если не DPC)
            for (i, j), delay in sigma_delays.items():
                # Обновление головы: r_j >= r_i + d_i
                new_rj = r[i] + delay
                if new_rj > r[j] + 1e-9:
                    r[j] = new_rj
                    local_changed = True

                # Обновление хвоста: q_i >= q_j + d_j (НЕ d_i!)
                new_qi = q[j] + self.jobs[j].d_i  # ИСПРАВЛЕНО: используем d_j, а не d_i
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
        """
        Основная рекурсивная процедура Algorithm 1 (Bal).
        Раздел 4.
        """
        self.nodes_explored += 1

        # Проверка лимита времени
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return

        # Проверка глубины (эвристическая)
        if depth > 3 * self.n:
            return

        # Шаг 1: Вычисление нижней границы
        lb = self._calculate_lower_bound(data)
        if lb >= upper_bound - 1e-6:
            self.pruned_by_bound += 1
            return

        # Шаг 2: Построение расписания эвристикой LTH
        schedule, makespan, start_times = self._longest_tail_heuristic(data)

        # Обновление лучшего решения
        if makespan < self.best_makespan - 1e-9:
            self.best_makespan = makespan
            self.best_schedule = schedule.copy()
            upper_bound = min(upper_bound, makespan)
            if lb >= makespan - 1e-6:
                return

        # Шаг 3: Постобработка (Propositions 3.3 и 3.4)
        # Цикл: пока есть изменения, перезапускаем LTH
        max_pp_iter = self.n  # Максимум n итераций
        for _ in range(max_pp_iter):
            changed = self._postprocess(data, schedule, start_times, makespan)
            if not changed:
                break
            schedule, makespan, start_times = self._longest_tail_heuristic(data)
            if makespan < self.best_makespan - 1e-9:
                self.best_makespan = makespan
                self.best_schedule = schedule.copy()
                upper_bound = min(upper_bound, makespan)

        # Шаг 4: Ветвление
        critical_info = self._find_critical_path(schedule, start_times, data)
        if critical_info is None:
            return

        c = critical_info['c']
        J = critical_info['J']

        # c = 0: расписание оптимально для данной подзадачи
        if c == 0:
            return

        # Проверка условий сильного ветвления (Теорема 3.1)
        if self._check_strong_branching_conditions(critical_info, data, start_times):
            # Сильное ветвление
            self.strong_branches += 1
            self._apply_strong_branching(data, c, J, upper_bound, depth + 1)
        else:
            # Попытка обратной задачи
            if self._try_reverse_strong_branching(data, upper_bound, depth + 1):
                self.strong_branches += 1
            else:
                # Слабое ветвление
                self.weak_branches += 1
                self._apply_weak_branching(data, critical_info, upper_bound, depth + 1)

    # =====================================================================
    # НИЖНЯЯ ГРАНИЦА
    # =====================================================================

    def _calculate_lower_bound(self, data: Dict) -> float:
        """
        Вычисляет нижнюю границу для подзадачи.
        LB = max(r_min + sum(d_i) + q_min, max_{(i,j) in DPC}(r_i + d_i + l_ij + q_j))
        """
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
        for (i, j) in self.dpc_pairs:
            lij = self.l_matrix[i][j]
            path_length = r[i] + self.jobs[i].d_i + lij + q[j]
            if path_length > lb2:
                lb2 = path_length

        return max(lb1, lb2)

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

        # Инкрементальный подход: счётчики невыполненных предшественников
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

        # Начальное множество доступных работ
        ready = {j for j in unscheduled if remaining_preds[j] == 0}

        while unscheduled:
            if not ready:
                break

            # Выбираем работу с максимальным q среди released
            released = {j for j in ready if r[j] <= current_time + 1e-9}

            if released:
                k = max(released, key=lambda j: q[j])
                s_k = max(current_time, r[k])
            else:
                min_r = min(r[j] for j in ready)
                candidates = {j for j in ready if abs(r[j] - min_r) < 1e-9}
                k = max(candidates, key=lambda j: q[j])
                s_k = max(current_time, r[k])

            # Планируем работу k
            start_times[k] = s_k
            schedule.append(k)
            current_time = s_k + self.jobs[k].d_i
            unscheduled.remove(k)
            ready.remove(k)

            # Обновляем счётчики последователей
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

        # Вычисление makespan
        C_max = 0.0
        for j in schedule:
            completion_with_delivery = start_times[j] + self.jobs[j].d_i + q[j]
            if completion_with_delivery > C_max:
                C_max = completion_with_delivery

        return schedule, C_max, start_times

    # =====================================================================
    # ПОИСК КРИТИЧЕСКОГО ПУТИ
    # =====================================================================

    def _find_critical_path(self, schedule: List[int], start_times: Dict[int, float],
                            data: Dict) -> Optional[Dict]:
        """
        Находит критический путь в графе G(S).
        Строгое соответствие статье: раздел 3, Figure 3.

        Граф G(S) = (N, E), где:
        - N = J ∪ {0, n} (0 — исток, n — сток)
        - Дуги:
          (0, i) вес r_i
          (i, j) вес l_{ij} если (i,j) ∈ DPC, иначе d_i если j следует сразу за i
          (i, n) вес d_i + q_i
        """
        if not schedule:
            return None

        n_jobs = len(schedule)
        # Исток = индекс n_jobs, сток = n_jobs + 1
        SOURCE = n_jobs
        SINK = n_jobs + 1

        # DP[i] = длина длиннейшего пути от истока до узла i
        dp = [-float('inf')] * (n_jobs + 2)
        parent = [-1] * (n_jobs + 2)
        dp[SOURCE] = 0.0

        # Индекс работы в расписании
        job_to_pos = {j: idx for idx, j in enumerate(schedule)}

        # Инициализация дуг (0, i) для всех работ
        for idx, j in enumerate(schedule):
            w = data['r'][j]  # Вес дуги (0, j) = r_j
            if w > dp[idx]:
                dp[idx] = w
                parent[idx] = SOURCE

        # Прямой проход по всем дугам между работами
        for idx in range(n_jobs):
            i = schedule[idx]
            # Дуга к следующей работе в расписании (стандартная дуга)
            if idx + 1 < n_jobs:
                j = schedule[idx + 1]
                w = dp[idx] + self.jobs[i].d_i
                if w > dp[idx + 1]:
                    dp[idx + 1] = w
                    parent[idx + 1] = idx

            # DPC-дуги от i ко всем j, которые идут позже в расписании
            for (j, lij) in self._outgoing_dpc.get(i, []):
                pos_j = job_to_pos.get(j)
                if pos_j is not None and pos_j > idx:
                    w = dp[idx] + lij
                    if w > dp[pos_j]:
                        dp[pos_j] = w
                        parent[pos_j] = idx

        # Дуги (i, n) от каждой работы к стоку
        for idx, j in enumerate(schedule):
            w = dp[idx] + self.jobs[j].d_i + data['q'][j]
            if w > dp[SINK]:
                dp[SINK] = w
                parent[SINK] = idx

        # Восстановление критического пути от стока к истоку
        critical_path = []
        cur = SINK
        while cur != SOURCE:
            parent_cur = parent[cur]
            if parent_cur < 0:
                # Обрыв пути (не должно происходить)
                break
            if parent_cur < n_jobs:  # Это работа, не исток
                critical_path.append(schedule[parent_cur])
            cur = parent_cur
        critical_path.reverse()

        if not critical_path:
            return None

        # Поиск работы c: первой работы на критическом пути, где r_i < s_i
        c = 0
        c_index = -1
        for idx, j in enumerate(critical_path):
            if data['r'][j] < start_times[j] - 1e-9:
                c = j
                c_index = idx
                break

        if c == 0:
            # Все работы на критическом пути начинаются ровно в своё время освобождения
            return {
                'critical_path': critical_path,
                'c': 0,
                'J': set(),
                'start_times': start_times
            }

        # J = работы на критическом пути после c
        J = set(critical_path[c_index + 1:])

        return {
            'critical_path': critical_path,
            'c': c,
            'J': J,
            'c_index': c_index,
            'start_times': start_times
        }

    # =====================================================================
    # ПРОВЕРКА УСЛОВИЙ СИЛЬНОГО ВЕТВЛЕНИЯ (Теорема 3.1)
    # =====================================================================

    def _check_strong_branching_conditions(self, critical_info: Dict, data: Dict,
                                           start_times: Dict[int, float]) -> bool:
        """
        Проверка условий Теоремы 3.1 (Theorem 3.1).

        Условия для сильного ветвления:
        1. Сегмент C(c, n) не содержит precedence дуг (т.е. DPC дуг, включая с l_ij = d_i).
           Стандартные дуги расписания (j начинается сразу после i) — НЕ считаются precedence arcs.
        2. r_i >= max(t_i, t_c) для всех i ∈ J.
        """
        critical_path = critical_info['critical_path']
        c = critical_info['c']
        J = critical_info['J']

        if c == 0 or not J:
            return False

        try:
            c_index = critical_path.index(c)
        except ValueError:
            return False

        # Условие 1: сегмент C(c, n) = critical_path[c_index:] не содержит precedence дуг
        segment = critical_path[c_index:]
        for k in range(len(segment) - 1):
            u = segment[k]
            v = segment[k + 1]
            if (u, v) in self.dpc_pairs:
                return False
            if v in data['sigma'].get(u, set()):
                return False

        # Условие 2: r_i >= max(t_i, t_c) для всех i ∈ J
        t_c = start_times[c]
        for i in J:
            t_i = start_times[i]
            if data['r'][i] < max(t_i, t_c) - 1e-9:
                return False

        return True

    # =====================================================================
    # СИЛЬНОЕ ВЕТВЛЕНИЕ
    # =====================================================================

    def _apply_strong_branching(self, data: Dict, c: int, J: Set[int],
                                upper_bound: float, depth: int) -> None:
        """
        Применяет сильное ветвление: c -> J и J -> c.
        Раздел 4, шаг 4a.
        """
        # Применяем тесты Карлье
        if self._carlier_tests(data, c, J, upper_bound):
            self._update_heads_and_tails(data)
            self._branch_and_bound(data, upper_bound, depth)
            return

        # Ветвь 1: c предшествует всем работам из J
        data1 = self._copy_data(data)
        for j in J:
            self._add_precedence(data1, c, j)
            required_rj = data1['r'][c] + self.jobs[c].d_i
            if (c, j) in self.dpc_pairs:
                required_rj = data1['r'][c] + self.l_matrix[c][j]
            if required_rj > data1['r'][j]:
                data1['r'][j] = required_rj
            required_qc = data1['q'][j] + self.jobs[j].d_i
            if (c, j) in self.dpc_pairs:
                required_qc = data1['q'][j] + self.l_matrix[c][j]
            if required_qc > data1['q'][c]:
                data1['q'][c] = required_qc
        self._update_heads_and_tails(data1)
        self._branch_and_bound(data1, upper_bound, depth)

        # Ветвь 2: c следует за всеми работами из J
        data2 = self._copy_data(data)
        for j in J:
            self._add_precedence(data2, j, c)
            required_rc = data2['r'][j] + self.jobs[j].d_i
            if (j, c) in self.dpc_pairs:
                required_rc = data2['r'][j] + self.l_matrix[j][c]
            if required_rc > data2['r'][c]:
                data2['r'][c] = required_rc
            required_qj = data2['q'][c] + self.jobs[c].d_i
            if (j, c) in self.dpc_pairs:
                required_qj = data2['q'][c] + self.l_matrix[j][c]
            if required_qj > data2['q'][j]:
                data2['q'][j] = required_qj
        self._update_heads_and_tails(data2)
        self._branch_and_bound(data2, upper_bound, depth)

    def _carlier_tests(self, data: Dict, c: int, J: Set[int], upper_bound: float) -> bool:
        """
        Логические тесты Карлье (Carlier, 1982).
        Раздел 4, шаг 4a статьи.
        """
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
            # Test 1
            if r[c] + self.jobs[c].d_i + sum_d_J + q[k] >= upper_bound - 1e-6:
                for j in J:
                    if self._add_precedence(data, j, k):
                        constraints_added = True
                if constraints_added:
                    required_rk = r[c] + self.jobs[c].d_i
                    for j in J:
                        required_rk = max(required_rk, r[j] + self.jobs[j].d_i)
                        if (j, k) in self.dpc_pairs:
                            required_rk = max(required_rk, r[j] + self.l_matrix[j][k])
                    if required_rk > r[k]:
                        r[k] = required_rk

            # Test 2
            if min_r_J + sum_d_J + self.jobs[k].d_i + q[k] >= upper_bound - 1e-6:
                for j in J:
                    if self._add_precedence(data, k, j):
                        constraints_added = True
                if constraints_added:
                    required_qk = self.jobs[c].d_i  # безопасное начальное значение
                    for j in J:
                        required_qk = max(required_qk, q[j] + self.jobs[j].d_i)
                        if (k, j) in self.dpc_pairs:
                            required_qk = max(required_qk, q[j] + self.l_matrix[k][j])
                    if required_qk > q[k]:
                        q[k] = required_qk

        return constraints_added

    # =====================================================================
    # ОБРАТНАЯ ЗАДАЧА
    # =====================================================================

    def _create_reverse_data(self, data: Dict) -> Dict:
        """
        Создаёт обратную задачу: меняет r ↔ q, σ ↔ π, инвертирует DPC.
        """
        rev_data = {
            'r': {j: data['q'][j] for j in self.jobs},
            'q': {j: data['r'][j] for j in self.jobs},
            'sigma': {k: set(v) for k, v in data.get('pi', {}).items()},
            'pi': {k: set(v) for k, v in data.get('sigma', {}).items()},
        }
        return rev_data

    def _create_reverse_dpc(self) -> Tuple[Dict, Set]:
        """
        Создаёт инвертированные DPC матрицу и множество пар.
        L_rev(j, i) = L(i, j) - d_i + d_j
        """
        rev_l_matrix = defaultdict(lambda: defaultdict(float))
        rev_dpc_pairs = set()

        for (i, j) in self.dpc_pairs:
            lij = self.l_matrix[i][j]
            rev_lji = lij - self.jobs[i].d_i + self.jobs[j].d_i
            rev_l_matrix[j][i] = max(rev_lji, self.jobs[j].d_i)
            rev_dpc_pairs.add((j, i))

        return rev_l_matrix, rev_dpc_pairs

    def _build_reverse_incoming_outgoing(self, rev_l_matrix, rev_dpc_pairs):
        """Строит списки входящих/исходящих DPC для обратной задачи."""
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for (i, j) in rev_dpc_pairs:
            if i in self.jobs and j in self.jobs:
                lij = rev_l_matrix[i][j]
                incoming[j].append((i, lij))
                outgoing[i].append((j, lij))
        return incoming, outgoing

    def _try_reverse_strong_branching(self, data: Dict, upper_bound: float, depth: int) -> bool:
        """
        Пытается применить сильное ветвление к обратной задаче.
        Раздел 4, шаг 4.
        """
        # Создаём обратные данные
        reverse_data = self._create_reverse_data(data)
        rev_l_matrix, rev_dpc_pairs = self._create_reverse_dpc()
        rev_incoming, rev_outgoing = self._build_reverse_incoming_outgoing(rev_l_matrix, rev_dpc_pairs)

        # Сохраняем оригинальные атрибуты
        original_l_matrix = self.l_matrix
        original_dpc_pairs = self.dpc_pairs
        original_incoming = self._incoming_dpc
        original_outgoing = self._outgoing_dpc

        # Подменяем на обратные
        self.l_matrix = rev_l_matrix
        self.dpc_pairs = rev_dpc_pairs
        self._incoming_dpc = rev_incoming
        self._outgoing_dpc = rev_outgoing

        try:
            # Усиливаем головы/хвосты в обратной задаче
            self._update_heads_and_tails(reverse_data)

            # Запускаем LTH на обратной задаче
            rev_schedule, rev_makespan, rev_starts = self._longest_tail_heuristic(reverse_data)

            if not rev_schedule:
                return False

            # Ищем критический путь
            rev_critical = self._find_critical_path(rev_schedule, rev_starts, reverse_data)
            if rev_critical is None or rev_critical['c'] == 0:
                return False

            # Проверяем условия сильного ветвления
            if not self._check_strong_branching_conditions(rev_critical, reverse_data, rev_starts):
                return False

            # Условия соблюдены! Запускаем стандартное ветвление,
            # но теперь оно будет работать с обратными атрибутами
            rev_c = rev_critical['c']
            rev_J = rev_critical['J']

            # Это рекурсивно создаст и решит подзадачи ОБРАТНОЙ задачи
            self._apply_strong_branching(reverse_data, rev_c, rev_J, upper_bound, depth)

            return True
        finally:
            # Гарантированно восстанавливаем оригинальные атрибуты
            self.l_matrix = original_l_matrix
            self.dpc_pairs = original_dpc_pairs
            self._incoming_dpc = original_incoming
            self._outgoing_dpc = original_outgoing

    # =====================================================================
    # СЛАБОЕ ВЕТВЛЕНИЕ
    # =====================================================================

    def _apply_weak_branching(self, data: Dict, critical_info: Dict,
                              upper_bound: float, depth: int) -> None:
        """Применяет слабое ветвление."""
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
        """Создаёт и исследует подзадачу с отношением i -> j."""
        data_copy = self._copy_data(data)
        self._add_precedence(data_copy, i, j)
        # Обновление головы j
        required_rj = data_copy['r'][i] + self.jobs[i].d_i
        if (i, j) in self.dpc_pairs:
            required_rj = data_copy['r'][i] + self.l_matrix[i][j]
        if required_rj > data_copy['r'][j]:
            data_copy['r'][j] = required_rj
        # Обновление хвоста i (симметрично)
        required_qi = data_copy['q'][j] + self.jobs[j].d_i
        if (i, j) in self.dpc_pairs:
            required_qi = data_copy['q'][j] + self.l_matrix[i][j]
        if required_qi > data_copy['q'][i]:
            data_copy['q'][i] = required_qi
        self._update_heads_and_tails(data_copy)
        self._branch_and_bound(data_copy, upper_bound, depth)

    def _select_weak_branching_pair(self, data: Dict,
                                    critical_info: Dict) -> Tuple[Optional[int], Optional[int]]:
        """
        Выбирает пару работ (i, j) для слабого ветвления.
        Строго по разделу 4, шаг 4b.
        """
        critical_path = critical_info['critical_path']
        c = critical_info['c']
        start_times = critical_info['start_times']
        C_max = self.best_makespan

        if not critical_path:
            return None, None

        # Сначала ищем критический путь без precedence дуг в C(c, n)
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
            # Случай 2: essential precedence дуги
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
        """
        Ищет критический путь, у которого сегмент C(c, n) не содержит precedence дуг.
        """
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
        """
        Находит все критические пути в графе G(S).
        Возвращает список из одного пути (упрощение).
        """
        schedule = sorted(start_times.keys(), key=lambda j: start_times[j])
        critical_info = self._find_critical_path(schedule, start_times, data)
        return [critical_info] if critical_info else []

    def _is_essential_precedence_arc(self, u: int, v: int, data: Dict,
                                     start_times: Dict[int, float],
                                     C_max: float) -> bool:
        """Проверяет, является ли дуга (u, v) существенной (essential)."""
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
        """Проверяет, упорядочены ли уже работы i и j."""
        sigma = data['sigma']
        pi = data['pi']
        if j in sigma.get(i, set()) or i in pi.get(j, set()):
            return True
        if (i, j) in self.dpc_pairs:
            return True
        return False

    # =====================================================================
    # ПОСТОБРАБОТКА (Propositions 3.3 и 3.4)
    # =====================================================================

    def _postprocess(self, data: Dict, schedule: List[int],
                     start_times: Dict[int, float], C_max: float) -> bool:
        """
        Применяет Proposition 3.3 и 3.4 для усиления ограничений.
        """
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
                        required_rk = t_j + self.jobs[j].d_i
                        if (j, k) in self.dpc_pairs:
                            required_rk = t_j + self.l_matrix[j][k]
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
                        required_qk = q_i + self.jobs[i].d_i
                        if (k, i) in self.dpc_pairs:
                            required_qk = q_i + self.l_matrix[k][i]
                        if required_qk > q[k]:
                            q[k] = required_qk
                        changed = True

        return changed

    # =====================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =====================================================================

    def _add_precedence(self, data: Dict, i: int, j: int) -> bool:
        """
        Добавляет отношение i -> j с транзитивным замыканием.
        Возвращает False, если создаётся цикл.
        """
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
        """Создаёт глубокую копию данных подзадачи."""
        return {
            'r': data['r'].copy(),
            'q': data['q'].copy(),
            'sigma': {k: set(v) for k, v in data['sigma'].items()},
            'pi': {k: set(v) for k, v in data['pi'].items()},
        }