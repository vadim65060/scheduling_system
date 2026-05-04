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
        # Используем self.dpc_pairs из базового Scheduler
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
        """
        Основная рекурсивная процедура алгоритма Bal.
        Соответствует шагам 2-4 Algorithm 1.
        """
        self.nodes_explored += 1

        # --- Проверка лимита времени ---
        if time.time() - self.start_time > self.time_limit:
            self.timed_out = True
            return

        # --- Проверка глубины ---
        if depth > 3 * self.n:
            return

        # --- Вычисление нижней границы ---
        lb = self._calculate_lower_bound(data)
        if lb >= upper_bound - 1e-6:
            self.pruned_by_bound += 1
            return

        # --- Шаг 2. Построение расписания эвристикой LTH ---
        schedule, makespan, start_times = self._longest_tail_heuristic(data)

        # Обновление лучшего решения
        if makespan < self.best_makespan - 1e-9:
            self.best_makespan = makespan
            self.best_schedule = schedule.copy()
            upper_bound = min(upper_bound, makespan)
            if lb >= makespan - 1e-6:
                return

        # --- Шаг 3. Постобработка (Propositions 3.3 и 3.4) ---
        max_postprocess_iterations = 100
        pp_iter = 0
        while pp_iter < max_postprocess_iterations:
            pp_iter += 1
            changed = self._postprocess(data, schedule, start_times, makespan)
            if not changed:
                break
            schedule, makespan, start_times = self._longest_tail_heuristic(data)
            if makespan < self.best_makespan - 1e-9:
                self.best_makespan = makespan
                self.best_schedule = schedule.copy()
                upper_bound = min(upper_bound, makespan)

        # --- Шаг 4. Ветвление ---
        critical_info = self._find_critical_path(schedule, start_times, makespan, data)
        if critical_info is None:
            return

        c = critical_info['c']
        J = critical_info['J']

        # Случай c = 0: расписание оптимально для данной подзадачи
        if c == 0:
            return

        # Проверка условий сильного ветвления (Теорема 3.1)
        can_use_strong = self._check_strong_branching_conditions(
            critical_info, data, start_times
        )

        if can_use_strong:
            self.strong_branches += 1
            self._apply_strong_branching(data, c, J, upper_bound, depth + 1)
        else:
            # Попытка обратной задачи
            reverse_data = self._create_reverse_problem(data)

            # СОХРАНЯЕМ оригинальные l_matrix и dpc_pairs
            original_l_matrix = self.l_matrix
            original_dpc_pairs = self.dpc_pairs
            original_incoming = self._incoming_dpc
            original_outgoing = self._outgoing_dpc

            # ПОДМЕНЯЕМ на обратные
            self.l_matrix = reverse_data['l_matrix']
            self.dpc_pairs = reverse_data['dpc_pairs']
            self._incoming_dpc = defaultdict(list)
            self._outgoing_dpc = defaultdict(list)
            for (i, j) in self.dpc_pairs:
                if i in self.jobs and j in self.jobs:
                    lij = self.l_matrix[i][j]
                    self._incoming_dpc[j].append((i, lij))
                    self._outgoing_dpc[i].append((j, lij))

            self._update_heads_and_tails(reverse_data)
            rev_schedule, rev_makespan, rev_starts = self._longest_tail_heuristic(reverse_data)

            rev_critical = self._find_critical_path(
                rev_schedule, rev_starts, rev_makespan, reverse_data
            )

            # ВОССТАНАВЛИВАЕМ оригинальные
            self.l_matrix = original_l_matrix
            self.dpc_pairs = original_dpc_pairs
            self._incoming_dpc = original_incoming
            self._outgoing_dpc = original_outgoing

            if (rev_critical is not None and
                    self._check_strong_branching_conditions(
                        rev_critical, reverse_data, rev_starts
                    )):
                self.strong_branches += 1
                rev_c = rev_critical['c']
                rev_J = rev_critical['J']
                self._apply_strong_branching_reversed(data, rev_c, rev_J, upper_bound, depth + 1)
            else:
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

    def _all_predecessors_scheduled(self, j: int, scheduled: Set[int], data: Dict) -> bool:
        """
        Проверяет, что все предшественники работы j уже запланированы.
        Учитывает исходные DPC (через _incoming_dpc) и sigma-отношения (через pi[j]).
        """
        # Проверяем исходные DPC
        for (i, _) in self._incoming_dpc.get(j, []):
            if i not in scheduled:
                return False

        # Проверяем pi-предшественников (pi[j] = те, кто должен быть перед j)
        for pred in data['pi'].get(j, set()):
            if pred not in scheduled:
                return False

        return True

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
                            C_max: float, data: Dict) -> Optional[Dict]:
        """
        Находит критический путь в графе G(S).
        Оптимизированная версия с прямым проходом по расписанию.
        """
        if not schedule:
            return None

        n = len(schedule)
        N = n + 2
        source = n
        sink = n + 1

        dp = [-float('inf')] * N
        parent = [-1] * N
        dp[source] = 0.0

        # Дуги source -> работа
        for idx, j in enumerate(schedule):
            w = max(data['r'][j], start_times[j])
            if w > dp[idx]:
                dp[idx] = w
                parent[idx] = source

        # Предварительно строим обратный индекс для быстрого поиска DPC
        job_to_idx = {j: idx for idx, j in enumerate(schedule)}

        # Дуги между работами
        for idx_i in range(n):
            i = schedule[idx_i]

            # 1. Обычная дуга к следующей работе (всегда существует)
            if idx_i + 1 < n:
                j_next = schedule[idx_i + 1]
                w = dp[idx_i] + self.jobs[i].d_i
                if w > dp[idx_i + 1]:
                    dp[idx_i + 1] = w
                    parent[idx_i + 1] = idx_i

            # 2. DPC-дуги от i ко всем j, идущим позже в расписании
            for (j, lij) in self._outgoing_dpc.get(i, []):
                idx_j = job_to_idx.get(j)
                if idx_j is not None and idx_j > idx_i:
                    w = dp[idx_i] + lij
                    if w > dp[idx_j]:
                        dp[idx_j] = w
                        parent[idx_j] = idx_i

        # Дуги работа -> sink
        for idx, j in enumerate(schedule):
            w = dp[idx] + self.jobs[j].d_i + data['q'][j]
            if w > dp[sink]:
                dp[sink] = w
                parent[sink] = idx

        # Восстановление критического пути
        path = []
        cur = sink
        while cur != source and cur >= 0:
            if cur < n:
                path.insert(0, schedule[cur])
            cur = parent[cur]

        if not path:
            return None

        # Поиск c: первая работа с r_i < s_i
        c = 0
        c_index = 0
        for idx, j in enumerate(path):
            if data['r'][j] < start_times[j] - 1e-9:
                c = j
                c_index = idx
                break

        if c == 0:
            return {
                'critical_path': path,
                'c': 0,
                'J': set(),
                'start_times': start_times
            }

        J = set(path[c_index + 1:])

        return {
            'critical_path': path,
            'c': c,
            'J': J,
            'start_times': start_times
        }

    # =====================================================================
    # ПРОВЕРКА УСЛОВИЙ СИЛЬНОГО ВЕТВЛЕНИЯ (Теорема 3.1)
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

        # Проверка 1: сегмент от c до конца не содержит precedence дуг
        segment = critical_path[c_index:]
        for k in range(len(segment) - 1):
            u, v = segment[k], segment[k + 1]
            # DPC-дуга
            if (u, v) in self.dpc_pairs:
                return False
            # Sigma-дуга (добавленное отношение)
            if v in data['sigma'].get(u, set()):
                return False
            # Обычная дуга (v начинается сразу после u)
            if abs(start_times[v] - (start_times[u] + self.jobs[u].d_i)) < 1e-9:
                return False

        # Проверка 2: r_i >= max(t_i, t_c) для всех i из J
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
        """Применяет сильное ветвление с тестами Карлье."""
        if self._carlier_tests(data, c, J, upper_bound):
            self.pruned_by_test += 1
            return

        # Ветвь 1: c -> все J (c precedes all J)
        # σ(c) := σ(c) ∪ J
        data1 = self._copy_data(data)
        for j in J:
            self._add_precedence(data1, c, j)
            # Обновляем r_j
            required_rj = data1['r'][c] + self.jobs[c].d_i
            if (c, j) in self.dpc_pairs:
                required_rj = data1['r'][c] + self.l_matrix[c][j]
            if required_rj > data1['r'][j]:
                data1['r'][j] = required_rj
            # Симметрично обновляем q_c
            required_qc = data1['q'][j] + self.jobs[j].d_i
            if (c, j) in self.dpc_pairs:
                required_qc = data1['q'][j] + self.l_matrix[c][j]
            if required_qc > data1['q'][c]:
                data1['q'][c] = required_qc
        self._update_heads_and_tails(data1)
        self._branch_and_bound(data1, upper_bound, depth)

        # Ветвь 2: все J -> c (c succeeds all J)
        # π(c) := π(c) ∪ J
        data2 = self._copy_data(data)
        for j in J:
            self._add_precedence(data2, j, c)
            # Обновляем r_c (J предшествуют c, поэтому голова c зависит от J)
            required_rc = data2['r'][j] + self.jobs[j].d_i
            if (j, c) in self.dpc_pairs:
                required_rc = data2['r'][j] + self.l_matrix[j][c]
            if required_rc > data2['r'][c]:
                data2['r'][c] = required_rc
            # Симметрично обновляем q_j
            required_qj = data2['q'][c] + self.jobs[c].d_i
            if (j, c) in self.dpc_pairs:
                required_qj = data2['q'][c] + self.l_matrix[j][c]
            if required_qj > data2['q'][j]:
                data2['q'][j] = required_qj
        self._update_heads_and_tails(data2)
        self._branch_and_bound(data2, upper_bound, depth)

    def _carlier_tests(self, data: Dict, c: int, J: Set[int], upper_bound: float) -> bool:
        """Логические тесты Карлье."""
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

        for k in K:
            # Тест 1: r_c + d_c + sum_d_J + q_k >= UB  =>  k succeeds J (J -> k)
            if r[c] + self.jobs[c].d_i + sum_d_J + q[k] >= upper_bound - 1e-6:
                for j in J:
                    self._add_precedence(data, j, k)  # J предшествуют k
                return True

            # Тест 2: min_r_J + sum_d_J + d_k + q_k >= UB  =>  k precedes J (k -> J)
            if min_r_J + sum_d_J + self.jobs[k].d_i + q[k] >= upper_bound - 1e-6:
                for j in J:
                    self._add_precedence(data, k, j)  # k предшествует J
                return True

        return False

    # =====================================================================
    # ОБРАТНАЯ ЗАДАЧА
    # =====================================================================

    def _create_reverse_problem(self, data: Dict) -> Dict:
        """
        Создаёт обратную задачу: меняет r<->q, sigma<->pi, и инвертирует l_matrix.
        L_rev(j, i) = L(i, j) - d_i + d_j  (как в статье, стр. 10)
        """
        r = data['r']
        q = data['q']

        # Инвертированная матрица задержек
        rev_l_matrix = defaultdict(lambda: defaultdict(float))
        for (i, j) in self.dpc_pairs:
            lij = self.l_matrix[i][j]
            rev_lji = lij - self.jobs[i].d_i + self.jobs[j].d_i
            rev_l_matrix[j][i] = rev_lji

        # Инвертированные пары DPC
        rev_dpc_pairs = {(j, i) for (i, j) in self.dpc_pairs}

        return {
            'r': {j: q[j] for j in self.jobs},
            'q': {j: r[j] for j in self.jobs},
            'sigma': {k: set(v) for k, v in data['pi'].items()},
            'pi': {k: set(v) for k, v in data['sigma'].items()},
            'l_matrix': rev_l_matrix,
            'dpc_pairs': rev_dpc_pairs,
        }

    def _apply_strong_branching_reversed(self, data: Dict, c: int, J: Set[int],
                                         upper_bound: float, depth: int) -> None:
        """Применяет сильное ветвление, полученное из обратной задачи."""
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
                self.pruned_by_test += 1
                return

        # Ветвь 1: все J -> c (J precede c)
        data1 = self._copy_data(data)
        for j in J:
            self._add_precedence(data1, j, c)
            # Обновляем r_c
            required_rc = data1['r'][j] + self.jobs[j].d_i
            if (j, c) in self.dpc_pairs:
                required_rc = data1['r'][j] + self.l_matrix[j][c]
            if required_rc > data1['r'][c]:
                data1['r'][c] = required_rc
            # Симметрично обновляем q_j
            required_qj = data1['q'][c] + self.jobs[c].d_i
            if (j, c) in self.dpc_pairs:
                required_qj = data1['q'][c] + self.l_matrix[j][c]
            if required_qj > data1['q'][j]:
                data1['q'][j] = required_qj
        self._update_heads_and_tails(data1)
        self._branch_and_bound(data1, upper_bound, depth)

        # Ветвь 2: c -> все J (c precede J)
        data2 = self._copy_data(data)
        for j in J:
            self._add_precedence(data2, c, j)
            # Обновляем r_j
            required_rj = data2['r'][c] + self.jobs[c].d_i
            if (c, j) in self.dpc_pairs:
                required_rj = data2['r'][c] + self.l_matrix[c][j]
            if required_rj > data2['r'][j]:
                data2['r'][j] = required_rj
            # Симметрично обновляем q_c
            required_qc = data2['q'][j] + self.jobs[j].d_i
            if (c, j) in self.dpc_pairs:
                required_qc = data2['q'][j] + self.l_matrix[c][j]
            if required_qc > data2['q'][c]:
                data2['q'][c] = required_qc
        self._update_heads_and_tails(data2)
        self._branch_and_bound(data2, upper_bound, depth)

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
        required_rj = data_copy['r'][i] + self.jobs[i].d_i
        if (i, j) in self.dpc_pairs:
            required_rj = data_copy['r'][i] + self.l_matrix[i][j]
        if required_rj > data_copy['r'][j]:
            data_copy['r'][j] = required_rj
        self._update_heads_and_tails(data_copy)
        self._branch_and_bound(data_copy, upper_bound, depth)

    def _select_weak_branching_pair(self, data: Dict,
                                    critical_info: Dict) -> Tuple[Optional[int], Optional[int]]:
        """Выбирает пару работ для слабого ветвления."""
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
                if r[node] < max(start_times[node], t_j) - 1e-9:
                    if not self._are_ordered_pair(node, j, data):
                        return node, j
            return None, None
        else:
            k, l = essential_arc
            j = l
            try:
                l_index = critical_path.index(l)
            except ValueError:
                return None, None
            segment_l = critical_path[l_index:]
            for node in segment_l:
                if node != l and r[node] < start_times[l] - 1e-9:
                    if not self._are_ordered_pair(node, j, data):
                        return node, j
            return None, None

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
        """Применяет Proposition 3.3 и 3.4 для усиления ограничений."""
        if not schedule:
            return False

        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']
        pi = data['pi']

        # Находим критический путь
        critical_info = self._find_critical_path(schedule, start_times, C_max, data)
        if critical_info is None or critical_info['c'] == 0:
            return False

        critical_path = critical_info['critical_path']

        # Предварительно вычисляем essential дуги на критическом пути
        cp_has_essential = [False] * len(critical_path)
        for k in range(len(critical_path) - 1):
            u, v = critical_path[k], critical_path[k + 1]
            cp_has_essential[k] = self._is_essential_precedence_arc(u, v, data, start_times, C_max)

        # Префиксные суммы для быстрой проверки сегментов критического пути
        cp_prefix = [False] * (len(critical_path) + 1)
        for i in range(1, len(critical_path)):
            cp_prefix[i + 1] = cp_prefix[i] or cp_has_essential[i - 1]

        def cp_segment_has_essential(start: int, end: int) -> bool:
            if start >= end:
                return False
            return cp_prefix[end] != cp_prefix[start + 1] or cp_has_essential[start]

        # Proposition 3.3 (прямой проход по критическому пути)
        for idx, j in enumerate(critical_path):
            K = critical_path[idx + 1:]
            if not K:
                continue

            if cp_segment_has_essential(idx, len(critical_path) - 1):
                continue

            q_j = q[j]
            t_j = start_times[j]

            # Проверяем условия: q_k >= q_j для всех k, и r_k >= t_j для k не в σ(j)
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

        # Proposition 3.4 (обратный проход по критическому пути)
        for idx in range(len(critical_path) - 1, -1, -1):
            i = critical_path[idx]
            K = critical_path[:idx]
            if not K:
                continue

            if cp_segment_has_essential(0, idx):
                continue

            r_i = r[i]
            q_i = q[i]
            t_i = start_times[i]

            # Проверяем условия: r_k >= r_i для всех k, и q_k >= q_i для k не в π(i)
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
                        required_qk = q[i] + self.jobs[i].d_i
                        if (k, i) in self.dpc_pairs:
                            required_qk = q[i] + self.l_matrix[k][i]
                        if required_qk > q[k]:
                            q[k] = required_qk
                        changed = True

        return changed

    def _segment_has_essential(self, data: Dict, schedule: List[int],
                               start_idx: int, end_idx: int,
                               start_times: Dict[int, float], C_max: float) -> bool:
        """Проверяет, содержит ли сегмент расписания существенные дуги."""
        for k_idx in range(start_idx, end_idx):
            u, v = schedule[k_idx], schedule[k_idx + 1]
            if self._is_essential_precedence_arc(u, v, data, start_times, C_max):
                return True
        return False

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

        # Инициализация множеств при необходимости
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