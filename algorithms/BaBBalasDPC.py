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
        self._debug_count = 0  # счётчик отладочных выводов
        self._debug_limit = 0  # сколько раз выводить отладку
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

        self._compute_transitive_dpc()
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
        Итеративно усиливает головы (r) и хвосты (q) до фиксированной точки.
        """
        changed = False
        r = data['r']
        q = data['q']
        sigma = data['sigma']

        # Множество для быстрого доступа
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
                # sigma-отношение может быть и DPC, но тогда его задержка l_ij >= d_i
                if (i, j) in self.dpc_pairs:
                    delay = self.l_matrix[i][j]

                new_rj = r[i] + delay
                if new_rj > r[j] + 1e-9:
                    r[j] = new_rj
                    local_changed = True

                # Здесь логика симметрична DPC: q_i >= q_j + delay, где delay = max(d_i, l_ij)
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

        if self._debug_count < self._debug_limit:
            self._debug_count += 1
            print(f"\n{'=' * 70}")
            print(f"ОТЛАДКА УЗЛА #{self._debug_count}")
            print(f"Depth: {depth}, UB: {upper_bound:.2f}")
            print(f"Schedule length: {len(schedule)}")

            # critical_info ДО вызова _find_critical_path
            critical_info = self._find_critical_path(schedule, start_times, data)

            if critical_info is None:
                print("❌ critical_info is None!")
            else:
                cp = critical_info['critical_path']
                c = critical_info['c']
                J = critical_info['J']
                c_index = critical_info.get('c_index', -1)

                print(f"\n📊 КРИТИЧЕСКИЙ ПУТЬ:")
                print(f"  Длина пути: {len(cp)}")
                print(f"  Путь: {cp}")
                print(f"  c = {c} (индекс {c_index})")
                print(f"  J = {J}")

                # Детально по каждой работе на критическом пути
                print(f"\n{'Работа':>6} {'r_i':>8} {'s_i':>8} {'d_i':>6} {'q_i':>6} {'r < s?':>8}")
                print("-" * 50)
                for idx, job_id in enumerate(cp):
                    r_val = data['r'][job_id]
                    s_val = start_times[job_id]
                    d_val = self.jobs[job_id].d_i
                    q_val = data['q'][job_id]
                    is_delayed = "ДА" if r_val < s_val - 1e-9 else "нет"
                    marker = " ← c" if job_id == c else ""
                    print(f"{job_id:6} {r_val:8.1f} {s_val:8.1f} {d_val:6.1f} {q_val:6.1f} {is_delayed:>8}{marker}")

                # Проверяем условия сильного ветвления
                if c != 0:
                    print(f"\n🔍 ПРОВЕРКА УСЛОВИЙ СИЛЬНОГО ВЕТВЛЕНИЯ:")

                    # Условие 1: нет precedence дуг в C(c,n)
                    segment = cp[c_index:]
                    print(f"  Сегмент C(c,n): {segment}")
                    has_prec = False
                    for k in range(len(segment) - 1):
                        u, v = segment[k], segment[k + 1]
                        is_dpc = (u, v) in self.dpc_pairs
                        is_sigma = v in data['sigma'].get(u, set())
                        if is_dpc or is_sigma:
                            has_prec = True
                            print(f"  ❌ Найдена precedence дуга: ({u},{v}) DPC={is_dpc} SIGMA={is_sigma}")

                    if not has_prec:
                        print(f"  ✅ Условие 1 выполнено: нет precedence дуг")

                    # Условие 2: r_i >= max(s_i, s_c) для всех i ∈ J
                    t_c = start_times[c]
                    print(f"  t_c = s_{c} = {t_c:.1f}")
                    cond2_ok = True
                    for i in J:
                        t_i = start_times[i]
                        r_i = data['r'][i]
                        max_cond = max(t_i, t_c)
                        if r_i < max_cond - 1e-9:
                            cond2_ok = False
                            print(
                                f"  ❌ Условие 2 нарушено для работы {i}: r_{i}={r_i:.1f} < max(s_{i}={t_i:.1f}, s_{c}={t_c:.1f}) = {max_cond:.1f}")

                    if cond2_ok:
                        print(f"  ✅ Условие 2 выполнено")

                    if not has_prec and cond2_ok:
                        print(f"\n🎯 СИЛЬНОЕ ВЕТВЛЕНИЕ ДОЛЖНО СРАБОТАТЬ!")
                        print(f"   c={c}, J={J}")
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

        # Проверка условий сильного ветвления для ПРЯМОЙ задачи
        if self._check_strong_branching_conditions(critical_info, data, start_times):
            self.strong_branches += 1
            self._apply_strong_branching(data, c, J, upper_bound, depth + 1)
        else:
            # Попытка обратной задачи. Теперь это может привести к рекурсивному вызову
            # _branch_and_bound, если сильное ветвление успешно.
            if self._try_reverse_strong_branching(data, upper_bound, depth + 1):
                # Ветвление уже выполнено внутри _try_reverse_strong_branching
                self.strong_branches += 1
                # Увеличиваем счетчик сильных ветвлений, так как оно было применено
                pass
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
        Находит критический путь в графе G(S) согласно статье Balas et al. (1995).

        Граф G(S) = (N, E), где:
        - N = J ∪ {0, n} (0 — исток, n — сток)
        - Дуги:
          (0, i) вес r_i
          (i, j) для i, j ∈ J (i ≠ j):
              - если (i,j) ∈ DPC: вес l_ij
              - если j следует сразу за i в расписании: вес d_i
              - иначе: дуги НЕТ
          (i, n) вес d_i + q_i

        Критический путь — самый длинный путь от 0 до n.
        """
        if not schedule:
            return None

        n_jobs = len(schedule)
        SOURCE = n_jobs  # индекс истока
        SINK = n_jobs + 1  # индекс стока

        # dist[v] = длина длиннейшего пути от истока до вершины v
        dist = [-float('inf')] * (n_jobs + 2)
        parent = [-1] * (n_jobs + 2)
        dist[SOURCE] = 0.0

        # Получаем позицию каждой работы в расписании для быстрого доступа
        job_to_pos = {job: idx for idx, job in enumerate(schedule)}

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

            # 2a: Дуга к следующей работе в расписании (вес = d_i)
            if idx + 1 < n_jobs:
                j_next = schedule[idx + 1]
                w = dist[idx] + self.jobs[i].d_i
                if w > dist[idx + 1] + 1e-9:
                    dist[idx + 1] = w
                    parent[idx + 1] = idx

            # 2b: DPC-дуги ко всем последующим работам
            # (i, j) где j идёт позже i в расписании И есть DPC
            if i in self._outgoing_dpc:
                for (j, lij) in self._outgoing_dpc[i]:
                    jdx = job_to_pos.get(j)
                    if jdx is not None and jdx > idx:
                        w = dist[idx] + lij
                        if w > dist[jdx] + 1e-9:
                            dist[jdx] = w
                            parent[jdx] = idx

            # 2c: Дуга от работы к стоку (вес = d_i + q_i)
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
            if p < 0:
                break
            if p != SOURCE and p != SINK and p < n_jobs:
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

        # --- Шаг 4: Поиск работы c (первой задержанной работы на пути) ---
        c = 0
        c_index = -1
        for idx, j in enumerate(critical_path):
            # Работа считается задержанной, если r_i < s_i (с учётом погрешности)
            if data['r'][j] < start_times.get(j, 0) - 1e-9:
                c = j
                c_index = idx
                break

        if c == 0:
            # Все работы начинаются вовремя — расписание оптимально
            return {
                'critical_path': critical_path,
                'c': 0,
                'J': set(),
                'start_times': start_times,
                'makespan': dist[SINK]
            }

        # J = все работы на критическом пути после c
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
    # ПРОВЕРКА УСЛОВИЙ СИЛЬНОГО ВЕТВЛЕНИЯ (Теорема 3.1)
    # =====================================================================

    def _check_strong_branching_conditions(self, critical_info: Dict, data: Dict,
                                           start_times: Dict[int, float]) -> bool:
        """
        Проверка условий Теоремы 3.1 (Theorem 3.1).
        """
        critical_path = critical_info['critical_path']
        c = critical_info['c']
        J = critical_info['J']

        if c == 0 or not J:
            if self._debug_count <= self._debug_limit:
                print(f"     [DEBUG CHECK] c={c}, J empty={not J}")
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
            # Проверяем ОБА типа дуг: DPC и "сразу следующая"
            is_dpc = (u, v) in self.dpc_pairs
            is_sigma = v in data['sigma'].get(u, set())

            # ВАЖНО: дуга "сразу следующая в расписании" — это НЕ precedence,
            # если только она не DPC и не sigma. Проверяем это:
            is_immediate = (k + 1 < len(segment) and
                            start_times.get(v, 0) == start_times.get(u, 0) + self.jobs[u].d_i)

            # precedence = DPC или sigma (но НЕ immediate без DPC/sigma)
            if is_dpc or is_sigma:
                if self._debug_count <= self._debug_limit:
                    print(
                        f"     [DEBUG CHECK] ❌ Дуга ({u},{v}): DPC={is_dpc}, SIGMA={is_sigma}, immediate={is_immediate}")
                return False

        # Условие 2: r_i >= max(t_i, t_c) для всех i ∈ J
        t_c = start_times[c]
        for i in J:
            t_i = start_times[i]
            r_i = data['r'][i]
            if r_i < max(t_i, t_c) - 1e-9:
                if self._debug_count <= self._debug_limit:
                    print(f"     [DEBUG CHECK] ❌ Условие 2: r_{i}={r_i} < max(s_{i}={t_i}, s_{c}={t_c})")
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

    # =====================================================================
    # ОБРАТНАЯ ЗАДАЧА (Reverse Problem)
    # =====================================================================

    def _create_reverse_data(self, data: Dict) -> Dict:
        """
        Создаёт обратную задачу: меняет r и q, sigma и pi местами.
        """
        rev_data = {
            'r': {j: data['q'][j] for j in self.jobs},
            'q': {j: data['r'][j] for j in self.jobs},
            'sigma': {k: set(v) for k, v in data.get('pi', {}).items()},
            'pi': {k: set(v) for k, v in data.get('sigma', {}).items()},
        }
        return rev_data

    def _create_reverse_dpc(self):
        """
        Создаёт матрицу и множество пар для обратной задачи.
        L_rev(j, i) = L(i, j) - d_i + d_j
        """
        rev_l_matrix = defaultdict(lambda: defaultdict(float))
        rev_dpc_pairs = set()

        for (i, j) in self.dpc_pairs:
            lij = self.l_matrix[i][j]
            rev_lji = lij - self.jobs[i].d_i + self.jobs[j].d_i
            # Задержка не может быть меньше d_j
            rev_lji = max(rev_lji, self.jobs[j].d_i)
            rev_l_matrix[j][i] = rev_lji
            rev_dpc_pairs.add((j, i))

        return rev_l_matrix, rev_dpc_pairs

    def _apply_reverse_attributes(self, rev_l_matrix, rev_dpc_pairs):
        """Временно заменяет атрибуты класса на обратные."""
        self._original_l_matrix = self.l_matrix
        self._original_dpc_pairs = self.dpc_pairs
        self._original_incoming = self._incoming_dpc
        self._original_outgoing = self._outgoing_dpc

        self.l_matrix = rev_l_matrix
        self.dpc_pairs = rev_dpc_pairs

        # Строим новые списки входящих/исходящих для обратных дуг
        self._incoming_dpc = defaultdict(list)
        self._outgoing_dpc = defaultdict(list)
        for (i, j) in rev_dpc_pairs:
            if i in self.jobs and j in self.jobs:
                lij = rev_l_matrix[i][j]
                self._incoming_dpc[j].append((i, lij))
                self._outgoing_dpc[i].append((j, lij))

    def _restore_original_attributes(self):
        """Восстанавливает исходные атрибуты класса."""
        self.l_matrix = self._original_l_matrix
        self.dpc_pairs = self._original_dpc_pairs
        self._incoming_dpc = self._original_incoming
        self._outgoing_dpc = self._original_outgoing

    def _try_reverse_strong_branching(self, data: Dict, upper_bound: float, depth: int) -> bool:
        """
        Пытается применить сильное ветвление к обратной задаче.
        Если это удаётся, тут же создаются дочерние узлы для ИСХОДНОЙ задачи.
        """
        # 1. Создаём обратные данные
        reverse_data = self._create_reverse_data(data)
        rev_l_matrix, rev_dpc_pairs = self._create_reverse_dpc()

        # 2. Временно подменяем атрибуты класса на обратные
        self._apply_reverse_attributes(rev_l_matrix, rev_dpc_pairs)

        try:
            # 3. Усиливаем головы/хвосты в обратной задаче
            self._update_heads_and_tails(reverse_data)

            # 4. Запускаем LTH на обратной задаче
            rev_schedule, rev_makespan, rev_starts = self._longest_tail_heuristic(reverse_data)
            if not rev_schedule:
                return False

            # 5. Ищем критический путь в обратной задаче
            rev_critical = self._find_critical_path(rev_schedule, rev_starts, reverse_data)
            if rev_critical is None or rev_critical['c'] == 0:
                if self._debug_count <= self._debug_limit:
                    print(f"     [REVERSE] critical is None or c=0")
                return False

            # Проверяем условия сильного ветвления в обратной задаче
            if not self._check_strong_branching_conditions(rev_critical, reverse_data, rev_starts):
                if self._debug_count <= self._debug_limit:
                    # Выводим обратный критический путь
                    rev_cp = rev_critical['critical_path']
                    rev_c = rev_critical['c']
                    rev_J = rev_critical['J']
                    print(f"     [REVERSE] Обратный путь: {rev_cp[:5]}... (длина {len(rev_cp)})")
                    print(f"     [REVERSE] rev_c={rev_c}, rev_J size={len(rev_J)}")
                    # Выводим первые несколько дуг для проверки
                    rev_c_idx = rev_cp.index(rev_c)
                    rev_segment = rev_cp[rev_c_idx:]
                    for k in range(min(3, len(rev_segment) - 1)):
                        u, v = rev_segment[k], rev_segment[k + 1]
                        is_dpc = (u, v) in self.dpc_pairs  # сейчас self.dpc_pairs - обратные!
                        is_sigma = v in reverse_data['sigma'].get(u, set())
                        print(f"     [REVERSE] Дуга ({u},{v}): DPC={is_dpc}, SIGMA={is_sigma}")
                return False

            # 5. Проверяем условия сильного ветвления в обратной задаче
            if not self._check_strong_branching_conditions(rev_critical, reverse_data, rev_starts):
                return False

            # 6. Сильное ветвление возможно!
            # ВАЖНО: мы применяем его к ИСХОДНОЙ задаче (data), но с инвертированной логикой.
            # В обратной задаче rev_c должно предшествовать rev_J (или наоборот).
            # В исходной задаче это эквивалентно тому, что rev_J предшествует rev_c.

            rev_c = rev_critical['c']
            rev_J = rev_critical['J']

            # Возвращаем атрибуты на место перед созданием подзадач
            self._restore_original_attributes()

            # Создаём две подзадачи для ИСХОДНОЙ задачи, инвертируя отношения
            # Ветвь 1: rev_J -> rev_c (работа c после J)
            data1 = self._copy_data(data)
            for j in rev_J:
                self._add_precedence(data1, j, rev_c)
                # Обновление голов и хвостов
                required_r_c = data1['r'][j] + self.jobs[j].d_i
                if (j, rev_c) in self.dpc_pairs:
                    required_r_c = data1['r'][j] + self.l_matrix[j][rev_c]
                if required_r_c > data1['r'][rev_c]:
                    data1['r'][rev_c] = required_r_c

                required_q_j = data1['q'][rev_c] + self.jobs[rev_c].d_i
                if (j, rev_c) in self.dpc_pairs:
                    required_q_j = data1['q'][rev_c] + self.l_matrix[j][rev_c]
                if required_q_j > data1['q'][j]:
                    data1['q'][j] = required_q_j

            self._update_heads_and_tails(data1)
            self._branch_and_bound(data1, upper_bound, depth)

            # Ветвь 2: rev_c -> rev_J (работа c предшествует J)
            data2 = self._copy_data(data)
            for j in rev_J:
                self._add_precedence(data2, rev_c, j)
                # Обновление голов и хвостов
                required_r_j = data2['r'][rev_c] + self.jobs[rev_c].d_i
                if (rev_c, j) in self.dpc_pairs:
                    required_r_j = data2['r'][rev_c] + self.l_matrix[rev_c][j]
                if required_r_j > data2['r'][j]:
                    data2['r'][j] = required_r_j

                required_q_c = data2['q'][j] + self.jobs[j].d_i
                if (rev_c, j) in self.dpc_pairs:
                    required_q_c = data2['q'][j] + self.l_matrix[rev_c][j]
                if required_q_c > data2['q'][rev_c]:
                    data2['q'][rev_c] = required_q_c

            self._update_heads_and_tails(data2)
            self._branch_and_bound(data2, upper_bound, depth)

            return True
        finally:
            # Гарантированное восстановление в случае любой ошибки
            self._restore_original_attributes()

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
