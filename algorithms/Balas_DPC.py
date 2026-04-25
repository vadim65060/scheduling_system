import copy
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import threading

from core.Algorithm import Algorithm
from core.job import Job
from core.utils import Timer


class BalasDPC(Algorithm):
    """
    Реализация алгоритма Balas–Lenstra–Vazacopoulos для
    one-machine scheduling with delayed precedence constraints (DPC),
    максимально близко к статье:

      "The One-machine Problem with Delayed Precedence Constraints
       and Its Use in Job Shop Scheduling"
       (Balas, Lenstra, Vazacopoulos, Management Science, 1995)

    ВАЖНО:
      - Никакой собственной LTH здесь нет.
      - Для получения порядка работ используется только self.heuristic_class.
      - Все времена t_i, критический путь, essential arcs и т.д. считаются внутри этого класса.
      - Makespan считается через базовый Scheduler.calculate_makespan (DPC-модель).
    """

    def __init__(self,
                 jobs: List[Job],
                 precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                 heuristic_class: type = None,
                 max_depth: int = 1000,
                 debug: bool = False):
        super().__init__(jobs, precedence_constraints)
        if heuristic_class is None:
            try:
                from algorithms.lth import LTH
                self.heuristic_class = LTH
            except Exception:
                self.heuristic_class = None
        else:
            self.heuristic_class = heuristic_class

        self.max_depth = max_depth
        self.debug = debug

        # Лучшее решение
        self.best_makespan = float('inf')
        self.best_schedule: Optional[List[int]] = None

        # Счётчики
        self.nodes_explored = 0
        self.pruning_count = 0
        self.branch_count = 0
        self.strong_branch_count = 0
        self.weak_branch_count = 0

        # L(i,j): задержки (delayed precedence), L(i,j) >= 0
        # здесь храним "чистые" L(i,j), без max(l_ij, d_i)
        self.l_matrix: Dict[int, Dict[int, float]] = defaultdict(dict)
        if precedence_constraints:
            for (i, j), val in precedence_constraints.items():
                if val is not None and val > 0:
                    self.l_matrix[i][j] = float(val)

        # π(j): предшественники j, σ(i): последователи i
        self.pi: Dict[int, Set[int]] = defaultdict(set)
        self.sigma: Dict[int, Set[int]] = defaultdict(set)
        for i, row in self.l_matrix.items():
            for j, val in row.items():
                if val > 0:
                    self.pi[j].add(i)
                    self.sigma[i].add(j)

        # visited states
        self.visited_states = set()
        self.exhaustive_search = True
        self.root_lb = 0.0

        # Параметры гибридного поведения
        self.accept_heuristic_and_stop = False
        self.extra_nodes_after_heuristic = 0
        self._extra_nodes_used = 0

    # ---------------------------
    # Логирование
    # ---------------------------

    def log(self, message: str, depth: int = 0):
        if not self.debug:
            return

        indent = "  " * depth
        line = f"{indent}{message}\n"

        # потокобезопасная запись в файл
        with self.log_lock:
            self.log_file.write(line)
            self.log_file.flush()

    def _get_precedence_dict_from_matrix(self,
                                         l_matrix: Dict[int, Dict[int, float]],
                                         allowed: Optional[Set[int]] = None) -> Dict[Tuple[int, int], float]:
        constraints = {}
        if allowed is None:
            for i, row in l_matrix.items():
                for j, val in row.items():
                    if val > 0:
                        constraints[(i, j)] = val
        else:
            for i, row in l_matrix.items():
                if i not in allowed:
                    continue
                for j, val in row.items():
                    if j in allowed and val > 0:
                        constraints[(i, j)] = val
        return constraints

    # ---------------------------
    # Основной solve
    # ---------------------------

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:

        if self.debug:
            import os
            pid = os.getpid()
            log_name = f"balas_{pid}.log"
            self.log_lock = threading.Lock()
            self.log_file = open(log_name, "a", encoding="utf-8")

        self.log("=" * 80)
        self.log(f"START FILE: {self.jobs}")
        self.log("=" * 80)
        timeout = kwargs.get('timeout', float('inf'))
        max_depth = kwargs.get('max_depth', self.max_depth)
        self.accept_heuristic_and_stop = kwargs.get('accept_heuristic_and_stop', False)
        self.extra_nodes_after_heuristic = kwargs.get('extra_nodes_after_heuristic', 0)

        with Timer() as timer:
            job_ids = list(self.jobs.keys())
            if not job_ids:
                stats = {
                    'algorithm': 'BalasDPC',
                    'execution_time': 0.0,
                    'nodes_explored': 0,
                    'pruning_count': 0,
                    'branch_count': 0,
                    'strong_branch_count': 0,
                    'weak_branch_count': 0,
                    'initial_bound': float('inf'),
                    'root_lower_bound': 0,
                    'optimal': True,
                    'exhaustive_search': True,
                    'heuristic_used': self.heuristic_class.__name__ if self.heuristic_class else None
                }
                return [], 0.0, stats

            # начальные r, q
            r = {j: self.jobs[j].r_i for j in job_ids}
            q = {j: self.jobs[j].q_i for j in job_ids}

            self.log("=" * 60)
            self.log("ЗАПУСК BalasDPC")
            self.log(f"Эвристика: {self.heuristic_class.__name__ if self.heuristic_class else 'None'}")
            self.log(f"Количество работ: {len(self.jobs)}")
            self.log("=" * 60)

            # Корневая нижняя граница h(K) на всём множестве
            self.root_lb = self._lower_bound(set(job_ids), r, q)
            self.log(f"Нижняя граница корня: {self.root_lb:.6f}")

            # Начальная эвристика (пользовательская)
            init_schedule, init_makespan = None, float('inf')
            if self.heuristic_class:
                try:
                    heuristic_jobs = [
                        Job(id=j, r_i=r[j], d_i=self.jobs[j].d_i, q_i=q[j])
                        for j in job_ids
                    ]
                    heuristic = self.heuristic_class(
                        heuristic_jobs,
                        self._get_precedence_dict_from_matrix(self.l_matrix)
                    )
                    init_schedule, init_makespan, _ = heuristic.solve()
                    self.log(f"Эвристика {self.heuristic_class.__name__}: C_max = {init_makespan}")
                except Exception as e:
                    self.log(f"Ошибка эвристики: {e}")
            if init_schedule:
                # Пересчёт makespan через базовый DPC-механизм
                self.best_makespan = init_makespan
                self.best_schedule = init_schedule.copy()
                self.log(f"Начальная верхняя граница: {self.best_makespan}")

            # Сброс счётчиков
            self.nodes_explored = 0
            self.pruning_count = 0
            self.branch_count = 0
            self.strong_branch_count = 0
            self.weak_branch_count = 0
            self.visited_states.clear()
            self.exhaustive_search = True
            self._extra_nodes_used = 0

            # Запуск ветвей и границ
            self._branch_and_bound(
                scheduled=[],
                unscheduled=set(job_ids),
                r=r,
                q=q,
                pi=copy.deepcopy(self.pi),
                sigma=copy.deepcopy(self.sigma),
                l_matrix=copy.deepcopy(self.l_matrix),
                upper_bound=self.best_makespan,
                timeout=timeout,
                start_time=timer.start,
                depth=0,
                max_depth=max_depth
            )
            self.log("=" * 60)
            self.log(f"Лучший C_max: {self.best_makespan:.6f}")
            self.log(f"Исследовано узлов: {self.nodes_explored}")
            self.log(f"Отсечений: {self.pruning_count}")
            self.log("=" * 60)

        is_optimal = False
        if self.exhaustive_search:
            is_optimal = True
        elif abs(self.best_makespan - self.root_lb) < 1e-9:
            is_optimal = True

        stats = {
            'algorithm': 'BalasDPC',
            'execution_time': timer.get_elapsed(),
            'nodes_explored': self.nodes_explored,
            'pruning_count': self.pruning_count,
            'branch_count': self.branch_count,
            'strong_branch_count': self.strong_branch_count,
            'weak_branch_count': self.weak_branch_count,
            'initial_bound': init_makespan if init_schedule else float('inf'),
            'root_lower_bound': self.root_lb,
            'optimal': is_optimal,
            'exhaustive_search': self.exhaustive_search,
            'heuristic_used': self.heuristic_class.__name__ if self.heuristic_class else None
        }

        self.log(f"END FILE, best={self.best_makespan}")
        self.log("=" * 80)

        try:
            self.log_file.close()
        except:
            pass

        if self.best_schedule is None:
            return [], 0.0, stats

        return self.best_schedule, self.best_makespan, stats

    def _can_apply_strong_branch(self,
                                 pi: Dict[int, Set[int]],
                                 J: List[int],
                                 c_job: int) -> bool:
        """
        True, если сильное ветвление реально добавляет новые дуги.
        Если все j∈J уже либо предшественники c_job, либо c_job уже
        предшественник всех j, то ветвление тривиально и приведёт к
        зацикливанию состояний.
        """
        # ветвь 1: c -> j для всех j ∈ J
        all_c_before_J = all(
            c_job in pi.get(j, set()) for j in J
        )
        # ветвь 2: j -> c для всех j ∈ J
        all_J_before_c = all(
            j in pi.get(c_job, set()) for j in J
        )

        # если хотя бы одна ветвь добавляет новую дугу — можно ветвиться
        return not (all_c_before_J or all_J_before_c)

    def _branch_and_bound(self,
                          scheduled: List[int],
                          unscheduled: Set[int],
                          r: Dict[int, float],
                          q: Dict[int, float],
                          pi: Dict[int, Set[int]],
                          sigma: Dict[int, Set[int]],
                          l_matrix: Dict[int, Dict[int, float]],
                          upper_bound: float,
                          timeout: float,
                          start_time: float,
                          depth: int,
                          max_depth: int) -> bool:

        if time.time() - start_time > timeout:
            self.exhaustive_search = False
            self.log("Таймаут", depth)
            return False
        if depth > max_depth:
            self.exhaustive_search = False
            self.log("Максимальная глубина", depth)
            return False

        # компактный ключ состояния
        l_items = tuple(sorted(
            (i, tuple(sorted(row.items())))
            for i, row in l_matrix.items()
            if row
        ))
        state_key = (tuple(scheduled),
                     tuple(sorted(unscheduled)),
                     tuple(sorted((k, tuple(sorted(v))) for k, v in pi.items())),
                     l_items)
        if state_key in self.visited_states:
            return False
        self.visited_states.add(state_key)

        self.nodes_explored += 1
        self.log(
            f"Узел depth={depth}; scheduled={scheduled}; unscheduled={sorted(unscheduled)}; best={self.best_makespan:.6f}",
            depth
        )

        # Лист: все запланированы
        if not unscheduled:
            # makespan через базовый DPC-механизм
            makespan, _ = self.calculate_makespan(scheduled)
            if makespan < self.best_makespan - 1e-9:
                self.best_makespan = makespan
                self.best_schedule = scheduled.copy()
                self.log(f"Новое лучшее решение (лист): C_max = {makespan:.6f}", depth)
                return True
            return False

        # Постпроцессинг (формулы (3)–(6) + грубая логика удаления дуг)
        r_upd, q_upd, pi_upd, sigma_upd, l_upd = self._postprocessing(
            unscheduled, r, q, pi, sigma, l_matrix
        )

        # Нижняя граница h(K)
        lb = self._lower_bound(unscheduled, r_upd, q_upd)
        self.log(f"LB={lb:.6f} для unscheduled={sorted(unscheduled)}", depth)
        if lb >= self.best_makespan - 1e-9:
            self.pruning_count += 1
            self.log(f"Отсечение по LB: {lb:.6f} >= best {self.best_makespan:.6f}", depth)
            return False

        # Эвристика на текущем подмножестве
        schedule_t = None
        if self.heuristic_class:
            try:
                heuristic_jobs = [
                    Job(id=j,
                        r_i=r_upd[j],
                        d_i=self.jobs[j].d_i,
                        q_i=q_upd[j]) for j in unscheduled
                ]
                heuristic = self.heuristic_class(
                    heuristic_jobs,
                    self._get_precedence_dict_from_matrix(l_upd, unscheduled)
                )
                schedule_t, makespan_t, _ = heuristic.solve()
                if schedule_t and set(schedule_t) == unscheduled:
                    full_schedule = scheduled + schedule_t
                    full_makespan, _ = self.calculate_makespan(full_schedule)
                    if full_makespan < self.best_makespan - 1e-9:
                        self.best_makespan = full_makespan
                        self.best_schedule = full_schedule.copy()
                        self.log(f"Эвристика улучшила UB: {self.best_makespan:.6f}", depth)
                    upper_bound = min(upper_bound, self.best_makespan)
                    if self.accept_heuristic_and_stop:
                        return True
            except Exception as e:
                self.log(f"Ошибка эвристики в узле: {e}", depth)

        # Если эвристика не дала порядок — fallback
        if not schedule_t or set(schedule_t) != unscheduled:
            schedule_t = sorted(list(unscheduled), key=lambda j: (r_upd[j], -q_upd[j]))

        # Критический путь по schedule_t (по графу G(t))
        critical_path, t_vals, C_max = self._find_critical_path(
            schedule_t, r_upd, q_upd, l_upd, pi_upd
        )
        self.log(f"Критический путь: {critical_path}", depth)

        # Essential arcs
        essential_arcs = self._find_essential_arcs(
            critical_path, t_vals, C_max, r_upd, q_upd, l_upd, pi_upd
        )

        # По Теореме 3.1: ищем c и множество J
        c_index, c_job, J = self._find_c_and_J(critical_path, t_vals, r_upd, essential_arcs)
        if c_job is not None and J:
            if self._check_strong_branch_conditions(
                    critical_path, c_index, J, t_vals, r_upd, essential_arcs
            ):
                if self._can_apply_strong_branch(pi_upd, J, c_job):
                    self.log(f"Сильное ветвление: c={c_job}, J={J}", depth)
                    self.strong_branch_count += 1
                    self.branch_count += 1
                    return self._strong_branching(
                        scheduled, unscheduled, r_upd, q_upd, pi_upd, sigma_upd, l_upd,
                        J, c_job, upper_bound, timeout, start_time, depth, max_depth
                    )

        # Reverse problem (раздел 4, шаг 4)
        rev_schedule_t, rev_t_vals, rev_C_max = self._reverse_problem_heuristic(
            unscheduled, r_upd, q_upd, l_upd, pi_upd, sigma_upd
        )
        rev_critical_path, _, _ = self._find_critical_path(
            rev_schedule_t, r_upd, q_upd, l_upd, pi_upd
        )
        rev_essential_arcs = self._find_essential_arcs(
            rev_critical_path, rev_t_vals, rev_C_max, r_upd, q_upd, l_upd, pi_upd
        )
        c_index_rev, c_job_rev, J_rev = self._find_c_and_J(
            rev_critical_path, rev_t_vals, r_upd, rev_essential_arcs
        )
        if c_job_rev is not None and J_rev:
            if self._check_strong_branch_conditions(
                    rev_critical_path, c_index_rev, J_rev, rev_t_vals, r_upd, rev_essential_arcs
            ):
                if self._can_apply_strong_branch(pi_upd, J, c_job):
                    self.log(f"Сильное ветвление (reverse): c={c_job_rev}, J={J_rev}", depth)
                    self.strong_branch_count += 1
                    self.branch_count += 1
                    return self._strong_branching(
                        scheduled, unscheduled, r_upd, q_upd, pi_upd, sigma_upd, l_upd,
                        J_rev, c_job_rev, upper_bound, timeout, start_time, depth, max_depth
                    )

        # Иначе — weak branching по паре (i, j) из критического пути
        i, j = self._select_branching_pair_weak(
            critical_path, essential_arcs, pi_upd, sigma_upd, l_upd, unscheduled
        )
        if i is None or j is None:
            return False

        self.log(f"Слабое ветвление по паре: {i} vs {j}", depth)
        self.weak_branch_count += 1
        self.branch_count += 1

        # Ветвь 1: i -> j
        pi1 = {k: set(v) for k, v in pi_upd.items()}
        sigma1 = {k: set(v) for k, v in sigma_upd.items()}
        pi1.setdefault(j, set()).add(i)
        sigma1.setdefault(i, set()).add(j)
        res1 = self._branch_and_bound(
            scheduled, unscheduled, r_upd, q_upd, pi1, sigma1, l_upd,
            upper_bound, timeout, start_time, depth + 1, max_depth
        )

        # Ветвь 2: j -> i
        pi2 = {k: set(v) for k, v in pi_upd.items()}
        sigma2 = {k: set(v) for k, v in sigma_upd.items()}
        pi2.setdefault(i, set()).add(j)
        sigma2.setdefault(j, set()).add(i)
        res2 = self._branch_and_bound(
            scheduled, unscheduled, r_upd, q_upd, pi2, sigma2, l_upd,
            upper_bound, timeout, start_time, depth + 1, max_depth
        )

        return res1 or res2

    # ---------------------------
    # Постпроцессинг
    # ---------------------------

    def _postprocessing(self,
                        unscheduled: Set[int],
                        r: Dict[int, float],
                        q: Dict[int, float],
                        pi: Dict[int, Set[int]],
                        sigma: Dict[int, Set[int]],
                        l_matrix: Dict[int, Dict[int, float]]):
        r_new = r.copy()
        q_new = q.copy()
        pi_new = {k: set(v) for k, v in pi.items()}
        sigma_new = {k: set(v) for k, v in sigma.items()}
        l_new = {i: dict(row) for i, row in l_matrix.items()}

        changed = True
        max_iter = max(1, len(unscheduled) * 6)
        iter_count = 0

        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1

            # (3) обновление r_j
            for i in list(unscheduled):
                for j in list(sigma_new.get(i, set())):
                    if j in unscheduled:
                        lij = l_new.get(i, {}).get(j, 0.0)
                        required_start = r_new.get(i, self.jobs[i].r_i) + self.jobs[i].d_i + lij
                        if required_start > r_new.get(j, self.jobs[j].r_i) + 1e-9:
                            r_new[j] = required_start
                            changed = True

            # (4) обновление q_i
            for j in list(unscheduled):
                for i in list(pi_new.get(j, set())):
                    if i in unscheduled:
                        lij = l_new.get(i, {}).get(j, 0.0)
                        required_tail = q_new.get(j, self.jobs[j].q_i) + self.jobs[j].d_i + lij
                        if required_tail > q_new.get(i, self.jobs[i].q_i) + 1e-9:
                            q_new[i] = required_tail
                            changed = True

            # (6) транзитивное замыкание L(i,j)
            new_edges = []
            for i in list(unscheduled):
                for k in list(sigma_new.get(i, set())):
                    if k in unscheduled:
                        for j in list(sigma_new.get(k, set())):
                            if j in unscheduled and j != i:
                                l_ik = l_new.get(i, {}).get(k, 0.0)
                                l_kj = l_new.get(k, {}).get(j, 0.0)
                                l_ij = l_ik + l_kj
                                if l_ij > l_new.get(i, {}).get(j, 0.0) + 1e-9:
                                    new_edges.append((i, j, l_ij))
            for i, j, val in new_edges:
                if i not in l_new:
                    l_new[i] = {}
                l_new[i][j] = val
                pi_new.setdefault(j, set()).add(i)
                sigma_new.setdefault(i, set()).add(j)
                changed = True

            # Грубое удаление дуг, которые не являются delayed (L(i,j) <= d_i)
            to_remove = []
            for i in list(unscheduled):
                for j, lij in list(l_new.get(i, {}).items()):
                    if lij <= self.jobs[i].d_i + 1e-9:
                        to_remove.append((i, j))
            if to_remove:
                for i, j in to_remove:
                    del l_new[i][j]
                    if j in pi_new:
                        pi_new[j].discard(i)
                    if i in sigma_new:
                        sigma_new[i].discard(j)
                changed = True

        return r_new, q_new, pi_new, sigma_new, l_new

    # ---------------------------
    # Нижняя граница h(K)
    # ---------------------------

    def _lower_bound(self,
                     unscheduled: Set[int],
                     r: Dict[int, float],
                     q: Dict[int, float]) -> float:
        if not unscheduled:
            return 0.0
        jobs_list = list(unscheduled)
        min_r = min(r[j] for j in jobs_list)
        total_p = sum(self.jobs[j].d_i for j in jobs_list)
        min_q = min(q[j] for j in jobs_list)
        return min_r + total_p + min_q

    # ---------------------------
    # Критический путь и essential arcs
    # ---------------------------

    def _find_critical_path(self,
                            schedule: List[int],
                            r: Dict[int, float],
                            q: Dict[int, float],
                            l_matrix: Dict[int, Dict[int, float]],
                            pi: Dict[int, Set[int]]) -> Tuple[List[int], Dict[int, float], float]:
        """
        Безопасное построение критического пути.
        """
        t: Dict[int, float] = {}

        # 1. Вычисляем t_i
        for i in schedule:
            preds = pi.get(i, set())
            valid_preds = [p for p in preds if p in t]

            if valid_preds:
                t[i] = max(
                    t[p] + self.jobs[p].d_i + l_matrix.get(p, {}).get(i, 0.0)
                    for p in valid_preds
                )
                t[i] = max(t[i], r[i])
            else:
                t[i] = r[i]

        # 2. Находим вершину, завершающую критический путь
        best_i = max(schedule, key=lambda i: t[i] + self.jobs[i].d_i + q[i])
        C_max = t[best_i] + self.jobs[best_i].d_i + q[best_i]

        # 3. Восстанавливаем путь назад
        path = [best_i]
        cur = best_i

        while True:
            preds = pi.get(cur, set())
            valid_preds = [p for p in preds if p in t]

            if not valid_preds:
                break

            found = None
            for p in valid_preds:
                val = t[p] + self.jobs[p].d_i + l_matrix.get(p, {}).get(cur, 0.0)
                if abs(val - t[cur]) < 1e-6:
                    found = p
                    break

            if found is None:
                break

            path.append(found)
            cur = found

        path.reverse()
        return path, t, C_max

    def _find_essential_arcs(self,
                             critical_path: List[int],
                             t_vals: Dict[int, float],
                             C_max: float,
                             r: Dict[int, float],
                             q: Dict[int, float],
                             l_matrix: Dict[int, Dict[int, float]],
                             pi: Dict[int, Set[int]]) -> Set[Tuple[int, int]]:
        essential = set()
        for j in critical_path:
            for i in pi.get(j, set()):
                lij = l_matrix.get(i, {}).get(j, 0.0)
                if lij > self.jobs[i].d_i + 1e-9:
                    if r[j] + 1e-9 < t_vals[j] and q[i] + 1e-9 < C_max - t_vals[i] - self.jobs[i].d_i:
                        essential.add((i, j))
        return essential

    def _find_c_and_J(self,
                      critical_path: List[int],
                      t_vals: Dict[int, float],
                      r: Dict[int, float],
                      essential_arcs: Set[Tuple[int, int]]) -> Tuple[int, Optional[int], List[int]]:
        if not critical_path:
            return -1, None, []

        c_index = -1
        for idx, i in enumerate(critical_path):
            if abs(r[i] - t_vals[i]) < 1e-6:
                c_index = idx

        if c_index == -1 or c_index == len(critical_path) - 1:
            return -1, None, []

        c_job = critical_path[c_index]
        J = critical_path[c_index + 1:]

        path_segment = set(critical_path[c_index:])
        for (i, j) in essential_arcs:
            if i in path_segment and j in path_segment:
                return -1, None, []

        return c_index, c_job, J

    def _check_strong_branch_conditions(self,
                                        critical_path: List[int],
                                        c_index: int,
                                        J: List[int],
                                        t_vals: Dict[int, float],
                                        r: Dict[int, float],
                                        essential_arcs: Set[Tuple[int, int]]) -> bool:
        if not J:
            return False
        c_job = critical_path[c_index]
        t_c = t_vals[c_job]

        for j in J:
            t_j = t_vals[j]
            if r[j] + 1e-9 < max(t_j, t_c):
                return False

        return True

    def _strong_branching(self,
                          scheduled: List[int],
                          unscheduled: Set[int],
                          r: Dict[int, float],
                          q: Dict[int, float],
                          pi: Dict[int, Set[int]],
                          sigma: Dict[int, Set[int]],
                          l_matrix: Dict[int, Dict[int, float]],
                          J: List[int],
                          c_job: int,
                          upper_bound: float,
                          timeout: float,
                          start_time: float,
                          depth: int,
                          max_depth: int) -> bool:
        # Ветвь 1: c -> j для всех j ∈ J
        pi1 = {k: set(v) for k, v in pi.items()}
        sigma1 = {k: set(v) for k, v in sigma.items()}
        for j in J:
            pi1.setdefault(j, set()).add(c_job)
            sigma1.setdefault(c_job, set()).add(j)

        res1 = self._branch_and_bound(
            scheduled, unscheduled, r, q, pi1, sigma1, l_matrix,
            upper_bound, timeout, start_time, depth + 1, max_depth
        )

        # Ветвь 2: j -> c для всех j ∈ J
        pi2 = {k: set(v) for k, v in pi.items()}
        sigma2 = {k: set(v) for k, v in sigma.items()}
        for j in J:
            pi2.setdefault(c_job, set()).add(j)
            sigma2.setdefault(j, set()).add(c_job)

        res2 = self._branch_and_bound(
            scheduled, unscheduled, r, q, pi2, sigma2, l_matrix,
            upper_bound, timeout, start_time, depth + 1, max_depth
        )

        return res1 or res2

    # ---------------------------
    # Reverse problem через эвристику
    # ---------------------------

    def _reverse_problem_heuristic(self,
                                   unscheduled: Set[int],
                                   r: Dict[int, float],
                                   q: Dict[int, float],
                                   l_matrix: Dict[int, Dict[int, float]],
                                   pi: Dict[int, Set[int]],
                                   sigma: Dict[int, Set[int]]) -> Tuple[List[int], Dict[int, float], float]:
        # строим обратные r', q', L', π', σ'
        r_rev: Dict[int, float] = {}
        q_rev: Dict[int, float] = {}
        L_rev: Dict[int, Dict[int, float]] = defaultdict(dict)
        pi_rev: Dict[int, Set[int]] = defaultdict(set)
        sigma_rev: Dict[int, Set[int]] = defaultdict(set)

        for i in unscheduled:
            r_rev[i] = q[i]
            q_rev[i] = r[i]

        for i in unscheduled:
            for j, lij in l_matrix.get(i, {}).items():
                if j in unscheduled:
                    val = lij - self.jobs[i].d_i + self.jobs[j].d_i
                    if val > 0:
                        L_rev[j][i] = val
                        pi_rev[i].add(j)
                        sigma_rev[j].add(i)

        # запускаем эвристику на обратной задаче
        schedule_rev = list(unscheduled)
        if self.heuristic_class:
            try:
                heuristic_jobs = [
                    Job(id=j,
                        r_i=r_rev[j],
                        d_i=self.jobs[j].d_i,
                        q_i=q_rev[j]) for j in unscheduled
                ]
                heuristic = self.heuristic_class(
                    heuristic_jobs,
                    self._get_precedence_dict_from_matrix(L_rev, unscheduled)
                )
                schedule_rev, _, _ = heuristic.solve()
            except Exception:
                schedule_rev = list(unscheduled)

        # считаем t'_i и C'_max в обратной задаче
        t_rev: Dict[int, float] = {}
        for i in schedule_rev:
            preds = pi_rev.get(i, set())
            valid_preds = [p for p in preds if p in t_rev]
            if valid_preds:
                t_rev[i] = max(
                    t_rev[p] + self.jobs[p].d_i + L_rev.get(p, {}).get(i, 0.0)
                    for p in valid_preds
                )
                t_rev[i] = max(t_rev[i], r_rev[i])
            else:
                t_rev[i] = r_rev[i]

        if schedule_rev:
            best_i = max(schedule_rev, key=lambda i: t_rev[i] + self.jobs[i].d_i + q_rev[i])
            C_max_rev = t_rev[best_i] + self.jobs[best_i].d_i + q_rev[best_i]
        else:
            C_max_rev = 0.0

        return schedule_rev, t_rev, C_max_rev

    # ---------------------------
    # Weak branching pair
    # ---------------------------

    def _select_branching_pair_weak(self, critical_path, essential_arcs, pi, sigma, l_matrix, unscheduled):
        best_pair = None
        best_score = -float('inf')

        for idx in range(len(critical_path) - 1):
            i = critical_path[idx]
            j = critical_path[idx + 1]

            if (i, j) in essential_arcs or (j, i) in essential_arcs:
                continue

            # score: насколько сильно порядок i/j влияет на makespan
            score = (
                    self.jobs[i].q_i + self.jobs[j].q_i +
                    l_matrix.get(i, {}).get(j, 0.0) +
                    l_matrix.get(j, {}).get(i, 0.0)
            )

            if score > best_score:
                best_score = score
                best_pair = (i, j)

        return best_pair if best_pair else (None, None)
