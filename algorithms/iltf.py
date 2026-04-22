from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from core.Algorithm import Algorithm
from core.job import Job
from core.utils import Timer


class ILTF(Algorithm):
    """
    ILTF (Idle Largest Tail First) для задачи 1|r_i, q_i, dpc|C_max.

    - Один процессор, без прерываний.
    - DPC: T(j) >= T(i) + d_i + l_ij.
    - Предварительно пересчитываем r_i, q_i по критическим путям (как в статье).
    - Строим нижнюю границу LB.
    - В основном цикле реализуем логику ILTF:
      * выбираем по Джексону (максимальный q среди готовых),
      * проверяем «важную» будущую задачу u*,
      * при необходимости ждём её и/или заполняем простой другой задачей.
    """

    def __init__(
        self,
        jobs: List[Job],
        precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None
    ):
        super().__init__(jobs, precedence_constraints)
        self.name = "ILTF"

        # Пустой случай
        if not self.jobs:
            self.successors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            self.predecessors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            self.start_jobs: List[int] = []
            self.end_jobs: List[int] = []
            self.path_from_start: Dict[int, float] = {}
            self.path_to_end: Dict[int, float] = {}
            self.LB: float = 0.0
            return

        # Построение графа по l_matrix (delayed precedence)
        self._build_graph()
        # Критические пути и обновление r_i, q_i
        self._compute_paths_and_update_rq()
        # Нижняя граница LB
        self._compute_lower_bound()

    # ------------------------------------------------------------------
    # Построение графа
    # ------------------------------------------------------------------

    def _build_graph(self):
        self.successors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.predecessors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

        # self.l_matrix уже построен в базовом Algorithm из precedence_constraints
        for i, row in self.l_matrix.items():
            for j, lij in row.items():
                if lij > 0:
                    self.successors[i].append((j, lij))
                    self.predecessors[j].append((i, lij))

        job_ids = list(self.jobs.keys())
        self.start_jobs = [j for j in job_ids if not self.predecessors[j]]
        self.end_jobs = [j for j in job_ids if not self.successors[j]]

    # ------------------------------------------------------------------
    # Критические пути и обновление r_i, q_i
    # ------------------------------------------------------------------

    def _compute_paths_and_update_rq(self):
        job_ids = list(self.jobs.keys())

        # Максимальные пути от "начала" (s) до каждой вершины (для r*)
        dist_from_start = {j: float("-inf") for j in job_ids}
        # Для стартовых вершин длина пути = r_j
        for j in self.start_jobs:
            dist_from_start[j] = self.jobs[j].r_i

        changed = True
        while changed:
            changed = False
            for i in job_ids:
                if dist_from_start[i] == float("-inf"):
                    continue
                for j, lij in self.successors[i]:
                    cand = dist_from_start[i] + lij
                    if cand > dist_from_start[j] + 1e-9:
                        dist_from_start[j] = cand
                        changed = True

        # Максимальные пути от каждой вершины до "конца" (t) (для q*)
        dist_to_end = {j: float("-inf") for j in job_ids}
        for j in self.end_jobs:
            dist_to_end[j] = self.jobs[j].q_i

        changed = True
        while changed:
            changed = False
            for j in job_ids:
                if dist_to_end[j] == float("-inf"):
                    continue
                for i, lij in self.predecessors[j]:
                    cand = dist_to_end[j] + lij
                    if cand > dist_to_end[i] + 1e-9:
                        dist_to_end[i] = cand
                        changed = True

        self.path_from_start = {j: max(0.0, d) if d > float("-inf") else 0.0
                                for j, d in dist_from_start.items()}
        self.path_to_end = {j: max(0.0, d) if d > float("-inf") else 0.0
                            for j, d in dist_to_end.items()}

        # Обновляем r_i, q_i
        for j in job_ids:
            job = self.jobs[j]
            job.r_i = max(job.r_i, self.path_from_start[j])
            job.q_i = max(job.q_i, self.path_to_end[j])

    # ------------------------------------------------------------------
    # Нижняя граница LB
    # ------------------------------------------------------------------

    def _compute_lower_bound(self):
        job_ids = list(self.jobs.keys())
        if not job_ids:
            self.LB = 0.0
            return

        # Критический путь tcp в расширенном графе (s,t)
        # Приближённо: максимальное r*_j + (сумма d_i по пути) + q*_j.
        # В коде используем упрощённую оценку: max_j (path_from_start[j] + self.jobs[j].q_i)
        tcp = 0.0
        for j in job_ids:
            val = self.path_from_start[j] + self.jobs[j].q_i
            if val > tcp:
                tcp = val

        r_min = min(job.r_i for job in self.jobs.values())
        q_min = min(job.q_i for job in self.jobs.values())
        total_p = sum(job.d_i for job in self.jobs.values())

        self.LB = max(tcp, r_min + total_p + q_min)

    # ------------------------------------------------------------------
    # Вспомогательные функции ILTF
    # ------------------------------------------------------------------

    def _all_preds_done(self, job_id: int, scheduled: Set[int]) -> bool:
        for pred, _ in self.predecessors[job_id]:
            if pred not in scheduled:
                return False
        return True

    def _get_ready_jobs(self, scheduled: Set[int], current_time: float) -> List[int]:
        ready = []
        for j, job in self.jobs.items():
            if j in scheduled:
                continue
            if not self._all_preds_done(j, scheduled):
                continue
            if job.r_i <= current_time + 1e-9:
                ready.append(j)
        return ready

    def _find_future_important_job(
        self,
        scheduled: Set[int],
        current_time: float,
        horizon: float
    ) -> Optional[int]:
        """
        Ищет u*:
        - станет готовой в (current_time, horizon],
        - q(u*) > LB/2,
        - q(u*) - q(u) > r(u*) - current_time (проверка делается снаружи).
        Здесь возвращаем кандидата по максимальному q среди подходящих по времени.
        """
        best_job = None
        best_q = float("-inf")

        for j, job in self.jobs.items():
            if j in scheduled:
                continue
            if not self._all_preds_done(j, scheduled):
                continue

            ready_time = max(job.r_i, current_time)
            if current_time < ready_time <= horizon + 1e-9:
                if job.q_i > best_q:
                    best_q = job.q_i
                    best_job = j

        return best_job

    def _find_job_for_idle(
        self,
        scheduled: Set[int],
        current_time: float,
        until_time: float
    ) -> Optional[int]:
        """
        Ищет задание u1, которое можно полностью выполнить в [current_time, until_time],
        с максимальным q.
        """
        best_job = None
        best_q = float("-inf")

        for j, job in self.jobs.items():
            if j in scheduled:
                continue
            if not self._all_preds_done(j, scheduled):
                continue

            start = max(current_time, job.r_i)
            end = start + job.d_i
            if end <= until_time + 1e-9 and job.q_i > best_q:
                best_q = job.q_i
                best_job = j

        return best_job

    # ------------------------------------------------------------------
    # Основной алгоритм ILTF
    # ------------------------------------------------------------------

    def solve(self, **kwargs):
        """
        Возвращает (schedule, C_max, stats).

        schedule — порядок работ (перестановка id),
        C_max — makespan по DPC-модели (через calculate_makespan),
        stats — словарь с базовой статистикой.
        """
        with Timer() as timer:
            if self.n == 0:
                stats = {
                    'algorithm': self.name,
                    'execution_time': 0.0,
                    'iterations': 0,
                    'idle_times': 0,
                    'LB': 0.0,
                    'gap': 0.0,
                    'schedule_length': 0
                }
                return [], 0.0, stats

            schedule: List[int] = []
            scheduled: Set[int] = set()

            current_time = min(job.r_i for job in self.jobs.values())

            iterations = 0
            idle_events = 0

            while len(scheduled) < self.n:
                iterations += 1

                ready = self._get_ready_jobs(scheduled, current_time)

                if not ready:
                    # Нет готовых — переходим к ближайшему времени готовности
                    next_time = float("inf")
                    for j, job in self.jobs.items():
                        if j in scheduled:
                            continue
                        if not self._all_preds_done(j, scheduled):
                            continue
                        if job.r_i > current_time + 1e-9:
                            next_time = min(next_time, job.r_i)

                    if next_time < float("inf"):
                        idle_events += 1
                        current_time = next_time
                        continue
                    else:
                        # Теоретически не должно случиться, но на всякий случай
                        break

                # Шаг 1: Джексон — выбираем u с максимальным q среди готовых
                u = max(ready, key=lambda j: self.jobs[j].q_i)
                u_job = self.jobs[u]
                horizon = current_time + u_job.d_i

                # Шаг 2: ищем кандидата u* в (current_time, horizon]
                u_star = self._find_future_important_job(scheduled, current_time, horizon)

                selected: int

                if u_star is not None:
                    u_star_job = self.jobs[u_star]
                    ready_time_star = max(u_star_job.r_i, current_time)

                    # Проверяем условие важности:
                    # q(u*) > LB/2 и q(u*) - q(u) > r(u*) - current_time
                    if (
                        u_star_job.q_i > self.LB / 2.0 + 1e-9 and
                        u_star_job.q_i - u_job.q_i > ready_time_star - current_time + 1e-9
                    ):
                        # Шаг 3: ищем u1 для заполнения простоя до r(u*)
                        u1 = self._find_job_for_idle(
                            scheduled,
                            current_time,
                            ready_time_star
                        )
                        if u1 is not None:
                            selected = u1
                        else:
                            # Ждём u*, затем ставим её
                            if ready_time_star > current_time + 1e-9:
                                idle_events += 1
                                current_time = ready_time_star
                            selected = u_star
                    else:
                        # Условие важности не выполнено — ставим u
                        selected = u
                else:
                    # Нет подходящей будущей важной задачи — ставим u
                    selected = u

                # Выполняем выбранную задачу
                job = self.jobs[selected]
                start_time = max(current_time, job.r_i)
                current_time = start_time + job.d_i

                schedule.append(selected)
                scheduled.add(selected)

            # Один процессор, makespan считаем через базовый DPC-механизм
            C_max, _ = self.calculate_makespan(schedule)

        gap = ((C_max - self.LB) / self.LB) * 100.0 if self.LB > 1e-9 else 0.0

        stats = {
            'algorithm': self.name,
            'execution_time': timer.get_elapsed(),
            'iterations': iterations,
            'idle_times': idle_events,
            'LB': self.LB,
            'gap': gap,
            'schedule_length': len(schedule),
        }

        return schedule, C_max, stats
