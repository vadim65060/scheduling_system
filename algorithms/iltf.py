from typing import List, Dict, Tuple, Optional, Set

from core.Algorithm import Algorithm
from core.job import Job
from core.utils import Timer


class ILTF(Algorithm):
    """
    ILTF (Idle Largest Tail First) для задачи 1|r_i, q_i, dpc|C_max.

    Алгоритм из статьи CSIT:
    - Пересчитывает r_i, q_i по критическим путям с учётом всех DPC
    - Строит нижнюю границу LB
    - В основном цикле: выбирает работу с max q среди готовых,
      проверяет "важную" будущую работу u* (q > LB/2, выигрыш > ожидание),
      при необходимости ждёт или заполняет простой другой работой
    """

    def __init__(
            self,
            jobs: List[Job],
            precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None
    ):
        super().__init__(jobs, precedence_constraints)
        self.name = "ILTF"

        if not self.jobs:
            self.predecessors: Dict[int, set] = {}
            self.successors: Dict[int, set] = {}
            self.start_jobs: List[int] = []
            self.end_jobs: List[int] = []
            self.r_star: Dict[int, float] = {}
            self.q_star: Dict[int, float] = {}
            self.LB: float = 0.0
            return

        # Используем готовые методы из Scheduler для построения графа
        self.predecessors = self._get_predecessors()
        self.successors = self._get_successors()

        job_ids = list(self.jobs.keys())  # self.jobs - это dict, .keys() корректен
        self.start_jobs = [j for j in job_ids if not self.predecessors.get(j)]
        self.end_jobs = [j for j in job_ids if not self.successors.get(j)]

        # Критические пути и обновление r_i, q_i (без модификации исходных job!)
        self._compute_paths_and_update_rq()

        # Нижняя граница LB
        self._compute_lower_bound()

    def _compute_paths_and_update_rq(self):
        """
        Вычисляет максимальные пути от начала до каждой вершины (r*)
        и от каждой вершины до конца (q*) с учётом всех DPC.
        Результаты сохраняет в self.r_star и self.q_star (не модифицирует job!).
        """
        job_ids = list(self.jobs.keys())

        # Максимальные пути от "начала" до каждой вершины (для r*)
        dist_from_start = {j: float("-inf") for j in job_ids}
        for j in self.start_jobs:
            dist_from_start[j] = self.jobs[j].r_i

        changed = True
        while changed:
            changed = False
            for i in job_ids:
                if dist_from_start[i] == float("-inf"):
                    continue
                for j in self.successors.get(i, set()):
                    lij = self.l_matrix[i][j]
                    cand = dist_from_start[i] + lij
                    if cand > dist_from_start[j] + 1e-9:
                        dist_from_start[j] = cand
                        changed = True

        # Максимальные пути от каждой вершины до "конца" (для q*)
        dist_to_end = {j: float("-inf") for j in job_ids}
        for j in self.end_jobs:
            dist_to_end[j] = self.jobs[j].q_i

        changed = True
        while changed:
            changed = False
            for j in job_ids:
                if dist_to_end[j] == float("-inf"):
                    continue
                for i in self.predecessors.get(j, set()):
                    lij = self.l_matrix[i][j]
                    cand = dist_to_end[j] + lij
                    if cand > dist_to_end[i] + 1e-9:
                        dist_to_end[i] = cand
                        changed = True

        # Сохраняем обновлённые значения в отдельных полях
        self.r_star = {}
        self.q_star = {}
        for j in job_ids:
            self.r_star[j] = max(self.jobs[j].r_i,
                                 0.0 if dist_from_start[j] == float("-inf") else dist_from_start[j])
            self.q_star[j] = max(self.jobs[j].q_i,
                                 0.0 if dist_to_end[j] == float("-inf") else dist_to_end[j])

    def _compute_lower_bound(self):
        """Вычисляет нижнюю границу LB согласно статье."""
        job_ids = list(self.jobs.keys())
        if not job_ids:
            self.LB = 0.0
            return

        # Критический путь tcp
        tcp = 0.0
        for j in job_ids:
            val = self.r_star[j] + self.q_star[j]
            if val > tcp:
                tcp = val

        r_min = min(self.r_star.values())
        q_min = min(self.q_star.values())
        total_p = sum(job.d_i for job in self.jobs.values())

        self.LB = max(tcp, r_min + total_p + q_min)

    def _all_preds_done(self, job_id: int, scheduled: Set[int]) -> bool:
        """Проверяет, что все предшественники (DPC) работы уже запланированы."""
        for pred in self.predecessors.get(job_id, set()):
            if pred not in scheduled:
                return False
        return True

    def _get_ready_jobs(self, scheduled: Set[int], current_time: float) -> List[int]:
        """Возвращает список готовых к выполнению работ."""
        ready = []
        for j in self.jobs:
            if j in scheduled:
                continue
            if not self._all_preds_done(j, scheduled):
                continue
            # Используем r_star вместо r_i
            if self.r_star[j] <= current_time + 1e-9:
                ready.append(j)
        return ready

    def _find_future_important_job(
            self,
            scheduled: Set[int],
            current_time: float,
            horizon: float
    ) -> Optional[int]:
        """
        Ищет u*: работу, которая станет доступной в (current_time, horizon]
        и имеет максимальный q*.
        """
        best_job = None
        best_q = float("-inf")

        for j in self.jobs:
            if j in scheduled:
                continue
            if not self._all_preds_done(j, scheduled):
                continue

            ready_time = max(self.r_star[j], current_time)
            if current_time < ready_time <= horizon + 1e-9:
                if self.q_star[j] > best_q:
                    best_q = self.q_star[j]
                    best_job = j

        return best_job

    def _find_job_for_idle(
            self,
            scheduled: Set[int],
            current_time: float,
            until_time: float
    ) -> Optional[int]:
        """
        Ищет работу u1, которую можно полностью выполнить
        в интервале [current_time, until_time].
        """
        best_job = None
        best_q = float("-inf")

        for j in self.jobs:
            if j in scheduled:
                continue
            if not self._all_preds_done(j, scheduled):
                continue

            start = max(current_time, self.r_star[j])
            end = start + self.jobs[j].d_i
            if end <= until_time + 1e-9 and self.q_star[j] > best_q:
                best_q = self.q_star[j]
                best_job = j

        return best_job

    def solve(self, **kwargs):
        """
        Запускает алгоритм ILTF.
        Возвращает (schedule, C_max, stats).
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

            # Начальное время = минимальное r*
            current_time = min(self.r_star.values())

            iterations = 0
            idle_events = 0

            # Локальные копии r* и q* для обновления в процессе
            r_local = self.r_star.copy()
            q_local = self.q_star.copy()

            while len(scheduled) < self.n:
                iterations += 1

                ready = self._get_ready_jobs(scheduled, current_time)

                if not ready:
                    # Переходим к ближайшему времени готовности
                    next_time = float("inf")
                    for j in self.jobs:
                        if j in scheduled:
                            continue
                        if not self._all_preds_done(j, scheduled):
                            continue
                        if r_local[j] > current_time + 1e-9:
                            next_time = min(next_time, r_local[j])

                    if next_time < float("inf"):
                        idle_events += 1
                        current_time = next_time
                        continue
                    else:
                        break

                # Шаг 1: выбираем u с максимальным q* среди готовых
                u = max(ready, key=lambda j: q_local[j])
                u_job = self.jobs[u]
                horizon = current_time + u_job.d_i

                # Шаг 2: ищем кандидата u* в (current_time, horizon]
                u_star = self._find_future_important_job(scheduled, current_time, horizon)

                if u_star is not None:
                    ready_time_star = max(r_local[u_star], current_time)

                    # Проверяем условие важности из статьи CSIT:
                    # q(u*) > LB/2 и q(u*) - q(u) > r(u*) - time
                    if (q_local[u_star] > self.LB / 2.0 + 1e-9 and
                            q_local[u_star] - q_local[u] > ready_time_star - current_time + 1e-9):

                        # Шаг 3: ищем u1 для заполнения простоя
                        u1 = self._find_job_for_idle(scheduled, current_time, ready_time_star)

                        if u1 is not None:
                            selected = u1
                        else:
                            # Ждём u*
                            if ready_time_star > current_time + 1e-9:
                                idle_events += 1
                                current_time = ready_time_star
                            selected = u_star
                    else:
                        selected = u
                else:
                    selected = u

                # Выполняем выбранную работу
                job = self.jobs[selected]
                start_time = max(current_time, r_local[selected])
                current_time = start_time + job.d_i

                schedule.append(selected)
                scheduled.add(selected)

                # Обновляем r* последователей (как в статье)
                for succ in self.successors.get(selected, set()):
                    if succ not in scheduled:
                        required_start = start_time + self.l_matrix[selected][succ]
                        if required_start > r_local[succ]:
                            r_local[succ] = required_start

            # Вычисляем makespan через базовый механизм
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