from typing import List, Dict, Tuple, Optional
from core.utils import Timer
from .lth import LTH


class MLTH(LTH):
    """
    Modified Longest Tail Heuristic (MLTH)
    Строгая реализация Algorithm 3 и Algorithm 4
    из статьи Zhang–Sauppe–Jacobson (2020).
    """

    def solve(self, **kwargs) -> Tuple[Optional[List[int]], float, Dict]:
        with Timer() as timer:
            if self.n == 0:
                return [], 0.0, {
                    "algorithm": "MLTH",
                    "execution_time": 0.0,
                    "iterations": 0,
                    "C_max": 0.0,
                    "schedule_length": 0,
                }

            schedule, C_max = self._mlth()

        return schedule, C_max, {
            "algorithm": "MLTH",
            "execution_time": timer.get_elapsed(),
            "iterations": self.n,
            "C_max": C_max,
            "schedule_length": len(schedule),
        }

    # ----------------------------------------------------------
    # Algorithm 3: начальное расписание MLTH
    # ----------------------------------------------------------
    def _mlth(self) -> Tuple[List[int], float]:
        t = 0.0
        S: List[int] = []

        # r_prime — текущие "головы" (release times) с учетом DPC
        r_prime: Dict[int, float] = {j: job.r_i for j, job in self.jobs.items()}
        pred = self._get_predecessors()
        succ = self._get_successors()

        # Основной цикл построения начального расписания
        while len(S) < self.n:
            # множество доступных работ: все предшественники уже в S
            Q = [j for j in self.jobs if j not in S and pred[j].issubset(S)]
            if not Q:
                break

            released = [j for j in Q if r_prime[j] <= t]
            unreleased = [j for j in Q if r_prime[j] > t]

            # выбор k среди уже "выпущенных" работ
            k = None
            q_k = float("-inf")
            if released:
                k = max(released, key=lambda j: self.jobs[j].q_i)
                q_k = self.jobs[k].q_i

            # выбор l среди еще не "выпущенных" работ
            l = None
            q_l = float("-inf")
            if unreleased:
                min_r = min(r_prime[j] for j in unreleased)
                cand = [j for j in unreleased if r_prime[j] == min_r]
                l = max(cand, key=lambda j: self.jobs[j].q_i)
                q_l = self.jobs[l].q_i

            # Ключевое правило MLTH (Algorithm 3, line 7-8):
            # Если у ближайшей невыпущенной работы хвост больше,
            # чем у лучшей выпущенной, то перескакиваем временем к ней
            # и ПЕРЕХОДИМ К СЛЕДУЮЩЕЙ ИТЕРАЦИИ, не планируя k.
            if l is not None and q_l > q_k:
                t = r_prime[l]
                continue

            # Если нет ни одной выпущенной работы — двигаем время до ближайшего r_prime
            if k is None:
                t = min(r_prime[j] for j in Q)
                continue

            # Если мы здесь, значит k существует и является лучшим выбором.
            # Планируем работу k
            s_k = max(t, r_prime[k])
            S.append(k)

            # Обновляем головы преемников по DPC
            for j in succ[k]:
                lij = self.l_matrix.get(k, {}).get(j, 0.0)
                if lij > 0:
                    need = s_k + lij
                    if r_prime[j] < need:
                        r_prime[j] = need

            t = s_k + self.jobs[k].d_i

        # Вычисляем расписание и C_max для начального расписания
        C, start_times, r_prime = self._compute_schedule(S)

        # Процедура пересоставления (Algorithm 4)
        S_rescheduled, C_rescheduled = self._reschedule(S)

        return S_rescheduled, C_rescheduled

    # ----------------------------------------------------------
    # Вычисление расписания (start_times, r_prime, C_max)
    # ----------------------------------------------------------
    def _compute_schedule(
            self, S: List[int]
    ) -> Tuple[float, Dict[int, float], Dict[int, float]]:
        start: Dict[int, float] = {}
        r_prime: Dict[int, float] = {j: self.jobs[j].r_i for j in self.jobs}
        pred = self._get_predecessors()

        current = 0.0
        for j in S:
            job = self.jobs[j]

            # Обновляем r_prime[j] по всем предшественникам
            for i in pred[j]:
                if i in start:
                    lij = self.l_matrix.get(i, {}).get(j, 0.0)
                    if lij > 0:
                        need = start[i] + lij
                        if r_prime[j] < need:
                            r_prime[j] = need

            s_j = max(current, r_prime[j])
            start[j] = s_j
            current = s_j + job.d_i

        if not S:
            return 0.0, {}, r_prime

        C = max(start[j] + self.jobs[j].d_i + self.jobs[j].q_i for j in S)
        return C, start, r_prime

    # ----------------------------------------------------------
    # Построение DAG G(S) строго по статье (для Algorithm 4)
    # ----------------------------------------------------------
    def _build_dag(self, S: List[int]) -> Tuple[Dict[int, List[Tuple[int, float]]], int]:
        edges: Dict[int, List[Tuple[int, float]]] = {}

        # Фиктивная вершина-сток t. Используем -1, так как ID работ начинаются с 0.
        t = -1
        edges[0] = []  # 0 - фиктивная вершина-исток
        for j in S:
            edges[j] = []
        edges[t] = []

        # Дуги 0 -> j с весом r_j
        for j in S:
            edges[0].append((j, self.jobs[j].r_i))

        # Дуги j -> k только если j предшествует k в расписании S
        pos = {job_id: idx for idx, job_id in enumerate(S)}
        for j in S:
            for k in S:
                if pos[j] < pos[k]:
                    lij = self.l_matrix.get(j, {}).get(k, 0.0)
                    edges[j].append((k, max(lij, self.jobs[j].d_i)))  # Вес дуги: lij или d_j

        # Дуги j -> t с весом d_j + q_j
        for j in S:
            w = self.jobs[j].d_i + self.jobs[j].q_i
            edges[j].append((t, w))

        return edges, t

    # ----------------------------------------------------------
    # Поиск максимального пути в DAG (0 -> t)
    # ----------------------------------------------------------
    def _longest_path(self, edges: Dict[int, List[Tuple[int, float]]], t: int) -> List[int]:
        # Топологическая сортировка: 0, затем работы в порядке S, затем t
        vertices = list(edges.keys())
        # Ключ для сортировки: 0 (False) -> работы (True) -> t (True)
        # но t должен быть после всех работ. Поэтому:
        vertices.sort(key=lambda v: (v == t, v == 0))  # t в конец, 0 в начало

        dist: Dict[int, float] = {v: float("-inf") for v in vertices}
        parent: Dict[int, Optional[int]] = {v: None for v in vertices}
        dist[0] = 0.0

        for u in vertices:
            if dist[u] == float("-inf"):
                continue
            for v, w in edges[u]:
                if dist[v] < dist[u] + w:
                    dist[v] = dist[u] + w
                    parent[v] = u

        # Восстановление пути 0 -> t
        path: List[int] = []
        cur: Optional[int] = t
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            cur = parent[cur]
        path.reverse()

        # Возвращаем только работы (положительные индексы, исключая 0 и t = -1)
        return [x for x in path if x > 0]

    # ----------------------------------------------------------
    # Algorithm 4: процедура пересоставления (Rescheduling)
    # ----------------------------------------------------------
    def _reschedule(
            self,
            S: List[int],
    ) -> Tuple[List[int], float]:
        max_iter = self.n ** 2  # Гарантированное число итераций по Теореме 2
        it = 0

        C, start_times, r_prime = self._compute_schedule(S)

        while it < max_iter:
            it += 1

            edges, t_node = self._build_dag(S)
            critical = self._longest_path(edges, t_node)

            if not critical:
                # Нет работ в критическом пути (маловероятно, но для безопасности)
                break

                # Ищем "отложенную" работу (delayed job) на критическом пути
            # Условие: r_prime[j] > start_times[j]
            delayed = None
            for j in critical:
                if r_prime.get(j, 0) > start_times.get(j, 0) + 1e-9:
                    delayed = j
                    break

            # Если отложенных работ нет — расписание валидно для ветвления
            if delayed is None:
                break

            i1 = critical[0]  # Первая работа на критическом пути

            # Ищем работу j1, которая непосредственно предшествует i1 в расписании S
            idx_i1 = S.index(i1)
            if idx_i1 == 0:
                # Если i1 — первая работа, то вставляем delayed в начало
                j1 = None
            else:
                j1 = S[idx_i1 - 1]

            # Удаляем delayed из текущей позиции
            S.remove(delayed)

            # Вставляем delayed сразу после j1 (т.е. перед i1)
            if j1 is None:
                S.insert(0, delayed)
            else:
                # После удаления delayed индекс i1 мог измениться, поэтому ищем его заново
                new_idx_i1 = S.index(i1)
                S.insert(new_idx_i1, delayed)

            # Пересчитываем расписание и C_max
            C, start_times, r_prime = self._compute_schedule(S)

        return S, C