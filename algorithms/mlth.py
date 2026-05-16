"""
Modified Longest Tail Heuristic (MLTH)
Строгая реализация Algorithm 3 и Algorithm 4
из статьи Zhang–Sauppe–Jacobson (2020).
"""

from typing import List, Dict, Tuple, Optional, Set

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
        """
        Algorithm 3 из статьи.
        Построение начального расписания с возможностью пропуска времени
        к ещё не выпущенной работе, если у неё больший хвост.
        """
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

            released = [j for j in Q if r_prime[j] <= t + 1e-9]
            unreleased = [j for j in Q if r_prime[j] > t + 1e-9]

            # выбор k среди уже "выпущенных" работ (с максимальным q_i)
            k = None
            q_k = float("-inf")
            if released:
                k = max(released, key=lambda j: self.jobs[j].q_i)
                q_k = self.jobs[k].q_i

            # выбор l среди еще не "выпущенных" работ
            # l — работа с минимальным r'_i, среди них выбираем с максимальным q_i
            l = None
            q_l = float("-inf")
            if unreleased:
                min_r = min(r_prime[j] for j in unreleased)
                cand = [j for j in unreleased if abs(r_prime[j] - min_r) < 1e-9]
                l = max(cand, key=lambda j: self.jobs[j].q_i)
                q_l = self.jobs[l].q_i

            # Ключевое правило MLTH (Algorithm 3, line 7-8):
            # Если у ближайшей невыпущенной работы хвост больше,
            # чем у лучшей выпущенной, то перескакиваем временем к ней
            # и НЕ планируем работу в этой итерации
            if l is not None and q_l > q_k + 1e-9:
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
                # Используем l_matrix как в LTH
                lij = self.l_matrix[k].get(j, self.jobs[k].d_i)
                if lij > 0:
                    need = s_k + lij
                    if r_prime[j] < need - 1e-9:
                        r_prime[j] = need

            t = s_k + self.jobs[k].d_i

        # Вычисляем makespan и определяем отложенные работы
        C_max, start_times = self.calculate_makespan(S)
        delayed_jobs = self._get_delayed_jobs(S, start_times, r_prime)

        # Процедура пересоставления (Algorithm 4) — только если есть отложенные работы
        if delayed_jobs:
            S_rescheduled, C_rescheduled = self._reschedule(S)
            return S_rescheduled, C_rescheduled

        return S, C_max

    # ----------------------------------------------------------
    # Вспомогательный метод для определения отложенных работ
    # ----------------------------------------------------------
    def _get_delayed_jobs(
            self,
            S: List[int],
            start_times: Dict[int, float],
            r_prime: Dict[int, float]
    ) -> Set[int]:
        """
        Определяет отложенные работы (delayed jobs) в расписании S.
        Работа считается отложенной, если r'_j > s_j (с epsilon).
        """
        delayed = set()
        for j in S:
            # r_prime[j] — это текущее значение головы с учётом DPC
            # start_times[j] — фактическое время начала
            if r_prime.get(j, 0) > start_times.get(j, 0) + 1e-9:
                delayed.add(j)
        return delayed

    # ----------------------------------------------------------
    # Построение DAG G(S) строго по статье (для Algorithm 4)
    # ----------------------------------------------------------
    def _build_dag(self, S: List[int]) -> Tuple[Dict[int, List[Tuple[int, float]]], int]:
        """
        Построение графа G(S) согласно статье (стр. 3, Graph Representation):

        Vertex set: N = J ∪ {o} ∪ {t}, где o = 0, t = -1

        Edge set:
        - e_{oj} с весом r_j для всех j ∈ J
        - e_{ij} с весом l_ij для всех i, j ∈ J, где i предшествует j в S
          (согласно статье, если нет DPC, то l_ij = d_i по умолчанию)
        - e_{jt} с весом d_j + q_j для всех j ∈ J
        """
        t = -1  # фиктивная вершина-сток
        edges: Dict[int, List[Tuple[int, float]]] = {}

        # Инициализация списков смежности
        edges[0] = []  # исток
        for j in S:
            edges[j] = []
        edges[t] = []  # сток

        # Дуги o -> j с весом r_j
        for j in S:
            edges[0].append((j, self.jobs[j].r_i))

        # Дуги j -> k только если j предшествует k в расписании S
        pos = {job_id: idx for idx, job_id in enumerate(S)}
        for j in S:
            for k in S:
                if pos[j] < pos[k]:
                    # Согласно статье: вес = l_jk (если есть DPC) или d_j (по умолчанию)
                    # В self.l_matrix уже хранятся правильные значения (>= d_j)
                    lij = self.l_matrix[j].get(k, self.jobs[j].d_i)
                    edges[j].append((k, lij))

        # Дуги j -> t с весом d_j + q_j
        for j in S:
            w = self.jobs[j].d_i + self.jobs[j].q_i
            edges[j].append((t, w))

        return edges, t

    # ----------------------------------------------------------
    # Поиск максимального пути в DAG (0 -> t)
    # ----------------------------------------------------------
    def _longest_path(self, edges: Dict[int, List[Tuple[int, float]]], t_node: int) -> List[int]:
        """
        Находит критический путь в DAG от истока (0) до стока (t_node).
        Возвращает список работ на критическом пути (исключая 0 и t).
        """
        # Получаем все вершины графа
        vertices = list(edges.keys())

        # Топологическая сортировка:
        # 0 (исток) -> работы (по возрастанию ID) -> t (сток)
        def sort_key(v: int) -> Tuple[int, int]:
            if v == 0:
                return (0, 0)
            elif v == t_node:
                return (2, 0)
            else:
                return (1, v)

        vertices.sort(key=sort_key)

        # Инициализация расстояний и предков
        dist: Dict[int, float] = {v: float("-inf") for v in vertices}
        parent: Dict[int, Optional[int]] = {v: None for v in vertices}
        dist[0] = 0.0

        # Динамическое программирование в топологическом порядке
        for u in vertices:
            if dist[u] == float("-inf"):
                continue
            for v, w in edges.get(u, []):
                if dist[v] < dist[u] + w - 1e-9:
                    dist[v] = dist[u] + w
                    parent[v] = u

        # Восстановление пути от t_node к 0
        path: List[int] = []
        cur: Optional[int] = t_node
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

        # Возвращаем только работы (положительные индексы, исключая 0 и t_node)
        return [x for x in path if x > 0]

    # ----------------------------------------------------------
    # Algorithm 4: процедура пересоставления (Rescheduling)
    # ----------------------------------------------------------
    def _reschedule(self, S: List[int]) -> Tuple[List[int], float]:
        """
        Algorithm 4 из статьи.
        Переставляет отложенные работы (delayed jobs) для получения
        расписания, пригодного для ветвления.

        Теорема 2 гарантирует завершение не более чем за O(n²) итераций.
        """
        max_iter = self.n * self.n + 1  # O(n²) + 1 для безопасности
        it = 0

        # Копируем расписание для безопасной модификации
        S = list(S)

        # Вычисляем начальное расписание
        C_max, start_times = self.calculate_makespan(S)

        # Инициализируем r_prime (требуется для определения отложенных работ)
        r_prime = self._initialize_release_times()

        # Определяем отложенные работы
        delayed_jobs = self._get_delayed_jobs(S, start_times, r_prime)

        while it < max_iter and delayed_jobs:
            it += 1

            # Строим DAG и находим критический путь
            edges, t_node = self._build_dag(S)
            critical_path = self._longest_path(edges, t_node)

            if not critical_path:
                break

            # Ищем первый отложенный job на критическом пути
            delayed = None
            for j in critical_path:
                if j in delayed_jobs:
                    delayed = j
                    break

            if delayed is None:
                # Нет отложенных работ на критическом пути — выход
                break

            # i1 — первая работа на критическом пути
            i1 = critical_path[0]

            # Ищем работу j1, которая непосредственно предшествует i1 в расписании S
            idx_i1 = S.index(i1)
            if idx_i1 == 0:
                j1 = None  # i1 — первая работа, вставляем в начало
            else:
                j1 = S[idx_i1 - 1]

            # Удаляем delayed из текущей позиции
            S.remove(delayed)

            # Вставляем delayed сразу после j1 (т.е. перед i1)
            if j1 is None:
                S.insert(0, delayed)
            else:
                # После удаления delayed индекс i1 мог измениться, ищем его заново
                new_idx_i1 = S.index(i1)
                S.insert(new_idx_i1, delayed)

            # Пересчитываем расписание
            C_max, start_times = self.calculate_makespan(S)
            r_prime = self._initialize_release_times()
            delayed_jobs = self._get_delayed_jobs(S, start_times, r_prime)

        return S, C_max