from typing import Dict, Tuple, List
from collections import defaultdict, deque


def validate_graph(jobs, precedence) -> Tuple[bool, List[str]]:
    """
    Проверяет корректность графа задач.
    Возвращает:
        (is_valid, errors)
    """

    errors = []
    n = len(jobs)

    # ---------------------------------------------------------
    # 1. Проверка t_i, r_i, q_i
    # ---------------------------------------------------------
    for job in jobs:
        if job.d_i < 0:
            errors.append(f"t({job.id}) < 0")
        if job.r_i < 0:
            errors.append(f"r({job.id}) < 0")
        if job.q_i < 0:
            errors.append(f"q({job.id}) < 0")

    # ---------------------------------------------------------
    # 2. Проверка L(i,j)
    # ---------------------------------------------------------
    for (i, j), lij in precedence.items():
        job_i = next(job for job in jobs if job.id == i)
        if lij < job_i.d_i:
            errors.append(f"L({i},{j}) < t({i})")

    # ---------------------------------------------------------
    # 3. Проверка существования вершин
    # ---------------------------------------------------------
    job_ids = {job.id for job in jobs}
    for (i, j) in precedence:
        if i not in job_ids:
            errors.append(f"Edge ({i},{j}) has unknown source")
        if j not in job_ids:
            errors.append(f"Edge ({i},{j}) has unknown target")

    # ---------------------------------------------------------
    # 4. Проверка на циклы (Kahn's algorithm)
    # ---------------------------------------------------------
    indeg = defaultdict(int)
    graph = defaultdict(list)

    for (i, j), lij in precedence.items():
        graph[i].append(j)
        indeg[j] += 1

    queue = deque([job.id for job in jobs if indeg[job.id] == 0])
    visited = 0

    while queue:
        u = queue.popleft()
        visited += 1
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if visited != n:
        errors.append("Graph contains a cycle")

    # ---------------------------------------------------------
    # 5. Проверка достижимости всех задач
    # ---------------------------------------------------------
    reachable = set(queue)
    # Actually recompute reachable from all zero-indegree nodes
    queue = deque([job.id for job in jobs if indeg[job.id] == 0])
    reachable = set(queue)

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in reachable:
                reachable.add(v)
                queue.append(v)

    if len(reachable) != n:
        unreachable = job_ids - reachable
        errors.append(f"Unreachable tasks: {sorted(unreachable)}")

    # ---------------------------------------------------------
    # 6. Проверка, что каждая задача может стать ready
    # ---------------------------------------------------------
    for job in jobs:
        preds = [i for (i, j) in precedence if j == job.id]
        # если нет предшественников — ок
        if not preds:
            continue
        # если есть, но они недостижимы
        for p in preds:
            if p not in reachable:
                errors.append(f"Task {job.id} has unreachable predecessor {p}")

    return len(errors) == 0, errors
