"""
Универсальный загрузчик тестовых файлов (.stg и .astg) с возможностью
генерации усложнённых вариантов (одна работа становится «узким местом»).

Использование:
    from universal_loader import load_any

    jobs, precedence = load_any("path/to/file.stg")
    jobs, precedence = load_any("path/to/file.astg", hard=True, hard_ratio=0.1, blowup_factor=2.0)
"""

import copy
import random
from typing import List, Dict, Tuple, Optional

from core.job import Job


# ======================================================================
# Загрузчики конкретных форматов
# ======================================================================

def load_astg_jobs(path: str) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[Tuple[int, int], int]]:
    """Загружает .astg файл.
    Возвращает:
        jobs:  {job_id: (processing_time, release_time, delivery_time)}
        edges: {(i, j): L(i, j)}
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    n = int(lines[0])
    jobs = {}
    edges = {}

    idx = 1
    for _ in range(n):
        id_, t, r, q = map(int, lines[idx].split())
        jobs[id_] = (t, r, q)
        idx += 1

    for line in lines[idx:]:
        i, j, lij = map(int, line.split())
        edges[(i, j)] = lij

    return jobs, edges


def load_stg(path: str) -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """Загружает .stg файл (Standard Task Graph).
    В STG нет release times и delivery times, они полагаются равными 0.
    L(i, j) = d_i (классическое правило STG).
    """
    jobs: List[Job] = []
    precedence: Dict[Tuple[int, int], float] = {}

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError("Некорректный STG-файл: первая строка должна содержать число задач")

    for line in lines[1:]:
        parts = line.split()

        if parts[0].startswith("#"):
            continue

        if len(parts) < 3:
            continue

        task_id = int(parts[0])
        d_i = float(parts[1])
        k = int(parts[2])

        preds = []
        if k > 0:
            preds = list(map(int, parts[3:3 + k]))

        jobs.append(Job(id=task_id, r_i=0.0, d_i=d_i, q_i=0.0))

        for p in preds:
            precedence[(p, task_id)] = d_i

    return jobs, precedence


# ======================================================================
# Основной загрузчик
# ======================================================================

def load_any(
        path: str,
        hard: bool = False,
        hard_ratio: float = 0.1,
        blowup_factor: float = 2.0,
        seed: Optional[int] = 0,
) -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """
    Универсальный загрузчик тестовых файлов.

    Определяет формат файла по расширению:
      *.stg  → load_stg
      *.astg → load_astg_jobs

    Если hard=True, после загрузки делает случайную работу «узким местом»:
      - строит быстрое LTH-расписание;
      - выбирает случайную работу среди первых (hard_ratio * 100)% расписания;
      - устанавливает её длительность > суммы длительностей всех остальных работ;
      - корректирует связанные с ней DPC.

    Args:
        path:          путь к .stg или .astg файлу
        hard:          если True, генерирует усложнённый вариант
        hard_ratio:    доля начала расписания, из которой выбирается работа
        blowup_factor: множитель для новой длительности (умножается на сумму остальных)
        seed:          зерно для random (None — не устанавливается)

    Returns:
        (jobs, precedence)  –  список Job и словарь DPC
    """
    if seed is not None:
        pass
        # random.seed(seed)

    # --- Загрузка исходных данных ---
    if path.endswith(".astg"):
        raw_jobs, raw_edges = load_astg_jobs(path)
        jobs = [Job(id=job_id, d_i=t, r_i=r, q_i=q) for job_id, (t, r, q) in raw_jobs.items()]
        precedence = {(i, j): int(lij) for (i, j), lij in raw_edges.items()}
    elif path.endswith(".stg"):
        jobs, precedence = load_stg(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path}")

    # --- Усложнение (если требуется) ---
    if not hard:
        return jobs, precedence

    # Строим быстрое расписание с помощью LTH
    from algorithms.iltf import ILTF
    algo = ILTF(copy.deepcopy(jobs), copy.deepcopy(precedence))
    schedule, _, _ = algo.solve()

    if not schedule or len(schedule) < 2:
        return jobs, precedence

    # Выбираем работу среди первых (hard_ratio * 100)% расписания
    n = len(schedule)
    end_idx = max(1, int(n * hard_ratio))
    candidates = schedule[:end_idx]
    #
    # if not candidates:
    #     return jobs, precedence

    target_job_id = candidates[-1]

    # Сумма длительностей всех работ
    total_sum = sum(j.d_i for j in jobs)

    # Новая длительность
    new_duration = int(total_sum * blowup_factor)
    jobs[target_job_id].d_i = new_duration

    # Корректируем DPC: для всех дуг, где target_job_id — предшественник,
    # задержка должна быть не меньше новой длительности
    keys_to_update = []
    for (i, j), lij in precedence.items():
        if i == target_job_id:
            precedence[(i, j)] = max(new_duration, lij)

    for i, j, new_lij in keys_to_update:
        precedence[(i, j)] = new_lij

    return jobs, precedence
