"""
STG Loader — загрузчик графов задач из Standard Task Graph Set Project (.stg)

Формат строки:
    <task_id> <processing_time> <num_predecessors> <pred1> <pred2> ...

Особенности:
    - r_i = 0 (в STG нет release times)
    - q_i = 0 (в STG нет delivery times)
    - L(i,j) = d_i  (классическое правило STG: задержка равна длительности предшественника)
"""

from typing import List, Dict, Tuple
from core.job import Job


def load_stg(path: str) -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """
    Загружает STG-файл и возвращает:
        - список Job
        - словарь ограничений предшествования {(i, j): L(i,j)}

    Args:
        path: путь к .stg файлу

    Returns:
        jobs: List[Job]
        precedence: Dict[(i,j), L(i,j)]
    """

    jobs: List[Job] = []
    precedence: Dict[Tuple[int, int], float] = {}

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # первая строка — количество задач
    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError("Некорректный STG-файл: первая строка должна содержать число задач")

    # парсим задачи
    for line in lines[1:]:
        parts = line.split()

        # строки с комментариями пропускаем
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

        # r_i = 0, q_i = 0 (в STG нет этих параметров)
        jobs.append(Job(id=task_id, r_i=0.0, d_i=d_i, q_i=0.0))

        # L(i,j) = d_i (классическое правило STG)
        for p in preds:
            precedence[(p, task_id)] = d_i

    return jobs, precedence


def print_stg_summary(jobs: List[Job], precedence: Dict[Tuple[int, int], float]):
    """
    Печатает краткую сводку загруженного STG-графа.
    """

    print("====================================")
    print("         STG GRAPH SUMMARY          ")
    print("====================================")
    print(f"Tasks: {len(jobs)}")
    print(f"Edges: {len(precedence)}")
    print("------------------------------------")

    # статистика предшественников
    pred_count = {}
    for (i, j) in precedence:
        pred_count.setdefault(j, 0)
        pred_count[j] += 1

    if pred_count:
        max_pred = max(pred_count.values())
        min_pred = min(pred_count.values())
        avg_pred = sum(pred_count.values()) / len(pred_count)
    else:
        max_pred = min_pred = avg_pred = 0

    print(f"Max predecessors: {max_pred}")
    print(f"Min predecessors: {min_pred}")
    print(f"Avg predecessors: {avg_pred:.3f}")
    print("====================================")


if __name__ == '__main__':
    job, prec = load_stg('data/100/rand0000.stg')
    print_stg_summary(job, prec)
