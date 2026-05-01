"""
Оценка влияния DPC с помощью RTE на разных эвристиках.
Запускает RTE(LTH), RTE(MLTH), RTE(ILTF), RTE(BestOfHeuristics), RTE(BalasB&BDPC)
и выводит средние значения Cmax_relaxed, Cmax_real и Δ.

Поддерживает параллельный запуск и прогресс-бар (tqdm).
"""

import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type

from tqdm import tqdm

from algorithms.BaBBalasDPC import BalasBaBDPC
from algorithms.rte import RTE
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from algorithms.iltf import ILTF
from algorithms.BestOfHeuristics import BestOfHeuristics
from core.job import Job

# Импортируем загрузчик тестов из batch_runner
from batch_runner import load_any
from core.validator import validate_schedule
# ----------------------------------------------------------------------
# Конфигурация эвристик
# ----------------------------------------------------------------------
HEURISTICS: Dict[str, Type] = {
    "LTH": LTH,
    "MLTH": MLTH,
    "ILTF": ILTF,
    "BestOfHeuristics": BestOfHeuristics,
    "BalasB&BDPC": BalasBaBDPC,
}


# ----------------------------------------------------------------------
# Функция для параллельного запуска (одна задача)
# ----------------------------------------------------------------------
def run_rte_task(args: Tuple[str, str, str, float]) -> Dict:
    """
    Запускает RTE с конкретной эвристикой на одном файле.

    Args:
        args: Кортеж (folder, filename, heuristic_name, timeout)

    Returns:
        Словарь с результатами
    """
    folder, fname, hname, timeout = args
    path = os.path.join(folder, fname)

    try:
        jobs, precedence = load_any(path)
    except Exception as e:
        return {
            "file": fname,
            "heuristic": hname,
            "C_relaxed": float("inf"),
            "C_real": float("inf"),
            "delta_pct": float("inf"),
            "error": str(e),
        }

    hclass = HEURISTICS[hname]

    # Особый случай для BalasBaBDPC и BestOfHeuristics
    rte = RTE(
        jobs=jobs,
        precedence_constraints=precedence.copy(),
        heuristic_class=hclass,
    )

    schedule, C_real, stats = rte.solve(timeout=timeout)
    C_relaxed = stats.get("heuristic_makespan", float("inf"))
    delta_pct = (C_real - C_relaxed) / C_relaxed * 100 if C_relaxed > 0 else 0.0
    jobs_dict = {job.id: job for job in jobs}
    is_valid, result = validate_schedule(
        schedule,
        jobs_dict,
        precedence,
        verbose=False
    )
    if not is_valid:
        print(f'INVALID schedule!\n"file": {fname},\n"heuristic": {hname}')
        return {
            "file": fname,
            "heuristic": hname,
            "C_relaxed": float("inf"),
            "C_real": float("inf"),
            "delta_pct": float("inf"),
            "error": 'INVALID schedule!',
        }

    return {
        "file": fname,
        "heuristic": hname,
        "C_relaxed": C_relaxed,
        "C_real": C_real,
        "delta_pct": delta_pct,
    }


def evaluate_rte_on_folder(
    folder: str,
    limit: int = 0,
    timeout: float = 30.0,
) -> Dict[str, Dict[str, float]]:
    """
    Для каждого файла в папке запускает RTE с разными эвристиками
    и собирает средние показатели.

    Использует параллельный запуск через multiprocessing.Pool.

    Возвращает словарь:
        { "LTH": {"avg_relaxed": ..., "avg_real": ..., "avg_delta_pct": ...}, ... }
    """
    files = sorted(
        f for f in os.listdir(folder) if f.endswith(".stg") or f.endswith(".astg")
    )
    if limit > 0:
        files = files[:limit]

    print(f"\nТестов: {len(files)}")
    print(f"Эвристики: {', '.join(HEURISTICS.keys())}")
    print(f"Таймаут: {timeout} с")
    print(f"Процессов: {mp.cpu_count()}\n")

    # Формируем список задач
    tasks = []
    for fname in files:
        for hname in HEURISTICS:
            tasks.append((folder, fname, hname, timeout))

    # Параллельный запуск с прогресс-баром
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for res in tqdm(
            pool.imap_unordered(run_rte_task, tasks),
            total=len(tasks),
            desc="Выполнение RTE-тестов",
            ncols=80,
        ):
            results.append(res)

    # Агрегируем результаты
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)

    for r in results:
        hname = r["heuristic"]
        if r.get("error"):
            print(f"  [Ошибка] {r['file']} / {hname}: {r['error']}")
            continue

        sums[hname]["relaxed"] += r["C_relaxed"]
        sums[hname]["real"] += r["C_real"]
        sums[hname]["delta_pct"] += r["delta_pct"]
        counts[hname] += 1

    # Усредняем
    result: Dict[str, Dict[str, float]] = {}
    for hname in HEURISTICS:
        n = counts[hname]
        if n == 0:
            result[hname] = {
                "avg_relaxed": float("inf"),
                "avg_real": float("inf"),
                "avg_delta_pct": float("inf"),
                "tests": 0,
            }
        else:
            result[hname] = {
                "avg_relaxed": sums[hname]["relaxed"] / n,
                "avg_real": sums[hname]["real"] / n,
                "avg_delta_pct": sums[hname]["delta_pct"] / n,
                "tests": n,
            }

    return result


def print_table(results: Dict[str, Dict[str, float]]):
    """Печатает таблицу в формате, близком к примеру."""
    print("\n" + "=" * 75)
    print("Оценка влияния DPC (RTE)")
    print("=" * 75)
    print(
        f"{'Эвристика':22} {'Тестов':>6} {'C_relaxed_avg':>14} {'C_real_avg':>12} {'Δ %':>8}"
    )
    print("-" * 75)

    # Сортируем по real Cmax для наглядности
    sorted_results = sorted(results.items(), key=lambda kv: kv[1]["avg_real"])

    for hname, r in sorted_results:
        if r["avg_real"] == float("inf"):
            print(f"{hname:22} {r['tests']:6} {'N/A':>14} {'N/A':>12} {'N/A':>8}")
        else:
            print(
                f"{hname:22} {r['tests']:6} {r['avg_relaxed']:14.1f} {r['avg_real']:12.1f} {r['avg_delta_pct']:+7.1f}%"
            )

    # Лучшие показатели
    valid_results = {k: v for k, v in results.items() if v["avg_real"] != float("inf")}
    if not valid_results:
        print("\nНет валидных результатов.")
        return

    best_delta = min(valid_results.items(), key=lambda kv: abs(kv[1]["avg_delta_pct"]))
    best_real = min(valid_results.items(), key=lambda kv: kv[1]["avg_real"])
    best_relaxed = min(valid_results.items(), key=lambda kv: kv[1]["avg_relaxed"])

    print("\n" + "-" * 75)
    print("Выводы:")
    print(
        f"  • {best_delta[0]:22} — наименее чувствителен к DPC "
        f"(Δ = {best_delta[1]['avg_delta_pct']:+.1f}%)"
    )
    print(
        f"  • {best_real[0]:22} — лучший реальный Cmax "
        f"(avg = {best_real[1]['avg_real']:.1f})"
    )
    print(
        f"  • {best_relaxed[0]:22} — лучший relaxed Cmax "
        f"(avg = {best_relaxed[1]['avg_relaxed']:.1f})"
    )


# ----------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RTE Evaluator — оценка влияния DPC")
    print("=" * 60)

    # Выбор папки
    base_dir = "data"
    if not os.path.isdir(base_dir):
        print(f"Папка '{base_dir}' не найдена. Укажите путь к папке с тестами:")
        base_dir = input("> ").strip()
        if not os.path.isdir(base_dir):
            print("Папка не существует. Выход.")
            return

    subfolders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    if not subfolders:
        print(f"В папке '{base_dir}' нет подпапок с тестами.")
        return

    print("\nДоступные папки с тестами:")
    for i, folder_name in enumerate(subfolders, 1):
        print(f"  {i}. {folder_name}")

    try:
        idx = int(input("\nВыберите папку: ").strip()) - 1
        folder = os.path.join(base_dir, subfolders[idx])
    except (ValueError, IndexError):
        print("Некорректный выбор. Выход.")
        return

    try:
        limit = int(input("Сколько тестов использовать (0 = все): ").strip() or "0")
    except ValueError:
        limit = 0

    print(f"\nПапка: {folder}")
    start_ts = time.time()

    results = evaluate_rte_on_folder(folder, limit=limit)
    print_table(results)

    elapsed = time.time() - start_ts
    print(f"\nОбщее время: {elapsed:.1f} с")


if __name__ == "__main__":
    main()