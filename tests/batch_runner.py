"""
Пакетный прогон алгоритмов по STG-файлам и новым astg-файлам.
 - Параллельный запуск
 - Финальное сравнение алгоритмов
 - Графики
 - Относительный C_max
 - Выбор количества тестов
"""
import csv
import datetime
import multiprocessing as mp
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
from tqdm import tqdm

from algorithms.BaBBalasDPC import BalasBaBDPC
from algorithms.Balas_DPC import BalasDPC
from algorithms.BestOfHeuristics import BestOfHeuristics
from algorithms.rte import RTE

from algorithms.iltf import ILTF
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from astg_jobs_loader import load_astg_jobs
from core.graph_validator import validate_graph
from core.job import Job
from stg_loader import load_stg

BALAS_DPS_ALGO = BestOfHeuristics
RTE_ALGO = BalasBaBDPC
ALGORITHMS = {
    # "BalasB&B DPC": BalasBaBDPC,
    f"RTE({RTE_ALGO.__name__ if RTE_ALGO.__name__ != "BestOfHeuristics" else "BoH"})": RTE,
    "LTH": LTH,
    "MLTH": MLTH,
    "ILTF": ILTF,
    "BoH": BestOfHeuristics,
    f"BalasDPC({BALAS_DPS_ALGO.__name__ if BALAS_DPS_ALGO.__name__ != "BestOfHeuristics" else "BoH"})": BalasDPC,
}


# ----------------------------------------------------------------------
# Универсальный загрузчик тестов
# ----------------------------------------------------------------------

def load_any(path: str):
    """
    Определяет формат файла:
      *.stg → load_stg
      *.astg → load_augmented
    Возвращает:
      jobs: List[Job]
      precedence: Dict[(i,j), L(i,j)]
    """

    if path.endswith(".astg"):
        raw_jobs, raw_edges = load_astg_jobs(path)

        jobs = []
        for job_id, (t, r, q) in raw_jobs.items():
            jobs.append(Job(id=job_id, d_i=t, r_i=r, q_i=q))

        precedence = {(i, j): lij for (i, j), lij in raw_edges.items()}
        return jobs, precedence

    else:
        # стандартный STG
        return load_stg(path)


# ----------------------------------------------------------------------
# Запуск одного алгоритма
# ----------------------------------------------------------------------

def run_single(path: str, algo_name: str, timeout: float = 10) -> Dict:
    jobs, precedence = load_any(path)

    # Проверка графа
    ok, errors = validate_graph(jobs, precedence)
    if not ok:
        # print(errors)
        return {
            "file": os.path.basename(path),
            "algorithm": algo_name,
            "C_max": float("inf"),
            "time": 0,
            "error": errors,
            "schedule": [],
        }

    Algo = ALGORITHMS[algo_name]
    if Algo is BalasBaBDPC:
        scheduler = Algo(jobs, precedence, time_limit=60)
    elif Algo is BalasDPC:
        scheduler = Algo(jobs, precedence, heuristic_class=BestOfHeuristics)
    else:
        scheduler = Algo(jobs, precedence)

    start = time.time()
    schedule, C_max, stats = scheduler.solve(timeout=timeout)
    elapsed = time.time() - start

    return {
        "file": os.path.basename(path),
        "algorithm": algo_name,
        "C_max": C_max,
        "time": elapsed,
        "schedule": schedule,
    }


# ----------------------------------------------------------------------
# Параллельный запуск
# ----------------------------------------------------------------------


def run_task(args):
    path, algo_name, timeout = args
    return run_single(path, algo_name, timeout)


def run_parallel(tasks):
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for res in tqdm(
                pool.imap_unordered(run_task, tasks),
                total=len(tasks),
                desc="Выполнение тестов",
                ncols=80
        ):
            results.append(res)
    return results


# ----------------------------------------------------------------------
# Пакетный прогон
# ----------------------------------------------------------------------

def run_batch(folder: str,
              algorithms: List[str],
              limit: int,
              timeout: float = 30.0,
              verbose: bool = False) -> List[Dict]:
    files = sorted(f for f in os.listdir(folder) if f.endswith(".stg") or f.endswith(".astg"))
    if limit > 0:
        files = files[:limit]

    print(f"\nИспользуем тестов: {len(files)}")
    print(f"Алгоритмы: {', '.join(algorithms)}\n")

    tasks = []
    for fname in files:
        path = os.path.join(folder, fname)
        for algo in algorithms:
            tasks.append((path, algo, timeout))

    print(f"Запуск {len(tasks)} задач в {min(mp.cpu_count(), len(tasks))} потоках...\n")
    results = run_parallel(tasks)

    # Если запускаем один алгоритм — выводим результаты каждого теста
    if verbose and len(algorithms) == 1:
        algo = algorithms[0]
        print(f"Результаты для алгоритма {algo}:\n")
        for r in results:
            print(f"{r['file']:15}  C_max={r['C_max']:6.1f}  time={r['time']:.3f}s")
            if 'error' in r:
                print(r['error'])

    return results


# ----------------------------------------------------------------------
# Финальное сравнение алгоритмов
# ----------------------------------------------------------------------

def final_comparison(results: List[Dict]):
    print("\n================= ФИНАЛЬНОЕ СРАВНЕНИЕ АЛГОРИТМОВ =================")

    stats: Dict[str, Dict[str, float]] = {}

    for algo in ALGORITHMS.keys():
        algo_results = [r for r in results if r["algorithm"] == algo]
        if not algo_results:
            continue

        C_vals = [r["C_max"] for r in algo_results]
        T_vals = [r["time"] for r in algo_results]

        stats[algo] = {
            "tests": len(algo_results),
            "avg_C": sum(C_vals) / len(C_vals),
            "min_C": min(C_vals),
            "max_C": max(C_vals),
            "avg_time": sum(T_vals) / len(T_vals),
        }
    best_algo = min(stats.items(), key=lambda x: x[1]["avg_C"])

    print(f"{'Алгоритм':15} {'Тестов':7} {'  %':6} {'C_avg':8} {'C_min':8} {'C_max':8} {'t_avg (s)':10}")
    print("-" * 60)

    for algo, s in sorted(stats.items(), key=lambda x: x[1]["avg_C"]):
        print(f"{algo:15} {s['tests']:7d} {s['avg_C']/best_algo[1]['avg_C']:6.2f} {s['avg_C']:8.2f} {s['min_C']:8.0f} "
              f"{s['max_C']:8.0f} {s['avg_time']:10.3f}")

    print("\nЛучший алгоритм по среднему C_max:", best_algo[0])

    return stats


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

def plot_results(stats: Dict[str, Dict[str, float]]):
    algos = list(stats.keys())
    avg_C = [stats[a]["avg_C"] for a in algos]
    avg_time = [stats[a]["avg_time"] for a in algos]

    best = min(avg_C)
    rel_C = [c / best for c in avg_C]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].bar(algos, rel_C, color="skyblue")
    axs[0].set_title("Относительный средний C_max")
    axs[0].set_ylabel("C_avg / C_best")
    axs[0].grid(axis="y", linestyle="--", alpha=0.5)

    axs[1].bar(algos, avg_time, color="salmon")
    axs[1].set_title("Среднее время работы")
    axs[1].set_ylabel("Время (сек)")
    axs[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def save_results_to_csv(results: List[Dict], filename: str):
    """
    Сохраняет результаты тестов в CSV.
    Поля:
      file, algorithm, C_max, time, schedule
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "algorithm", "C_max", "time", "schedule"])

        for r in results:
            schedule = r.get("schedule", [])
            writer.writerow([
                r["file"],
                r["algorithm"],
                r["C_max"],
                r["time"],
                " ".join(map(str, schedule))
            ])


def check_time(results: List[Dict]):
    sum_time = 0
    k = 0
    for r in results:
        if r["time"] < 20:
            sum_time += r["time"]
            k += 1

    print('k=', k, "sr time=", sum_time / k)


# ----------------------------------------------------------------------
# Интерфейс
# ----------------------------------------------------------------------

def main():
    print("Выберите режим:")
    print("1. Прогнать один алгоритм")
    print("2. Прогнать все алгоритмы и сравнить")

    mode = input("\nВаш выбор: ").strip()

    # --- Выбор папки с тестами ---
    base_dir = "data"
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    print("\nДоступные папки с тестами:")
    for i, folder_name in enumerate(subfolders, 1):
        print(f"{i}. {folder_name}")

    idx = int(input("\nВыберите папку: ").strip()) - 1
    folder = os.path.join(base_dir, subfolders[idx])

    print(f"\nИспользуется папка: {folder}")

    limit = int(input("Сколько тестов использовать (0 = все): ").strip() or "0")

    if mode == "1":
        print("\nВыберите алгоритм:")
        for i, name in enumerate(ALGORITHMS.keys(), 1):
            print(f"{i}. {name}")

        idx = int(input("\nВаш выбор: ").strip()) - 1
        algo = list(ALGORITHMS.keys())[idx]

        print(f"\nЗапуск алгоритма {algo}...\n")
        print(datetime.datetime.now())
        start = time.time()
        results = run_batch(folder, [algo], limit, verbose=True)
        stats = final_comparison(results)
        print(f'{time.time() - start}s')
        check_time(results)
        csv_name = f"results_{os.path.basename(folder)}.csv"
        # save_results_to_csv(results, csv_name)
        # print(f"\nРезультаты сохранены в файл: {csv_name}")

    else:
        print("\nЗапуск всех алгоритмов...\n")
        print(datetime.datetime.now())
        start = time.time()
        results = run_batch(folder, list(ALGORITHMS.keys()), limit)
        stats = final_comparison(results)
        print(f'{time.time()-start}s')
        csv_name = f"results_{os.path.basename(folder)}.csv"
        # save_results_to_csv(results, csv_name)
        plot_results(stats)


if __name__ == "__main__":
    main()

#      Алгоритм     Тестов  C_avg    C_min    C_max    t_avg (s)
# LTH  BalasDPC         180  4716.16  3813.00  6848.00      0.983
# MLTH BalasDPC         180  4720.85  3832.00  6822.00      1.011
# ILTF BalasDPC         180  4759.04  3849.00  6787.00      0.989
