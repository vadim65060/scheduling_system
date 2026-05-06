"""
Анализ статистики ветвления для алгоритма BalasBaBDPC.
Параллельный запуск тестов + сбор метрик ветвления.
"""

import csv
import datetime
import multiprocessing as mp
import os
import time
from typing import List, Dict, Tuple

from tqdm import tqdm

from algorithms.BaBBalasDPC import BalasBaBDPC
from core.graph_validator import validate_graph
from tests.jobs_loader import load_any


# ----------------------------------------------------------------------
# Запуск одного теста (для параллельного выполнения)
# ----------------------------------------------------------------------

def _run_single_branching_test(args: Tuple[str, str, float]) -> Dict:
    """
    Запускает BalasBaBDPC на одном файле и собирает статистику ветвления.
    Вынесена на уровень модуля для работы с multiprocessing.
    """
    path, folder, timeout = args

    fname = os.path.basename(path)

    # Загружаем задачу
    try:
        jobs, precedence = load_any(path)
    except Exception as e:
        return {
            "file": fname,
            "error": f"Ошибка загрузки: {str(e)}",
            "nodes_explored": 0,
            "strong_branches": 0,
            "weak_branches": 0,
            "total_branches": 0,
            "total_subproblems": 0,
            "strong_pct": 0.0,
            "C_max": float("inf"),
            "time": 0.0,
            "timed_out": False,
            "optimal": False,
        }

    # Проверка графа
    ok, errors = validate_graph(jobs, precedence)
    if not ok:
        return {
            "file": fname,
            "error": f"Ошибка графа: {errors}",
            "nodes_explored": 0,
            "strong_branches": 0,
            "weak_branches": 0,
            "total_branches": 0,
            "total_subproblems": 0,
            "strong_pct": 0.0,
            "C_max": float("inf"),
            "time": 0.0,
            "timed_out": False,
            "optimal": False,
        }

    # Запускаем алгоритм
    scheduler = BalasBaBDPC(jobs, precedence, time_limit=timeout)
    start = time.time()
    schedule, C_max, stats = scheduler.solve(timeout=timeout)
    elapsed = time.time() - start

    # Извлекаем статистику
    nodes = stats.get("nodes_explored", 0)
    strong = stats.get("strong_branches", 0)
    weak = stats.get("weak_branches", 0)

    # Общее количество ветвлений
    total_branches = strong + weak

    # Каждое ветвление создаёт 2 новые подзадачи
    total_subproblems = 2 * total_branches

    # Процент сильного ветвления
    strong_pct = (strong / total_branches * 100) if total_branches > 0 else 0.0

    return {
        "file": fname,
        "nodes_explored": nodes,
        "strong_branches": strong,
        "weak_branches": weak,
        "total_branches": total_branches,
        "total_subproblems": total_subproblems,
        "strong_pct": strong_pct,
        "C_max": C_max,
        "time": elapsed,
        "timed_out": stats.get("timed_out", False),
        "optimal": stats.get("optimal", False),
    }


# ----------------------------------------------------------------------
# Параллельный запуск
# ----------------------------------------------------------------------

def run_parallel_branching(tasks: List[Tuple[str, str, float]]) -> List[Dict]:
    """
    Параллельный запуск тестов с progress-bar.
    """
    results = []
    num_workers = min(mp.cpu_count(), len(tasks))

    with mp.Pool(processes=num_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(_run_single_branching_test, tasks),
            total=len(tasks),
            desc="Анализ ветвления",
            ncols=80
        ):
            results.append(res)

    return results


# ----------------------------------------------------------------------
# Основная функция запуска
# ----------------------------------------------------------------------

def run_branching_analysis(
    folder: str,
    limit: int = 0,
    timeout: float = 30.0
) -> List[Dict]:
    """
    Запускает алгоритм BalasBaBDPC на всех тестах из папки
    и собирает подробную статистику по ветвлению.

    Args:
        folder: путь к папке с тестами (.stg / .astg)
        limit: сколько тестов использовать (0 = все)
        timeout: таймаут на один тест в секундах

    Returns:
        Список словарей с результатами для каждого теста
    """
    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".stg") or f.endswith(".astg")
    )
    if limit > 0:
        files = files[:limit]

    print(f"\nЗапуск анализа ветвления...")
    print(f"Тестов: {len(files)}")
    print(f"Таймаут: {timeout} с")
    print(f"Потоков: {min(mp.cpu_count(), len(files))}\n")

    # Формируем задачи
    tasks = [(os.path.join(folder, fname), folder, timeout) for fname in files]

    # Запускаем параллельно
    start_time = time.time()
    results = run_parallel_branching(tasks)
    elapsed = time.time() - start_time

    print(f"\nГотово! Общее время: {elapsed:.1f} с")

    return results


# ----------------------------------------------------------------------
# Вывод статистики
# ----------------------------------------------------------------------

def print_branching_stats(results: List[Dict], folder_name: str = ""):
    """
    Печатает сводную статистику по ветвлению.
    """
    if not results:
        print("Нет данных для анализа")
        return

    # Фильтруем тесты без ошибок
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        print("Все тесты завершились с ошибками")
        return

    # Суммарные показатели
    total_nodes = sum(r["nodes_explored"] for r in valid)
    total_strong = sum(r["strong_branches"] for r in valid)
    total_weak = sum(r["weak_branches"] for r in valid)
    total_branches = total_strong + total_weak
    total_subproblems = 2 * total_branches
    overall_strong_pct = (total_strong / total_branches * 100) if total_branches > 0 else 0.0

    # Средние значения
    n = len(valid)
    avg_nodes = total_nodes / n
    avg_strong = total_strong / n
    avg_weak = total_weak / n
    avg_subproblems = total_subproblems / n
    avg_time = sum(r["time"] for r in valid) / n

    # Категории тестов
    only_strong = sum(1 for r in valid if r["strong_branches"] > 0 and r["weak_branches"] == 0)
    only_weak = sum(1 for r in valid if r["weak_branches"] > 0 and r["strong_branches"] == 0)
    mixed = sum(1 for r in valid if r["strong_branches"] > 0 and r["weak_branches"] > 0)
    no_branching = sum(1 for r in valid if r["strong_branches"] == 0 and r["weak_branches"] == 0)

    timed_out_count = sum(1 for r in valid if r.get("timed_out", False))
    optimal_count = sum(1 for r in valid if r.get("optimal", False))

    # Вывод
    print(f"\n{'='*70}")
    print(f"  СТАТИСТИКА ВЕТВЛЕНИЯ ДЛЯ АЛГОРИТМА BalasB&B DPC")
    if folder_name:
        print(f"  Папка: {folder_name}")
    print(f"{'='*70}")
    print(f"  Всего тестов:              {len(results):6d}")
    print(f"  Успешно решено:            {len(valid):6d}")
    print(f"  С ошибками:                {len(errors):6d}")
    print(f"  Оптимально доказано:       {optimal_count:6d}")
    print(f"  Прервано по таймауту:      {timed_out_count:6d}")
    print(f"{'='*70}")
    print(f"  ВСЕГО узлов исследовано:   {total_nodes:8d}")
    print(f"  В среднем на тест:         {avg_nodes:8.1f}")
    print(f"{'='*70}")
    print(f"  ВСЕГО сильных ветвлений:   {total_strong:8d}  ({overall_strong_pct:5.1f}%)")
    print(f"  ВСЕГО слабых ветвлений:    {total_weak:8d}  ({100-overall_strong_pct:5.1f}%)")
    print(f"  ВСЕГО ветвлений:           {total_branches:8d}")
    print(f"  ВСЕГО сгенерировано подзадач: {total_subproblems:8d}")
    print(f"{'='*70}")
    print(f"  В среднем на тест:")
    print(f"    Узлов:                   {avg_nodes:8.2f}")
    print(f"    Сильных ветвлений:       {avg_strong:8.2f}")
    print(f"    Слабых ветвлений:        {avg_weak:8.2f}")
    print(f"    Новых подзадач:          {avg_subproblems:8.2f}")
    print(f"{'='*70}")
    print(f"  Распределение тестов по типам ветвления:")
    print(f"    Только сильное:          {only_strong:6d}")
    print(f"    Только слабое:           {only_weak:6d}")
    print(f"    Оба типа:                {mixed:6d}")
    print(f"    Без ветвления:           {no_branching:6d}")
    print(f"{'='*70}")
    print(f"  Среднее время на тест:     {avg_time:8.2f} с")
    print(f"{'='*70}\n")

    # Детальная таблица
    print(f"{'Файл':40} {'Узлы':>6} {'Сильн':>6} {'Слаб':>6} {'%Сильн':>7} {'Подзадач':>8} {'C_max':>8} {'Время':>7}")
    print("-" * 95)
    for r in sorted(valid, key=lambda x: x["nodes_explored"], reverse=True):
        print(
            f"{r['file']:40} "
            f"{r['nodes_explored']:6d} "
            f"{r['strong_branches']:6d} "
            f"{r['weak_branches']:6d} "
            f"{r['strong_pct']:6.1f}% "
            f"{r['total_subproblems']:8d} "
            f"{r['C_max']:8.0f} "
            f"{r['time']:6.2f}s"
        )

    # Тесты с ошибками
    if errors:
        print(f"\nТесты с ошибками ({len(errors)}):")
        for r in errors:
            print(f"  {r['file']}: {r.get('error', 'неизвестная ошибка')}")


# ----------------------------------------------------------------------
# Сохранение в CSV
# ----------------------------------------------------------------------

def save_branching_stats_to_csv(results: List[Dict], filename: str):
    """
    Сохраняет статистику ветвления в CSV-файл.
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "nodes_explored", "strong_branches", "weak_branches",
            "total_branches", "total_subproblems", "strong_pct",
            "C_max", "time", "timed_out", "optimal"
        ])

        for r in results:
            writer.writerow([
                r.get("file", ""),
                r.get("nodes_explored", 0),
                r.get("strong_branches", 0),
                r.get("weak_branches", 0),
                r.get("total_branches", 0),
                r.get("total_subproblems", 0),
                f"{r.get('strong_pct', 0):.2f}",
                r.get("C_max", float("inf")),
                f"{r.get('time', 0):.3f}",
                r.get("timed_out", False),
                r.get("optimal", False),
            ])


# ----------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  АНАЛИЗ ЭФФЕКТИВНОСТИ ВЕТВЛЕНИЯ")
    print("  Алгоритм: BalasB&B DPC (Balas et al., 1995)")
    print("=" * 60)

    # Выбор папки
    base_dir = "data"
    subfolders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ]

    if not subfolders:
        print(f"Нет подпапок в {base_dir}!")
        exit(1)

    print("\nДоступные папки с тестами:")
    for i, folder_name in enumerate(subfolders, 1):
        # Считаем количество тестов в папке
        folder_path = os.path.join(base_dir, folder_name)
        n_tests = len([f for f in os.listdir(folder_path)
                       if f.endswith(".stg") or f.endswith(".astg")])
        print(f"  {i}. {folder_name} ({n_tests} тестов)")

    idx = int(input("\nВыберите папку: ").strip()) - 1
    if idx < 0 or idx >= len(subfolders):
        print("Неверный выбор!")
        exit(1)

    folder = os.path.join(base_dir, subfolders[idx])
    folder_name = subfolders[idx]

    limit = int(input("Сколько тестов использовать (0 = все): ").strip() or "0")
    timeout = float(input("Таймаут на тест в секундах (по умолчанию 60): ").strip() or "60")

    print(f"\n{'='*60}")
    print(f"  Папка: {folder_name}")
    print(f"  Тестов: {'все' if limit == 0 else limit}")
    print(f"  Таймаут: {timeout} с")
    print(f"{'='*60}")

    print(f"\nНачало: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Запуск анализа
    results = run_branching_analysis(folder, limit=limit, timeout=timeout)

    # Вывод статистики
    print_branching_stats(results, folder_name)

    print(f"Окончание: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Сохранение результатов
    save = input("\nСохранить результаты в CSV? (y/n): ").strip().lower()
    if save == "y":
        os.makedirs("results", exist_ok=True)
        csv_name = f"results/branching_stats_{folder_name}.csv"
        save_branching_stats_to_csv(results, csv_name)
        print(f"Результаты сохранены в {csv_name}")