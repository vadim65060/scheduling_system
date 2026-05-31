"""
Анализатор тестовых файлов (.stg и .astg)
Подсчитывает статистику по количеству работ и дуг (DPC)
"""

import os
from typing import Dict, Tuple

from tests.jobs_loader import load_any
mx_d = 1000

def analyze_file(filepath: str) -> Tuple[str, int, int, bool]:
    """
    Анализирует один файл и возвращает статистику

    Returns:
        (filename, num_jobs, num_edges, has_errors)
    """
    try:
        jobs, precedence = load_any(filepath)
        num_jobs = len(jobs)
        num_edges = len(precedence)
        global mx_d
        mx_d_t = min(job.d_i for job in jobs)
        mx_d = min(mx_d_t,mx_d)
        return (os.path.basename(filepath), num_jobs, num_edges, False)
    except Exception as e:
        print(f"Ошибка при загрузке {filepath}: {e}")
        return (os.path.basename(filepath), 0, 0, True)


def analyze_folder(folder_path: str, limit: int = 0) -> Dict:
    """
    Анализирует все файлы в папке

    Args:
        folder_path: путь к папке с тестами
        limit: ограничение на количество файлов (0 = все)

    Returns:
        словарь со статистикой
    """
    # Собираем все файлы .stg и .astg
    files = []
    for f in os.listdir(folder_path):
        if f.endswith(".stg") or f.endswith(".astg"):
            files.append(os.path.join(folder_path, f))

    if limit > 0:
        files = files[:limit]

    if not files:
        print(f"В папке {folder_path} не найдено .stg или .astg файлов")
        return {}

    print(f"\nАнализирую {len(files)} файлов...")

    # Анализируем каждый файл
    results = []
    jobs_counts = []
    edges_counts = []
    errors = 0

    for i, filepath in enumerate(files, 1):
        fname, num_jobs, num_edges, has_error = analyze_file(filepath)
        if has_error:
            errors += 1
        else:
            results.append((fname, num_jobs, num_edges))
            jobs_counts.append(num_jobs)
            edges_counts.append(num_edges)

        # Прогресс
        if i % 50 == 0:
            print(f"  Обработано {i}/{len(files)} файлов...")

    # Подсчёт статистики
    stats = {
        "total_files": len(files),
        "successful_files": len(jobs_counts),
        "errors": errors,
        "jobs": {
            "min": min(jobs_counts) if jobs_counts else 0,
            "max": max(jobs_counts) if jobs_counts else 0,
            "avg": sum(jobs_counts) / len(jobs_counts) if jobs_counts else 0,
            "sum": sum(jobs_counts) if jobs_counts else 0,
        },
        "edges": {
            "min": min(edges_counts) if edges_counts else 0,
            "max": max(edges_counts) if edges_counts else 0,
            "avg": sum(edges_counts) / len(edges_counts) if edges_counts else 0,
            "sum": sum(edges_counts) if edges_counts else 0,
        },
        "results": results,
    }

    return stats


def print_stats(stats: Dict, folder_name: str):
    """Красиво выводит статистику"""
    if not stats:
        return

    print("\n" + "=" * 60)
    print(f"📊 СТАТИСТИКА ТЕСТОВ: {folder_name}")
    print("=" * 60)

    print(f"\n📁 Всего файлов: {stats['total_files']}")
    print(f"✅ Успешно загружено: {stats['successful_files']}")
    print(f"❌ Ошибок: {stats['errors']}")

    if stats['successful_files'] == 0:
        print("\nНет успешно загруженных файлов для анализа")
        return

    print("\n" + "-" * 40)
    print("📊 КОЛИЧЕСТВО РАБОТ (JOBS):")
    print("-" * 40)
    print(f"   Минимум:  {stats['jobs']['min']}")
    print(f"   Максимум: {stats['jobs']['max']}")
    print(f"   Среднее:  {stats['jobs']['avg']:.2f}")
    print(f"   Сумма:    {stats['jobs']['sum']}")

    print("\n" + "-" * 40)
    print("🔗 КОЛИЧЕСТВО ДУГ (DPC EDGES):")
    print("-" * 40)
    print(f"   Минимум:  {stats['edges']['min']}")
    print(f"   Максимум: {stats['edges']['max']}")
    print(f"   Среднее:  {stats['edges']['avg']:.2f}")
    print(f"   Сумма:    {stats['edges']['sum']}")

    print("\n" + "-" * 40)
    print("📈 ПЛОТНОСТЬ ГРАФА:")
    print("-" * 40)

    # Средняя плотность
    densities = []
    for _, num_jobs, num_edges in stats['results']:
        if num_jobs > 1:
            # Максимально возможное количество дуг для ориентированного графа без петель
            max_possible = num_jobs * (num_jobs - 1)
            density = num_edges / max_possible * 100
            densities.append(density)

    if densities:
        print(f"   Средняя плотность: {sum(densities) / len(densities):.2f}%")
        print(f"   Мин. плотность:    {min(densities):.2f}%")
        print(f"   Макс. плотность:   {max(densities):.2f}%")

    print("\n" + "-" * 40)
    print("📋 ПЕРВЫЕ 10 ФАЙЛОВ (для ознакомления):")
    print("-" * 40)
    for i, (fname, jobs, edges) in enumerate(stats['results'][:10], 1):
        print(f"   {i:2}. {fname:30} | jobs: {jobs:4} | edges: {edges:5}")


def analyze_all_folders(base_dir: str = "data", limit: int = 0):
    """
    Анализирует все подпапки в base_dir
    """
    if not os.path.exists(base_dir):
        print(f"Папка {base_dir} не найдена!")
        return

    subfolders = [f for f in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, f))]

    if not subfolders:
        print(f"В {base_dir} нет подпапок с тестами")
        return

    all_stats = {}

    for folder_name in subfolders:
        folder_path = os.path.join(base_dir, folder_name)
        stats = analyze_folder(folder_path, limit)
        if stats:
            all_stats[folder_name] = stats
            print_stats(stats, folder_name)

    # Сводная статистика по всем папкам
    if len(all_stats) > 1:
        print("\n" + "=" * 60)
        print("📊 СВОДНАЯ СТАТИСТИКА ПО ВСЕМ ПАПКАМ")
        print("=" * 60)

        print(f"\n{'Папка':20} {'Файлов':8} {'Ср. Jobs':10} {'Ср. Edges':10} {'Плотность':10}")
        print("-" * 60)

        for folder_name, stats in all_stats.items():
            if stats['successful_files'] > 0:
                print(f"{folder_name:20} {stats['successful_files']:8} "
                      f"{stats['jobs']['avg']:10.1f} "
                      f"{stats['edges']['avg']:10.1f} "
                      f"{stats['edges']['avg'] / (stats['jobs']['avg'] ** 2 if stats['jobs']['avg'] > 0 else 1) * 100:9.2f}%")

    return all_stats


def main():
    """Интерактивный режим"""
    print("=" * 60)
    print("📊 АНАЛИЗАТОР ТЕСТОВЫХ ФАЙЛОВ")
    print("=" * 60)

    print("\nВыберите режим:")
    print("1. Анализировать конкретную папку")
    print("2. Анализировать все папки в data/")
    print("3. Анализировать конкретный файл")

    mode = input("\nВаш выбор (1-3): ").strip()

    if mode == "1":
        base_dir = "data"
        subfolders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f))]

        if not subfolders:
            print("Нет доступных папок!")
            return

        print("\nДоступные папки:")
        for i, folder in enumerate(subfolders, 1):
            print(f"  {i}. {folder}")

        idx = int(input("\nВыберите папку: ").strip()) - 1
        folder_path = os.path.join(base_dir, subfolders[idx])

        limit = int(input("Сколько файлов проанализировать (0 = все): ").strip() or "0")

        stats = analyze_folder(folder_path, limit)
        print_stats(stats, subfolders[idx])

    elif mode == "2":
        limit = int(input("Сколько файлов на папку (0 = все): ").strip() or "0")
        analyze_all_folders("data", limit)

    elif mode == "3":
        filepath = input("Введите путь к файлу: ").strip()
        fname, jobs, edges, error = analyze_file(filepath)
        if not error:
            print(f"\n📄 Файл: {fname}")
            print(f"   Количество работ (jobs): {jobs}")
            print(f"   Количество дуг (edges): {edges}")
            if jobs > 1:
                density = edges / (jobs * (jobs - 1)) * 100
                print(f"   Плотность графа: {density:.2f}%")
        else:
            print("Не удалось загрузить файл")

    else:
        print("Неверный выбор!")


if __name__ == "__main__":
    main()
    print(mx_d)