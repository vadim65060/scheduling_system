#!/usr/bin/env python3
"""
Универсальный профайлер для тестирования алгоритмов задачи 1|r_j, q_j, DPC|C_max.
Позволяет легко переключаться между алгоритмами, тестовыми файлами и настройками.
"""

import cProfile
import pstats
import io
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Type

from algorithms.BaBBalasDPC import BalasBaBDPC
from algorithms.BestOfHeuristics import BestOfHeuristics
from algorithms.iltf import ILTF
from algorithms.l_depth import LDepth
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from algorithms.rte import RTE
from core.Algorithm import Algorithm
from core.graph_validator import validate_graph
from core.job import Job
from core.validator import validate_schedule
from tests.jobs_loader import load_any, generate_one_machine_dpc


# =============================================================================
# КОНФИГУРАЦИЯ ТЕСТИРОВАНИЯ (изменяйте здесь)
# =============================================================================

class TestConfig:
    """Централизованная конфигурация тестирования."""

    # --- Выбор алгоритма ---
    # Доступные варианты: BalasBaBDPC, LTH, MLTH, ILTF, BestOfHeuristics, LDepth, RTE
    ALGO_CLASS: Type[Algorithm] = BalasBaBDPC

    # --- Путь к тестовому файлу ---
    # Можно указать конкретный файл или директорию
    TEST_FILE = "tests/data/100_aug_l100/rand0004.astg"
    # TEST_FILE = "tests/data/100_aug_l75/rand0050.astg"
    # TEST_FILE = "tests/data/100/rand0001.stg"
    USE_RANDOM_TEST = False
    INIT_SEED = 1

    # --- Режим тестирования ---
    # "single" - один файл, "batch" - все файлы в директории
    TEST_MODE = "single"  # "single" или "batch"

    # Для пакетного режима: сколько файлов обработать (0 = все)
    BATCH_LIMIT = 5

    # --- Параметры алгоритма ---
    TIME_LIMIT = 30  # секунд

    # Особые параметры для некоторых алгоритмов
    ALGO_KWARGS = {
        BalasBaBDPC: {"time_limit": TIME_LIMIT},
        # RTE: {"heuristic_class": LTH},
    }

    # --- Параметры валидации ---
    VALIDATE = not USE_RANDOM_TEST
    STRICT_IDLE_CHECK = False  # True = все простои считаются ошибками
    VERBOSE_VALIDATION = True  # Показывать детальный отчёт валидации

    # --- Профилирование ---
    SAVE_PROFILE = False
    PRINT_PROFILE = ALGO_CLASS is BalasBaBDPC
    PROFILE_TOP_N = 20  # Сколько строк профиля показывать

    # --- Сохранение результатов ---
    SAVE_RESULTS = False
    OUTPUT_DIR = Path("test_results")

    # --- Дополнительные настройки ---
    VISUALIZE_SMALL = True  # Визуализировать расписания для N <= 20
    SHOW_DETAILED_SCHEDULE = False  # Показывать полное расписание в консоли

    @classmethod
    def ensure_output_dir(cls) -> None:
        """Создает выходную директорию если её нет."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_algo_kwargs(cls) -> dict:
        """Возвращает специфичные параметры для выбранного алгоритма."""
        algo_name = cls.ALGO_CLASS.__name__
        return cls.ALGO_KWARGS.get(algo_name, {})

    @classmethod
    def print_config(cls) -> None:
        """Выводит текущую конфигурацию."""
        print(f"\n{'=' * 70}")
        print(f"📋 КОНФИГУРАЦИЯ ТЕСТИРОВАНИЯ")
        print(f"{'=' * 70}")
        print(f"   Алгоритм:           {cls.ALGO_CLASS.__name__}")
        print(f"   Тестовый файл:      {cls.TEST_FILE}")
        print(f"   Режим:              {cls.TEST_MODE}")
        print(f"   Лимит времени:      {cls.TIME_LIMIT} сек")
        print(f"   Валидация:          {'Да' if cls.VALIDATE else 'Нет'}")
        print(f"   Строгая проверка:   {'Да' if cls.STRICT_IDLE_CHECK else 'Нет (MLTH/ILTF)'}")
        print(f"   Профилирование:     {'Да' if cls.SAVE_PROFILE else 'Нет'}")
        if cls.TEST_MODE == "batch":
            print(f"   Лимит файлов:       {cls.BATCH_LIMIT if cls.BATCH_LIMIT > 0 else 'все'}")
        print(f"{'=' * 70}\n")


# =============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
# =============================================================================

def load_test_data(file_path: str) -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """
    Загружает тестовые данные из файла.

    Args:
        file_path: Путь к файлу с данными

    Returns:
        Кортеж (список заданий, словарь ограничений предшествования)
    """
    if TestConfig.USE_RANDOM_TEST:
        seed = TestConfig.INIT_SEED
        while True:
            jobs_list, precedence = generate_one_machine_dpc(100, d_max=50, r_max=500, q_max=500, p=0.002, l_max=100,
                                                             seed=seed)
            ok, errors = validate_graph(jobs_list, precedence)
            if ok:
                break

            seed += 1

        print(f'seed: {seed}')
    else:
        print(f"📂 Загрузка: {os.path.basename(file_path)}")
        jobs_list, precedence = load_any(file_path)

    print(f"   📊 Работ: {len(jobs_list)}")
    print(f"   📊 DPC:   {len(precedence)}")

    return jobs_list, precedence


def get_test_files() -> List[str]:
    """
    Возвращает список файлов для тестирования в зависимости от режима.
    """
    if TestConfig.TEST_MODE == "single":
        return [TestConfig.TEST_FILE]

    # Пакетный режим: собираем все .astg и .stg файлы из директории
    test_dir = TestConfig.TEST_FILE
    if os.path.isfile(test_dir):
        test_dir = os.path.dirname(test_dir)

    files = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(('.astg', '.stg'))
    ])

    if TestConfig.BATCH_LIMIT > 0:
        files = files[:TestConfig.BATCH_LIMIT]

    return files


# =============================================================================
# ЗАПУСК АЛГОРИТМА
# =============================================================================

def run_algorithm(
        jobs_list: List[Job],
        precedence: Dict[Tuple[int, int], float],
        profile: bool = True
) -> Tuple[Optional[List[int]], float, Dict[str, Any], Optional[pstats.Stats]]:
    algo_name = TestConfig.ALGO_CLASS.__name__
    print(f"\n🚀 Запуск: {algo_name}")
    print(f"   ⏱️  Лимит: {TestConfig.TIME_LIMIT} сек")

    # Создаём экземпляр алгоритма с учётом особых параметров
    algo_kwargs = TestConfig.get_algo_kwargs()
    algo = TestConfig.ALGO_CLASS(jobs_list, precedence, **algo_kwargs)

    profiler = None
    profile_stats = None

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    schedule, C_max, stats = algo.solve()

    # Визуализация для маленьких задач
    if TestConfig.VISUALIZE_SMALL and len(jobs_list) <= 20 and schedule:
        algo.visualize(schedule, C_max, stats)

    if profile and profiler:
        profiler.disable()
        profile_stats = pstats.Stats(profiler)

    return schedule, C_max, stats, profile_stats


# =============================================================================
# ВАЛИДАЦИЯ
# =============================================================================

def validate_solution(
        schedule: List[int],
        jobs_dict: Dict[int, Job],
        precedence: Dict[Tuple[int, int], float],
        sigma: Optional[Dict[int, set]] = None,
        verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Валидирует найденное решение.

    Args:
        schedule: Расписание (порядок работ)
        jobs_dict: Словарь работ
        precedence: Ограничения DPC
        sigma: Дополнительные отношения предшествования
        verbose: Подробный вывод

    Returns:
        Кортеж (валидно ли, детали проверки)
    """
    print(f"\n🔍 Валидация...")

    is_valid, result = validate_schedule(
        schedule,
        jobs_dict,
        precedence,
        sigma=sigma,
        verbose=verbose,
        strict_idle_check=TestConfig.STRICT_IDLE_CHECK
    )

    if is_valid:
        print(f"   ✅ Расписание КОРРЕКТНО")
    else:
        print(f"   ❌ Найдены ошибки ({len(result.get('errors', []))})")

    return is_valid, result


# =============================================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# =============================================================================

def print_results_summary(
        C_max_algo: float,
        stats: Dict[str, Any],
        schedule: List[int],
        file_name: str = "",
        is_valid = True
) -> None:
    """
    Выводит сводку результатов работы алгоритма.

    Args:
        C_max_algo: Значение целевой функции
        stats: Статистика алгоритма
        schedule: Расписание
        file_name: Имя тестового файла
    """
    print(f"\n{'=' * 70}")
    print(f"📊 РЕЗУЛЬТАТЫ: {TestConfig.ALGO_CLASS.__name__}")
    if file_name:
        print(f"   Файл: {os.path.basename(file_name)}")
    print(f"{'=' * 70}")
    for (stat_name, value) in stats.items():
        print(f"{stat_name}: {value}")

    # Основные метрики
    print(f"\n📈 Основные метрики:")
    print(f"   • C_max:              {C_max_algo:.2f}")
    print(f"   • Длина расписания:   {len(schedule)}")
    print(f"   • Время выполнения:   {stats.get('execution_time', 'N/A'):.4f} сек")

    # Специфичная для алгоритма статистика
    if 'nodes_explored' in stats:
        print(f"\n📊 Ветвления и границы:")
        print(f"   • Узлов исследовано:  {stats.get('nodes_explored', 0)}")
        print(f"   • Strong branches:    {stats.get('strong_branches', 0)}")
        print(f"   • Weak branches:      {stats.get('weak_branches', 0)}")
        print(f"   • Pruned by bound:    {stats.get('pruned_by_bound', 0)}")
        print(f"   • Pruned by test:     {stats.get('pruned_by_test', 0)}")
        print(f"   • Timed out:          {stats.get('timed_out', False)}")
        print(f"   • Optimal:            {stats.get('optimal', False)}")
        print(f"   • Improvement:        {stats.get('improvement', 0):.2f}")

    if 'iterations' in stats:
        print(f"\n📊 Итерации:")
        print(f"   • Итераций:           {stats.get('iterations', 0)}")

    if 'idle_times' in stats:
        print(f"   • Простоев:           {stats.get('idle_times', 0)}")

    if 'LB' in stats:
        print(f"   • Нижняя граница:     {stats.get('LB', 0):.2f}")
        print(f"   • Gap:                {stats.get('gap', 0):.2f}%")

    # Показываем расписание если нужно
    if TestConfig.SHOW_DETAILED_SCHEDULE:
        # Загружаем данные для вычисления реальных времён
        jobs_list, precedence = load_any(file_name)
        jobs_dict = {job.id: job for job in jobs_list}

        # Вычисляем времена с учётом DPC
        scheduler = LTH(jobs_list,precedence)

        # Вычисляем makespan и получаем времена
        _, start_times = scheduler.calculate_makespan(schedule)
        n = 15
        if not is_valid:
            n=len(schedule)
        print(f"\n📋 Расписание ({n} работ):")
        print(f"   {'Поз.':4} {'Job':5} {'Start':>10} {'End':>10} {'Q':>6}")
        print(f"   {'-' * 40}")

        for idx, job_id in enumerate(schedule[:n], 1):
            job = jobs_dict[job_id]
            start = start_times.get(job_id, 0.0)
            end = start + job.d_i
            print(f"   {idx:4} {job_id:5} {start:10.1f} {end:10.1f} {job.q_i:6.1f}")

        if len(schedule) > 15:
            print(f"   ... и ещё {len(schedule) - n} работ")


def print_profile_stats(profile_stats: pstats.Stats, top_n: int = 20) -> None:
    """
    Выводит статистику профилирования.

    Args:
        profile_stats: Статистика профилирования
        top_n: Количество показываемых строк
    """
    print(f"\n{'=' * 70}")
    print(f"📈 ПРОФИЛИРОВАНИЕ (top {top_n})")
    print(f"{'=' * 70}")

    stream = io.StringIO()
    profile_stats.sort_stats('cumulative')
    profile_stats.print_stats(top_n)

    lines = stream.getvalue().split('\n')
    for line in lines[:top_n + 10]:
        if line.strip():
            print(f"   {line}")


def save_results_to_file(
        schedule: List[int],
        C_max: float,
        stats: Dict[str, Any],
        file_name: str = "",
        sigma: Optional[Dict[int, set]] = None
) -> None:
    """
    Сохраняет результаты в файл.

    Args:
        schedule: Расписание
        C_max: Значение целевой функции
        stats: Статистика
        file_name: Имя тестового файла
        sigma: Дополнительные отношения
    """
    TestConfig.ensure_output_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_name = TestConfig.ALGO_CLASS.__name__
    output_file = TestConfig.OUTPUT_DIR / f"{algo_name}_{timestamp}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"РЕЗУЛЬТАТЫ: {algo_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Тестовый файл: {file_name}\n")
        f.write(f"Лимит времени: {TestConfig.TIME_LIMIT} сек\n\n")

        f.write("ОСНОВНЫЕ МЕТРИКИ:\n")
        f.write(f"  C_max: {C_max:.2f}\n")
        f.write(f"  Длина расписания: {len(schedule)}\n")
        f.write(f"  Время выполнения: {stats.get('execution_time', 'N/A'):.4f} сек\n")

        if 'nodes_explored' in stats:
            f.write(f"  Узлов исследовано: {stats.get('nodes_explored', 'N/A')}\n")
            f.write(f"  Strong branches: {stats.get('strong_branches', 0)}\n")
            f.write(f"  Weak branches: {stats.get('weak_branches', 0)}\n")

        f.write(f"\nРАСПИСАНИЕ:\n  {schedule}\n")

        if sigma:
            total_relations = sum(len(v) for v in sigma.values())
            f.write(f"\nSIGMA ОТНОШЕНИЯ: {total_relations}\n")

    print(f"\n💾 Результаты сохранены: {output_file}")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def test_single_file(file_path: str) -> Tuple[Optional[List[int]], float, Dict[str, Any]]:
    """
    Тестирует алгоритм на одном файле.

    Args:
        file_path: Путь к файлу с данными

    Returns:
        Кортеж (расписание, C_max, статистика)
    """
    # 1. Загрузка данных
    jobs_list, precedence = load_test_data(file_path)

    if not jobs_list:
        print("❌ Не удалось загрузить данные")
        return None, float('inf'), {}

    jobs_dict = {job.id: job for job in jobs_list}

    # 2. Запуск алгоритма
    schedule, C_max, stats, profile_stats = run_algorithm(
        jobs_list,
        precedence,
        profile=TestConfig.SAVE_PROFILE or TestConfig.PRINT_PROFILE
    )

    if schedule is None:
        print("\n❌ Алгоритм не нашёл решения!")
        return None, float('inf'), stats

    # 3. Вывод результатов

    # 4. Валидация
    is_valid = True

    if TestConfig.VALIDATE:
        is_valid, _ = validate_solution(
            schedule,
            jobs_dict,
            precedence,
            sigma=None,
            verbose=TestConfig.VERBOSE_VALIDATION
        )

    print_results_summary(C_max, stats, schedule, file_path, is_valid)

    # 5. Сохранение результатов
    if TestConfig.SAVE_RESULTS:
        save_results_to_file(schedule, C_max, stats, file_path)

    # 6. Профилирование
    if TestConfig.PRINT_PROFILE and profile_stats:
        print_profile_stats(profile_stats, TestConfig.PROFILE_TOP_N)

    if profile_stats and TestConfig.SAVE_PROFILE:
        TestConfig.ensure_output_dir()
        profile_file = TestConfig.OUTPUT_DIR / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stats"
        print(f"\n💾 Профиль сохранён: {profile_file}")

    return schedule, C_max, stats


def test_batch_files(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Тестирует алгоритм на нескольких файлах.

    Args:
        file_paths: Список путей к файлам

    Returns:
        Словарь с результатами для каждого файла
    """
    results = {}
    total_files = len(file_paths)

    print(f"\n📦 Пакетный режим: {total_files} файлов")

    for i, file_path in enumerate(file_paths, 1):
        print(f"\n{'=' * 70}")
        print(f"📁 Файл {i}/{total_files}: {os.path.basename(file_path)}")
        print(f"{'=' * 70}")

        schedule, C_max, stats = test_single_file(file_path)
        results[file_path] = {
            'C_max': C_max,
            'schedule': schedule,
            'stats': stats
        }

    # Сводка по всем файлам
    print(f"\n{'=' * 70}")
    print(f"📊 СВОДКА ПО ВСЕМ ФАЙЛАМ")
    print(f"{'=' * 70}")

    successful = [(path, res) for path, res in results.items() if res['schedule'] is not None]

    if successful:
        C_max_values = [res['C_max'] for _, res in successful]
        times = [res['stats'].get('execution_time', 0) for _, res in successful]

        print(f"\n   Успешно решено: {len(successful)}/{total_files}")
        print(f"   Средний C_max:   {sum(C_max_values) / len(C_max_values):.2f}")
        print(f"   Мин. C_max:      {min(C_max_values):.2f}")
        print(f"   Макс. C_max:     {max(C_max_values):.2f}")
        print(f"   Среднее время:   {sum(times) / len(times):.4f} сек")
        print(f"   Общее время:     {sum(times):.4f} сек")

    return results


def main() -> None:
    """Главная точка входа."""
    print(f"\n{'=' * 70}")
    print(f"🧪 УНИВЕРСАЛЬНЫЙ ТЕСТИРОВЩИК АЛГОРИТМОВ")
    print(f"{'=' * 70}")

    TestConfig.print_config()

    try:
        if TestConfig.TEST_MODE == "single":
            schedule, C_max, stats = test_single_file(TestConfig.TEST_FILE)
        else:
            files = get_test_files()
            test_batch_files(files)

        print(f"\n{'=' * 70}")
        print(f"✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except FileNotFoundError as e:
        print(f"\n❌ Файл не найден: {e}")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    main()