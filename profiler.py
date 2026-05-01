#!/usr/bin/env python3
"""
Профилирование и тестирование алгоритма BalasBaBDPC для задачи 1|r_j, q_j, DPC|C_max
"""

import cProfile
import pstats
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any

from algorithms.BaBBalasDPC import BalasBaBDPC
from algorithms.BestOfHeuristics import BestOfHeuristics
from algorithms.iltf import ILTF
from algorithms.l_depth import LDepth
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from algorithms.rte import RTE
from core.job import Job
from core.validator import validate_schedule
from tests.astg_jobs_loader import load_astg_jobs


class TestConfig:
    """Конфигурация тестирования."""

    # Путь к тестовому файлу
    TEST_FILE = "tests/data/100_aug_l100/rand0157.astg"

    # Лимит времени для алгоритма (секунды)
    TIME_LIMIT = 30

    # Сохранять ли профилирование
    SAVE_PROFILE = True

    # Выводить ли подробную информацию
    VERBOSE = False

    # Выходная директория для результатов
    OUTPUT_DIR = Path("test_results")

    @classmethod
    def ensure_output_dir(cls) -> None:
        """Создает выходную директорию если её нет."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)


def load_test_data(file_path: str) -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """
    Загружает тестовые данные из файла.

    Args:
        file_path: Путь к файлу с данными

    Returns:
        Кортеж (список заданий, словарь ограничений предшествования)
    """
    print(f"📂 Загрузка файла: {file_path}")
    raw_jobs, raw_edges = load_astg_jobs(file_path)

    print(f"   📊 Количество работ: {len(raw_jobs)}")
    print(f"   📊 Количество DPC: {len(raw_edges)}")

    # Преобразуем в объекты Job
    jobs_list = []
    for job_id, (t, r, q) in raw_jobs.items():
        jobs_list.append(Job(id=job_id, d_i=t, r_i=r, q_i=q))

    precedence = {(i, j): lij for (i, j), lij in raw_edges.items()}

    return jobs_list, precedence


def run_algorithm(jobs_list: List[Job],
                  precedence: Dict[Tuple[int, int], float],
                  time_limit: float,
                  profile: bool = True) -> Tuple[Optional[List[int]], float, Dict[str, Any], Optional[pstats.Stats]]:
    """
    Запускает алгоритм с опциональным профилированием.

    Returns:
        Кортеж (расписание, C_max, статистика, профиль)
    """
    print(f"\n🚀 Запуск алгоритма BalasBaBDPC...")
    print(f"   ⏱️  Лимит времени: {time_limit} сек")

    algo = RTE(jobs_list, precedence, heuristic_class=ILTF)

    profiler = None
    profile_stats = None

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    schedule, C_max, stats = algo.solve()
    if len(jobs_list)<=20:
        algo.visualize(schedule, C_max, stats)

    if profile and profiler:
        profiler.disable()
        profile_stats = pstats.Stats(profiler)

    return schedule, C_max, stats, profile_stats


def validate_solution(schedule: List[int],
                      jobs_dict: Dict[int, Job],
                      precedence: Dict[Tuple[int, int], float],
                      sigma: Optional[Dict[int, set]] = None,
                      verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Валидирует найденное решение.

    Returns:
        Кортеж (валидно, детали валидации)
    """
    print(f"\n🔍 Валидация расписания...")

    is_valid, result = validate_schedule(
        schedule,
        jobs_dict,
        precedence,
        sigma=sigma,
        verbose=False
    )

    if is_valid:
        print(f"   ✅ Расписание КОРРЕКТНО")
    else:
        print(f"   ❌ Расписание НЕКОРРЕКТНО")
        if 'errors' in result:
            for err in result['errors'][:5]:
                print(f"      - {err}")

    return is_valid, result


def print_results_summary(C_max_algo: float,
                          C_max_scheduler: float,
                          stats: Dict[str, Any],
                          schedule: List[int],
                          sigma: Optional[Dict[int, set]] = None) -> None:
    """Выводит сводку результатов."""
    for (stat_name, value) in stats.items():
        print(f"{stat_name}: {value}")
    # print(stats)
    print(f"\n{'=' * 70}")
    print(f"📊 РЕЗУЛЬТАТЫ АЛГОРИТМА")
    print(f"{'=' * 70}")

    # Основные метрики
    print(f"\n📈 Основные метрики:")
    print(f"   • C_max:           {C_max_algo:.2f}")
    print(f"   • Длина расписания: {len(schedule)}")
    print(f"   • Узлов исследовано: {stats.get('nodes_explored', 'N/A')}")
    print(f"   • Время выполнения: {stats.get('execution_time', 'N/A'):.4f} сек")

    # Дополнительная статистика
    print(f"\n📊 Детали ветвления:")
    print(f"   • Strong branches:  {stats.get('strong_branches', 0)}")
    print(f"   • Weak branches:    {stats.get('weak_branches', 0)}")
    print(f"   • Pruned by bound:  {stats.get('pruned_by_bound', 0)}")
    print(f"   • Pruned by test:   {stats.get('pruned_by_test', 0)}")

    # Информация о решении
    print(f"\n🎯 Решение:")
    # print(f"   • Initial makespan: {stats.get('initial_makespan', 'N/A'):.2f}")
    print(f"   • Improvement:      {stats.get('improvement', 0):.2f}")
    print(f"   • Timed out:        {stats.get('timed_out', False)}")
    print(f"   • Optimal:          {stats.get('optimal', False)}")

    # Sigma информация
    if sigma:
        total_relations = sum(len(v) for v in sigma.values())
        print(f"   • Sigma relations:  {total_relations}")

    # Сравнение C_max
    print(f"\n✅ Сравнение C_max:")
    print(f"   • Алгоритм:   {C_max_algo:.2f}")
    print(f"   • Scheduler:  {C_max_scheduler:.2f}")

    if abs(C_max_algo - C_max_scheduler) < 1e-6:
        print(f"   • Статус:     ✅ СОВПАДАЕТ")
    else:
        print(f"   • Статус:     ⚠️  РАСХОДИТСЯ (разница: {abs(C_max_algo - C_max_scheduler):.2f})")


def save_results_to_file(file_path: Path,
                         schedule: List[int],
                         C_max: float,
                         stats: Dict[str, Any],
                         sigma: Optional[Dict[int, set]] = None) -> None:
    """Сохраняет результаты в файл."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = TestConfig.OUTPUT_DIR / f"results_{timestamp}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("РЕЗУЛЬТАТЫ РАБОТЫ АЛГОРИТМА BalasBaBDPC\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Тестовый файл: {TestConfig.TEST_FILE}\n\n")

        f.write("ОСНОВНЫЕ МЕТРИКИ:\n")
        f.write(f"  C_max: {C_max:.2f}\n")
        f.write(f"  Длина расписания: {len(schedule)}\n")
        f.write(f"  Узлов исследовано: {stats.get('nodes_explored', 'N/A')}\n")
        f.write(f"  Время выполнения: {stats.get('execution_time', 'N/A'):.4f} сек\n\n")

        f.write("ДЕТАЛИ ВЕТВЛЕНИЯ:\n")
        f.write(f"  Strong branches: {stats.get('strong_branches', 0)}\n")
        f.write(f"  Weak branches: {stats.get('weak_branches', 0)}\n")
        f.write(f"  Pruned by bound: {stats.get('pruned_by_bound', 0)}\n")
        f.write(f"  Pruned by test: {stats.get('pruned_by_test', 0)}\n\n")

        f.write("РАСПИСАНИЕ:\n")
        f.write(f"  {schedule}\n\n")

        if sigma:
            total_relations = sum(len(v) for v in sigma.values())
            f.write(f"SIGMA ОТНОШЕНИЯ:\n")
            f.write(f"  Всего отношений: {total_relations}\n")

    print(f"\n💾 Результаты сохранены в: {output_file}")


def print_profile_stats(profile_stats: pstats.Stats, top_n: int = 20) -> None:
    """Выводит статистику профилирования."""

    print(f"\n{'=' * 70}")
    print(f"📈 ПРОФИЛИРОВАНИЕ (top {top_n})")
    print(f"{'=' * 70}")

    stream = io.StringIO()
    profile_stats.sort_stats('cumulative')
    profile_stats.print_stats(top_n)

    # Выводим только самые важные строки
    lines = stream.getvalue().split('\n')
    for line in lines[:top_n + 10]:  # Заголовок + top_n строк
        if line.strip():
            print(f"   {line}")


def main() -> Tuple[Optional[List[int]], float, Dict[str, Any]]:
    """Главная функция."""

    print(f"\n{'=' * 70}")
    print(f"🧪 ТЕСТИРОВАНИЕ АЛГОРИТМА BalasBaBDPC")
    print(f"{'=' * 70}")

    # 1. Загрузка данных
    jobs_list, precedence = load_test_data(TestConfig.TEST_FILE)

    # Создаем словарь работ для валидации
    jobs_dict = {job.id: job for job in jobs_list}

    # 2. Запуск алгоритма
    schedule, C_max_algo, stats, profile_stats = run_algorithm(
        jobs_list,
        precedence,
        TestConfig.TIME_LIMIT,
        profile=TestConfig.SAVE_PROFILE
    )

    if schedule is None:
        print("\n❌ Алгоритм не нашел решения!")
        return None, float('inf'), stats

    # 3. Получение sigma из алгоритма
    sigma = getattr(BalasBaBDPC, 'best_sigma', None)
    if hasattr(BalasBaBDPC, 'best_sigma'):
        # Получаем sigma из последнего экземпляра
        sigma = None

    # 4. Проверка C_max через scheduler
    # Создаем временный экземпляр для расчета
    temp_algo = BalasBaBDPC(jobs_list, precedence, time_limit=1)
    C_max_scheduler, _ = temp_algo.calculate_makespan(schedule, sigma)

    # 5. Вывод результатов
    print_results_summary(C_max_algo, C_max_scheduler, stats, schedule, sigma)

    # 6. Валидация
    is_valid, _ = validate_solution(
        schedule,
        jobs_dict,
        precedence,
        sigma,
        verbose=False
    )

    # 7. Сохранение результатов
    if TestConfig.SAVE_PROFILE:
        TestConfig.ensure_output_dir()
        save_results_to_file(TestConfig.OUTPUT_DIR / "latest.txt",
                             schedule, C_max_algo, stats, sigma)

    # 8. Вывод профилирования
    if profile_stats and TestConfig.SAVE_PROFILE:
        print_profile_stats(profile_stats)
        # Сохраняем профиль
        profile_file = TestConfig.OUTPUT_DIR / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stats"
        # profile_stats.dump_stats(str(profile_file))
        print(f"\n💾 Профиль сохранен в: {profile_file}")

    return schedule, C_max_algo, stats


if __name__ == "__main__":
    try:
        schedule, C_max, stats = main()

        print(f"\n{'=' * 70}")
        print(f"✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print(f"{'=' * 70}")

        if schedule:
            print(f"🎯 Финальный C_max: {C_max:.2f}")
            print(f"📊 Исследовано узлов: {stats.get('nodes_explored', 'N/A')}")
            print(f"⏱️  Общее время: {stats.get('execution_time', 'N/A'):.4f} сек")

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()