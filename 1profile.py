import cProfile
import pstats
from typing import List, Dict, Tuple

from algorithms.BaBBalasDPC import BalasBaBDPC
from core.job import Job
from core.validator import validate_schedule, compute_makespan
from tests.astg_jobs_loader import load_astg_jobs


def main():
    # Загружаем файл
    file_path = "tests/data/100_aug_l100/rand0177.astg"
    print(f"Загрузка файла: {file_path}")
    raw_jobs, raw_edges = load_astg_jobs(file_path)

    print(f"Количество работ: {len(raw_jobs)}")
    print(f"Количество DPC: {len(raw_edges)}")

    # Преобразуем в объекты Job
    jobs_dict = {}
    for job_id, (t, r, q) in raw_jobs.items():
        jobs_dict[job_id] = Job(id=job_id, d_i=t, r_i=r, q_i=q)

    jobs_list = list(jobs_dict.values())
    precedence = {(i, j): lij for (i, j), lij in raw_edges.items()}

    # Запускаем алгоритм
    algo = BalasBaBDPC(jobs_list, precedence, time_limit=300)

    profiler = cProfile.Profile()
    profiler.enable()
    schedule, C_max_algo, stats = algo.solve()
    profiler.disable()

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ АЛГОРИТМА")
    print(f"{'='*60}")
    print(f"C_max (алгоритм): {C_max_algo:.2f}")
    print(f"Длина расписания: {len(schedule)}")
    print(f"Узлов исследовано: {stats.get('nodes_explored', 'N/A')}")
    print(f"Время выполнения: {stats.get('execution_time', 'N/A')}")

    # Получаем sigma из алгоритма
    sigma = getattr(algo, 'best_sigma', None)
    if sigma:
        total_relations = sum(len(v) for v in sigma.values())
        print(f"Сохранено sigma-отношений: {total_relations}")
    else:
        print("Sigma не сохранена")

    # Проверяем C_max через базовый метод с sigma
    C_max_scheduler, _ = algo.calculate_makespan(schedule, sigma)

    print(f"\n{'='*60}")
    print(f"СРАВНЕНИЕ C_max")
    print(f"{'='*60}")
    print(f"C_max (алгоритм):     {C_max_algo:.2f}")
    print(f"C_max (scheduler):     {C_max_scheduler:.2f}")

    if abs(C_max_algo - C_max_scheduler) < 1e-6:
        print(f"✅ C_max СОВПАДАЕТ!")
    else:
        print(f"⚠️  Расхождение: {abs(C_max_algo - C_max_scheduler):.2f}")

    # Валидация с sigma
    print(f"\n{'='*60}")
    print(f"ВАЛИДАЦИЯ РАСПИСАНИЯ")
    print(f"{'='*60}")

    is_valid, result = validate_schedule(
        schedule,
        jobs_dict,
        raw_edges,
        sigma=sigma,
        verbose=True
    )

    if is_valid:
        print(f"\n✅ Расписание ПОЛНОСТЬЮ КОРРЕКТНО")
        print(f"   C_max (валидатор): {result['details']['C_max']:.2f}")
    else:
        print(f"\n❌ Расписание НЕКОРРЕКТНО")
        for err in result['errors'][:5]:
            print(f"   - {err}")

    return schedule, C_max_algo, stats


if __name__ == "__main__":
    try:
        schedule, C_max, stats = main()
        print(f"\n{'='*60}")
        print(f"ГОТОВО")
        print(f"{'='*60}")
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()