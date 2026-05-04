"""
Улучшенный валидатор для задачи 1|r_j, q_j, DPC|C_max.
Поддерживает проверку обоснованных простоев в MLTH и ILTF.
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
from .job import Job


def validate_schedule(
        schedule: List[int],
        jobs: Dict[int, Job],
        precedence: Optional[Dict[Tuple[int, int], float]] = None,
        sigma: Optional[Dict[int, Set[int]]] = None,
        verbose: bool = True,
        strict_idle_check: bool = False  # новый параметр
) -> Tuple[bool, Dict]:
    """
    Проверяет корректность расписания для задачи 1|r_j, q_j, DPC|C_max.

    Args:
        strict_idle_check: Если True, отмечает все простои как ошибки.
                          Если False (по умолчанию), простои считаются обоснованными,
                          если работа с более длинным хвостом становится доступной позже.
    """
    errors = []
    warnings = []
    details = {}

    # Нормализация входных данных
    if precedence is None:
        precedence = {}
    if sigma is None:
        sigma = {}

    # Преобразуем sigma для быстрого поиска
    sigma_predecessors = defaultdict(set)
    for i, next_ids in sigma.items():
        for j in next_ids:
            sigma_predecessors[j].add(i)

    # Матрица DPC
    l_matrix = defaultdict(lambda: defaultdict(float))
    for (i, j), lij in precedence.items():
        job_i = jobs[i]
        if lij < job_i.d_i:
            errors.append(f"Некорректная DPC: ({i},{j}) имеет задержку {lij} < d_{i}={job_i.d_i}")
        l_matrix[i][j] = max(lij, job_i.d_i)

    # ---------------------------------------------------------
    # 1. Проверка состава расписания
    # ---------------------------------------------------------
    job_ids = set(jobs.keys())
    schedule_set = set(schedule)

    if len(schedule) != len(job_ids):
        errors.append(f"Размер расписания ({len(schedule)}) не совпадает с количеством работ ({len(job_ids)})")

    missing = job_ids - schedule_set
    if missing:
        errors.append(f"В расписании отсутствуют работы: {sorted(missing)}")

    extra = schedule_set - job_ids
    if extra:
        errors.append(f"В расписании присутствуют неизвестные работы: {sorted(extra)}")

    duplicates = [j for j in schedule if schedule.count(j) > 1]
    if duplicates:
        errors.append(f"Обнаружены дубликаты работ: {sorted(set(duplicates))}")

    if missing or duplicates or extra:
        if verbose:
            _print_report(False, errors, warnings, details, schedule, jobs)
        return False, {'errors': errors, 'warnings': warnings, 'details': details}

    # ---------------------------------------------------------
    # 2. Симуляция выполнения
    # ---------------------------------------------------------
    start_times = {}
    completion_times = {}
    current_time = 0.0

    for idx, job_id in enumerate(schedule):
        job = jobs[job_id]
        start = max(job.r_i, current_time)

        # Учет DPC
        for prev_id, prev_start in start_times.items():
            lij = l_matrix[prev_id][job_id]
            if lij > 0:
                start = max(start, prev_start + lij)

        # Учет Sigma
        for pred_id in sigma_predecessors.get(job_id, set()):
            if pred_id in start_times:
                start = max(start, start_times[pred_id] + jobs[pred_id].d_i)

        # Проверка на нарушение времени поступления
        if start < job.r_i - 1e-6:
            errors.append(f"Работа {job_id}: start={start:.3f} < r_{job_id}={job.r_i:.3f}")

        # Проверка на простои (с учетом особенностей MLTH/ILTF)
        if start > current_time + 1e-6:
            available_in_idle = []

            for j in job_ids - set(schedule[:idx]):
                # Проверяем, все ли предшественники выполнены
                preds_done = True
                for (i, jj), _ in precedence.items():
                    if jj == j and i not in start_times:
                        preds_done = False
                        break
                for pred_id in sigma_predecessors.get(j, set()):
                    if pred_id not in start_times:
                        preds_done = False
                        break

                if not preds_done:
                    continue

                job_j = jobs[j]
                potential_start = max(job_j.r_i, current_time)

                # Учет DPC для потенциального старта
                for prev_id, prev_start in start_times.items():
                    l_pj = l_matrix[prev_id][j]
                    if l_pj > 0:
                        potential_start = max(potential_start, prev_start + l_pj)

                if potential_start < start - 1e-6:
                    available_in_idle.append((j, potential_start, job_j.q_i))

            if available_in_idle:
                if strict_idle_check:
                    # Строгий режим: все простои - ошибки
                    job_list = ", ".join([f"{j} (готово в {t:.1f})" for j, t, _ in available_in_idle[:3]])
                    errors.append(
                        f"Необоснованный простой перед работой {job_id} (старт в {start:.1f}). "
                        f"Процессор простаивает с {current_time:.1f}, хотя работа(ы): [{job_list}] "
                        f"могли начаться раньше."
                    )
                else:
                    # Анализируем, обоснован ли простой
                    current_job_q = job.q_i
                    best_available_q = max(q for _, _, q in available_in_idle)

                    if best_available_q >= current_job_q:
                        # Есть доступная работа с не меньшим хвостом - простой необоснован
                        job_list = ", ".join([f"{j} (q={q})" for j, _, q in available_in_idle[:3]])
                        warnings.append(
                            f"Возможно неоптимальный простой перед работой {job_id} "
                            f"(старт в {start:.1f}, q={current_job_q}). "
                            f"Доступны работы с не меньшим хвостом: [{job_list}]"
                        )

        start_times[job_id] = start
        completion_times[job_id] = start + job.d_i
        current_time = start + job.d_i

    # ---------------------------------------------------------
    # 3. Проверка на пересечения
    # ---------------------------------------------------------
    sorted_by_start = sorted(start_times.items(), key=lambda x: x[1])
    for i in range(len(sorted_by_start) - 1):
        job1, start1 = sorted_by_start[i]
        job2, start2 = sorted_by_start[i + 1]
        end1 = completion_times[job1]
        if end1 > start2 + 1e-6:
            errors.append(f"Пересечение работ: {job1} (конец в {end1:.3f}) и {job2} (старт в {start2:.3f})")

    # ---------------------------------------------------------
    # 4. Проверка DPC
    # ---------------------------------------------------------
    for (i, j), lij in precedence.items():
        pos_i = schedule.index(i)
        pos_j = schedule.index(j)

        if pos_i < pos_j:
            actual_delay = start_times[j] - start_times[i]
            if actual_delay < lij - 1e-6:
                errors.append(
                    f"Нарушение DPC: {i} -> {j}. "
                    f"Требуемая задержка L={lij:.3f}, фактическая={actual_delay:.3f}"
                )
        else:
            errors.append(
                f"Грубое нарушение DPC: {i} -> {j}. "
                f"Работа {i} идёт после {j}, что противоречит ограничению."
            )

    # ---------------------------------------------------------
    # 5. Проверка Sigma
    # ---------------------------------------------------------
    for i, next_ids in sigma.items():
        if i not in schedule_set:
            continue
        pos_i = schedule.index(i)
        for j in next_ids:
            if j not in schedule_set:
                continue
            pos_j = schedule.index(j)
            if pos_i > pos_j:
                errors.append(f"Нарушение Sigma: {i} -> {j}")

    # ---------------------------------------------------------
    # 6. Вычисление C_max
    # ---------------------------------------------------------
    calculated_C_max = 0.0
    critical_jobs = []
    for job_id in schedule:
        delivery_end = completion_times[job_id] + jobs[job_id].q_i
        if delivery_end > calculated_C_max + 1e-6:
            calculated_C_max = delivery_end
            critical_jobs = [job_id]
        elif abs(delivery_end - calculated_C_max) < 1e-6:
            critical_jobs.append(job_id)

    # ---------------------------------------------------------
    # 7. Статистика
    # ---------------------------------------------------------
    total_processing = sum(j.d_i for j in jobs.values())
    makespan_time = max(completion_times.values()) if completion_times else 0
    total_idle = makespan_time - total_processing

    details.update({
        'start_times': start_times,
        'completion_times': completion_times,
        'C_max': calculated_C_max,
        'critical_path_jobs': critical_jobs,
        'total_processing': total_processing,
        'total_idle': total_idle,
        'machine_utilization': (total_processing / makespan_time * 100) if makespan_time > 0 else 0.0
    })

    is_valid = len(errors) == 0

    if verbose:
        _print_report(is_valid, errors, warnings, details, schedule, jobs)

    return is_valid, {'errors': errors, 'warnings': warnings, 'details': details}


def _print_report(is_valid, errors, warnings, details, schedule, jobs):
    """Выводит подробный отчёт о проверке."""
    print("\n" + "=" * 70)
    print("ВАЛИДАЦИЯ РАСПИСАНИЯ")
    print("=" * 70)

    if is_valid:
        print("✅ РАСПИСАНИЕ КОРРЕКТНО")
    else:
        print("❌ РАСПИСАНИЕ НЕКОРРЕКТНО")

    C_max = details.get('C_max', 0.0)
    print(f"\n📊 КЛЮЧЕВАЯ СТАТИСТИКА:")
    print(f"   • C_max: {C_max:.1f}")
    print(f"   • Работ: {len(schedule)}")
    print(f"   • Время обработки: {details.get('total_processing', 0):.1f}")
    print(f"   • Время простоя: {details.get('total_idle', 0):.1f}")
    print(f"   • Утилизация: {details.get('machine_utilization', 0):.1f}%")
    print(f"   • Критические работы: {details.get('critical_path_jobs', [])}")

    if warnings:
        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"   - {w}")
        if len(warnings) > 5:
            print(f"   ... и ещё {len(warnings) - 5}")

    if errors:
        print(f"\n❌ ОШИБКИ ({len(errors)}):")
        for e in errors[:5]:
            print(f"   - {e}")
        if len(errors) > 5:
            print(f"   ... и ещё {len(errors) - 5}")

    # Краткое расписание
    print(f"\n📋 РАСПИСАНИЕ (первые 10 работ):")
    print(f"{'Поз.':4} {'Job':5} {'r_i':>6} {'d_i':>6} {'q_i':>6} {'Start':>8} {'End':>8}")
    print("-" * 55)
    for idx, job_id in enumerate(schedule[:10], 1):
        job = jobs[job_id]
        start = details['start_times'][job_id]
        end = details['completion_times'][job_id]
        marker = " *" if job_id in details['critical_path_jobs'] else ""
        print(f"{idx:4} {job_id:5} {job.r_i:6.1f} {job.d_i:6.1f} {job.q_i:6.1f} {start:8.1f} {end:8.1f}{marker}")
    if len(schedule) > 10:
        print(f"   ... и ещё {len(schedule) - 10} работ")