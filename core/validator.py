"""
Валидатор для проверки корректности расписаний задачи 1|r_j, q_j, DPC|C_max.
Проверяет:
- Полноту расписания
- Release times (r_i)
- Отсутствие пересечений на машине
- Delayed Precedence Constraints (DPC)
- Добавленные в процессе B&B отношения предшествования (sigma)
- Корректность вычисления C_max
"""

from typing import List, Dict, Tuple, Set, Optional
from .job import Job


def validate_schedule(
        schedule: List[int],
        jobs: Dict[int, Job],
        l_matrix: Optional[Dict[int, Dict[int, float]]] = None,
        sigma: Optional[Dict[int, Set[int]]] = None,
        verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Проверяет корректность расписания для задачи 1|r_j, q_j, DPC|C_max.

    Args:
        schedule: Список ID работ в порядке выполнения
        jobs: Словарь {job_id: Job}
        l_matrix: Матрица задержек DPC {i: {j: L(i,j)}}
        sigma: Отношения предшествования {i: set(j)} означает i -> j
        verbose: Печатать ли подробный отчёт

    Returns:
        (is_valid, details): Кортеж (валидно ли расписание, словарь с деталями)
    """
    errors = []
    warnings = []
    details = {}

    if l_matrix is None:
        l_matrix = {}
    if sigma is None:
        sigma = {}

    # 1. Проверка наличия всех работ
    job_ids = set(jobs.keys())
    schedule_set = set(schedule)

    if len(schedule) != len(job_ids):
        errors.append(
            f"Размер расписания ({len(schedule)}) не совпадает с количеством работ ({len(job_ids)})"
        )

    missing = job_ids - schedule_set
    if missing:
        errors.append(f"Отсутствуют работы: {sorted(missing)}")

    duplicates = [j for j in schedule if schedule.count(j) > 1]
    if duplicates:
        errors.append(f"Дубликаты работ: {sorted(set(duplicates))}")

    extra = schedule_set - job_ids
    if extra:
        errors.append(f"Неизвестные работы: {sorted(extra)}")

    # Если есть критические ошибки, дальше можно не проверять
    if missing or duplicates or extra:
        if verbose:
            print("\n" + "=" * 60)
            print("❌ КРИТИЧЕСКИЕ ОШИБКИ В РАСПИСАНИИ")
            print("=" * 60)
            for err in errors:
                print(f"   - {err}")
        return False, {'errors': errors, 'warnings': warnings, 'details': details}

    # 2. Симуляция выполнения расписания
    start_times = {}
    completion_times = {}
    current_time = 0.0

    for idx, job_id in enumerate(schedule):
        job = jobs[job_id]

        # Начальное время старта: max(release_time, текущее время машины)
        start = max(job.r_i, current_time)

        # Учёт DPC из l_matrix (от уже запланированных работ)
        for prev_id, prev_start in start_times.items():
            lij = l_matrix.get(prev_id, {}).get(job_id, 0.0)
            if lij > 0:
                required_start = prev_start + lij
                if required_start > start:
                    start = required_start

        # Учёт sigma (добавленные отношения предшествования)
        # sigma[prev] содержит работы, которые должны идти ПОСЛЕ prev
        for prev_id, next_ids in sigma.items():
            if job_id in next_ids and prev_id in start_times:
                # prev_id -> job_id
                required_start = start_times[prev_id] + jobs[prev_id].d_i
                if required_start > start:
                    start = required_start

        # Проверка, что start >= r_j
        if start < job.r_i - 1e-6:
            errors.append(
                f"Работа {job_id}: start={start:.2f} < r_{job_id}={job.r_i:.2f}"
            )

        start_times[job_id] = start
        completion_times[job_id] = start + job.d_i
        current_time = start + job.d_i

    # 3. Проверка на пересечения (машина одна)
    sorted_by_start = sorted(start_times.items(), key=lambda x: x[1])
    for i in range(len(sorted_by_start) - 1):
        job1, start1 = sorted_by_start[i]
        job2, start2 = sorted_by_start[i + 1]
        end1 = completion_times[job1]

        if end1 > start2 + 1e-6:
            errors.append(
                f"Пересечение работ: {job1} заканчивается в {end1:.2f}, "
                f"{job2} начинается в {start2:.2f}"
            )

    # 4. Проверка DPC для всех пар
    for i in job_ids:
        for j in job_ids:
            lij = l_matrix.get(i, {}).get(j, 0.0)
            if lij > 0:
                pos_i = schedule.index(i)
                pos_j = schedule.index(j)

                if pos_i < pos_j:
                    # i идёт раньше j — проверяем задержку
                    actual_delay = start_times[j] - start_times[i]
                    if actual_delay < lij - 1e-6:
                        errors.append(
                            f"Нарушение DPC: {i} -> {j}, "
                            f"требуемая задержка={lij:.2f}, фактическая={actual_delay:.2f}"
                        )
                else:
                    # i идёт позже j — DPC не активно
                    warnings.append(
                        f"DPC {i} -> {j} не активно (i идёт после j в расписании)"
                    )

    # 5. Проверка sigma (добавленных отношений)
    for i, next_ids in sigma.items():
        if i not in schedule:
            continue
        pos_i = schedule.index(i)
        for j in next_ids:
            if j not in schedule:
                continue
            pos_j = schedule.index(j)
            if pos_i > pos_j:
                errors.append(
                    f"Нарушение sigma: {i} -> {j}, но {i} идёт после {j} в расписании"
                )

    # 6. Вычисление C_max
    C_max = 0.0
    critical_jobs = []
    for job_id in schedule:
        delivery_end = completion_times[job_id] + jobs[job_id].q_i
        if delivery_end > C_max + 1e-6:
            C_max = delivery_end
            critical_jobs = [job_id]
        elif abs(delivery_end - C_max) < 1e-6:
            critical_jobs.append(job_id)

    details['start_times'] = start_times
    details['completion_times'] = completion_times
    details['C_max'] = C_max
    details['critical_path_jobs'] = critical_jobs

    # 7. Дополнительные проверки
    total_processing = sum(j.d_i for j in jobs.values())
    makespan_time = max(completion_times.values()) if completion_times else 0
    total_idle = makespan_time - total_processing
    details['total_processing'] = total_processing
    details['total_idle'] = total_idle
    details['machine_utilization'] = (
        total_processing / makespan_time if makespan_time > 0 else 0
    )

    is_valid = len(errors) == 0

    if verbose:
        print("\n" + "=" * 60)
        print("ПРОВЕРКА РАСПИСАНИЯ")
        print("=" * 60)

        if is_valid:
            print("✅ РАСПИСАНИЕ КОРРЕКТНО")
        else:
            print("❌ РАСПИСАНИЕ НЕКОРРЕКТНО")

        print(f"\n📊 Статистика:")
        print(f"   C_max: {C_max:.2f}")
        print(f"   Количество работ: {len(schedule)}")
        print(f"   Общее время обработки: {total_processing:.2f}")
        print(f"   Время простоя: {total_idle:.2f}")
        print(f"   Утилизация машины: {details['machine_utilization'] * 100:.1f}%")
        print(f"   Критические работы: {critical_jobs}")

        if warnings:
            print(f"\n⚠️  Предупреждения ({len(warnings)}):")
            for w in warnings[:10]:
                print(f"   - {w}")
            if len(warnings) > 10:
                print(f"   ... и ещё {len(warnings) - 10}")

        if errors:
            print(f"\n❌ Ошибки ({len(errors)}):")
            for e in errors:
                print(f"   - {e}")

        # Детальное расписание
        print(f"\n📋 Детальное расписание:")
        print(
            f"{'Поз.':4} {'Job':5} {'r_i':>6} {'d_i':>6} {'q_i':>6} "
            f"{'Start':>8} {'End':>8} {'Delivery':>10}"
        )
        print("-" * 65)
        for idx, job_id in enumerate(schedule, 1):
            job = jobs[job_id]
            start = start_times[job_id]
            end = completion_times[job_id]
            delivery = end + job.q_i
            print(
                f"{idx:4} {job_id:5} {job.r_i:6.1f} {job.d_i:6.1f} {job.q_i:6.1f} "
                f"{start:8.1f} {end:8.1f} {delivery:10.1f}"
            )

    return is_valid, {'errors': errors, 'warnings': warnings, 'details': details}


def quick_validate(
        schedule: List[int],
        jobs: Dict[int, Job],
        l_matrix: Optional[Dict[int, Dict[int, float]]] = None,
        sigma: Optional[Dict[int, Set[int]]] = None
) -> float:
    """
    Быстрая проверка и вычисление C_max без подробного вывода.
    Возвращает C_max, если расписание корректно, иначе выбрасывает исключение.
    """
    is_valid, result = validate_schedule(
        schedule, jobs, l_matrix, sigma, verbose=False
    )
    if not is_valid:
        raise ValueError(f"Некорректное расписание: {result['errors']}")
    return result['details']['C_max']


def compute_makespan(
        schedule: List[int],
        jobs: Dict[int, Job],
        l_matrix: Optional[Dict[int, Dict[int, float]]] = None,
        sigma: Optional[Dict[int, Set[int]]] = None
) -> float:
    """
    Вычисляет C_max для расписания с учётом DPC и sigma.
    """
    if l_matrix is None:
        l_matrix = {}
    if sigma is None:
        sigma = {}

    start_times = {}
    current_time = 0.0

    for job_id in schedule:
        job = jobs[job_id]
        start = max(job.r_i, current_time)

        # DPC из l_matrix
        for prev_id, prev_start in start_times.items():
            lij = l_matrix.get(prev_id, {}).get(job_id, 0.0)
            if lij > 0:
                start = max(start, prev_start + lij)

        # Отношения из sigma
        for prev_id, next_ids in sigma.items():
            if job_id in next_ids and prev_id in start_times:
                start = max(start, start_times[prev_id] + jobs[prev_id].d_i)

        start_times[job_id] = start
        current_time = start + job.d_i

    return max(
        start_times[j] + jobs[j].d_i + jobs[j].q_i
        for j in schedule
    )