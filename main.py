"""
Примеры использования системы планирования
Модель: однопроцессорная 1|r_i, d_i, q_i, DPC|C_max
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple

from algorithms.BaBBalasDPC import BalasBaBDPC
from core.Algorithm import Algorithm
from algorithms.exact_bf import ExactBruteForce
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from algorithms.Balas_DPC import BalasDPC
from algorithms.iltf import ILTF
from core.job import Job
from core.utils import Visualizer
from evaluation.comparator import AlgorithmComparator


# ---------------------------------------------------------------------------
# Генераторы тестовых задач
# ---------------------------------------------------------------------------

def create_test_problem_1() -> Tuple[List[Job], Optional[Dict]]:
    """
    Тестовая задача 1 (6 заданий), без ограничений предшествования.
    Однопроцессорная DPC-модель.
    """
    jobs = [
        Job(id=1, r_i=0,  d_i=1, q_i=3),
        Job(id=2, r_i=2,  d_i=2, q_i=4),
        Job(id=3, r_i=3,  d_i=2, q_i=30),
        Job(id=4, r_i=12, d_i=1, q_i=10),
        Job(id=5, r_i=3,  d_i=1, q_i=17),
        Job(id=6, r_i=4,  d_i=1, q_i=2),
    ]
    return jobs, None


def create_test_problem_2() -> Tuple[List[Job], Optional[Dict]]:
    """
    Тестовая задача 2 (7 заданий), без ограничений предшествования.
    """
    jobs = [
        Job(id=1, r_i=2, d_i=4, q_i=3),
        Job(id=2, r_i=3, d_i=2, q_i=4),
        Job(id=3, r_i=1, d_i=5, q_i=8),
        Job(id=4, r_i=0, d_i=2, q_i=2),
        Job(id=5, r_i=3, d_i=1, q_i=1),
        Job(id=6, r_i=2, d_i=3, q_i=6),
        Job(id=7, r_i=1, d_i=1, q_i=3),
    ]
    return jobs, None


def create_test_problem_with_constraints() -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """
    Задача с ограничениями предшествования (delayed precedence constraints).
    """
    jobs = [
        Job(id=1, r_i=0, d_i=3, q_i=2),
        Job(id=2, r_i=1, d_i=4, q_i=1),
        Job(id=3, r_i=2, d_i=2, q_i=3),
        Job(id=4, r_i=0, d_i=5, q_i=4),
        Job(id=5, r_i=3, d_i=3, q_i=10),
    ]

    precedence_constraints: Dict[Tuple[int, int], float] = {
        (1, 2): 1,
        (1, 3): 2,
        (2, 4): 1,
        (3, 5): 2,
    }

    return jobs, precedence_constraints


def create_test_problem_for_iltf() -> Tuple[List[Job], Dict[Tuple[int, int], float]]:
    """
    Задача, где ILTF должен иметь преимущество перед LTH
    за счёт введения простоя ради важных работ.
    """
    jobs = [
        Job(id=1, r_i=0, d_i=2, q_i=3),
        Job(id=2, r_i=3, d_i=1, q_i=10),
        Job(id=3, r_i=1, d_i=2, q_i=2),
        Job(id=4, r_i=5, d_i=2, q_i=8),
        Job(id=5, r_i=2, d_i=1, q_i=4),
    ]

    precedence_constraints: Dict[Tuple[int, int], float] = {
        (1, 2): 3,
        (3, 4): 4,
        (2, 5): 2,
    }

    return jobs, precedence_constraints


# ---------------------------------------------------------------------------
# Вспомогательные функции выбора
# ---------------------------------------------------------------------------

def _select_problem() -> Tuple[List[Job], Optional[Dict]]:
    Visualizer.print_header("СИСТЕМА ПЛАНИРОВАНИЯ 1|r_i, d_i, q_i, DPC|C_max")

    print("Выберите тестовую задачу:")
    print("1. 6 заданий (простая, без ограничений)")
    print("2. 7 заданий (средняя, без ограничений)")
    print("3. С ограничениями предшествования")
    print("4. Специальная задача для ILTF")

    choice = input("\nВаш выбор (1-4): ").strip()

    if choice == "1":
        return create_test_problem_1()
    elif choice == "2":
        return create_test_problem_2()
    elif choice == "3":
        return create_test_problem_with_constraints()
    elif choice == "4":
        return create_test_problem_for_iltf()
    else:
        Visualizer.print_warning("Неверный выбор, используется задача 1")
        return create_test_problem_1()


def _select_algorithm(jobs: List[Job],
                      precedence: Optional[Dict]) -> Algorithm:
    Visualizer.print_header("ВЫБОР АЛГОРИТМА")

    print("Выберите алгоритм:")
    print("1. LTH (Longest Tail Heuristic)")
    print("2. MLTH (Modified LTH)")
    print("3. BalasDPC (ветви и границы по Balas–Lenstra–Vazacopoulos)")
    print("4. ILTF (Idle Largest Tail First)")
    print("5. Полный перебор (ExactBruteForce)")
    print("6. Ветви и границы (ExactBranchAndBound)")

    choice = input("\nВаш выбор (1-6): ").strip()

    if choice == "1":
        return LTH(jobs, precedence)
    elif choice == "2":
        return MLTH(jobs, precedence)
    elif choice == "3":
        # Для BalasDPC явно укажем эвристику
        return BalasBaBDPC(jobs, precedence)
    elif choice == "4":
        return ILTF(jobs, precedence)
    elif choice == "5":
        return ExactBruteForce(jobs)
    else:
        Visualizer.print_warning("Неверный выбор, используется MLTH")
        return MLTH(jobs, precedence)


# ---------------------------------------------------------------------------
# Сценарий: сравнение алгоритмов
# ---------------------------------------------------------------------------

def run_comparison() -> Dict[str, Dict]:
    """Запускает сравнение алгоритмов на выбранной задаче."""
    jobs, constraints = _select_problem()

    comparator = AlgorithmComparator()

    # Эвристики
    comparator.register_algorithm("BalasDPC", BalasDPC)
    comparator.register_algorithm("LTH", LTH)
    comparator.register_algorithm("MLTH", MLTH)
    comparator.register_algorithm("ILTF", ILTF)

    # Точные алгоритмы — только для небольших задач без ограничений
    if constraints is None and len(jobs) <= 8:
        comparator.register_algorithm("ExactBruteForce", ExactBruteForce, timeout=10.0)

    results = comparator.compare(jobs, constraints)

    print()
    comparator.visualize_best()

    return results


# ---------------------------------------------------------------------------
# Сценарий: запуск одного алгоритма
# ---------------------------------------------------------------------------

def test_single_algorithm() -> None:
    """Интерактивный запуск одного алгоритма на выбранной задаче."""
    jobs, precedence = _select_problem()
    scheduler = _select_algorithm(jobs, precedence)

    schedule, C_max, stats = scheduler.solve()

    if not schedule:
        Visualizer.print_error("Алгоритм не нашёл решение")
        return

    scheduler.visualize(schedule, C_max, stats)


# ---------------------------------------------------------------------------
# Главное меню
# ---------------------------------------------------------------------------

def main() -> None:

    """Главная интерактивная точка входа."""
    while True:
        Visualizer.print_header("ГЛАВНОЕ МЕНЮ")

        print("1. Сравнить все алгоритмы")
        print("2. Тестировать один алгоритм")
        print("3. Выход")

        choice = input("\nВаш выбор (1-3): ").strip()

        if choice == "1":
            run_comparison()
        elif choice == "2":
            test_single_algorithm()
        elif choice == "3":
            print("До свидания!")
            break
        else:
            Visualizer.print_warning("Неверный выбор, попробуйте снова")

        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main()
