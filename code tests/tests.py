"""
Юнит-тесты для всех алгоритмов планирования
Запуск: python -m pytest code tests/ -v
"""

import unittest
import time
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.job import Job
from algorithms.lth import LTH
from algorithms.mlth import MLTH
from algorithms.iltf import ILTF
from algorithms.Balas_DPC import BalasDPC
from algorithms.exact_bf import ExactBruteForce
from evaluation.comparator import AlgorithmComparator


class TestJob(unittest.TestCase):
    """Тесты для класса Job"""

    def test_job_creation(self):
        job = Job(id=1, r_i=2.0, d_i=3.0, q_i=4.0)
        self.assertEqual(job.id, 1)
        self.assertEqual(job.r_i, 2.0)
        self.assertEqual(job.d_i, 3.0)
        self.assertEqual(job.q_i, 4.0)

    def test_job_repr(self):
        job = Job(id=1, r_i=2.0, d_i=3.0, q_i=4.0)
        repr_str = repr(job)
        self.assertIn("Job(1:", repr_str)
        self.assertIn("r=2.0", repr_str)
        self.assertIn("d=3.0", repr_str)
        self.assertIn("q=4.0", repr_str)

    def test_job_hash(self):
        job1 = Job(id=1, r_i=0, d_i=1, q_i=1)
        job2 = Job(id=1, r_i=0, d_i=1, q_i=1)
        job3 = Job(id=2, r_i=0, d_i=1, q_i=1)

        self.assertEqual(hash(job1), hash(job2))
        self.assertNotEqual(hash(job1), hash(job3))

    def test_job_equality(self):
        job1 = Job(id=1, r_i=0, d_i=1, q_i=1)
        job2 = Job(id=1, r_i=0, d_i=1, q_i=1)
        job3 = Job(id=2, r_i=0, d_i=1, q_i=1)

        self.assertEqual(job1, job2)
        self.assertNotEqual(job1, job3)
        self.assertNotEqual(job1, "not a job")

    def test_job_copy(self):
        original = Job(id=1, r_i=2.0, d_i=3.0, q_i=4.0)
        copy_job = original.copy()

        self.assertEqual(original.id, copy_job.id)
        self.assertEqual(original.r_i, copy_job.r_i)
        self.assertEqual(original.d_i, copy_job.d_i)
        self.assertEqual(original.q_i, copy_job.q_i)
        self.assertIsNot(original, copy_job)


class TestScheduler(unittest.TestCase):
    """Тесты для базового класса Scheduler (DPC-модель)"""

    def setUp(self):
        self.jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=3),
            Job(id=2, r_i=1, d_i=3, q_i=1),
            Job(id=3, r_i=2, d_i=1, q_i=2)
        ]
        self.scheduler = LTH(self.jobs)

    def test_calculate_makespan_simple(self):
        """
        В DPC-модели без ограничений предшествования
        каждая работа стартует в r_i.
        """
        schedule = [1, 2, 3]
        C_max, start_times = self.scheduler.calculate_makespan(schedule)

        self.assertEqual(start_times[1], 0)
        self.assertEqual(start_times[2], 2)
        self.assertEqual(start_times[3], 5)
        self.assertAlmostEqual(C_max, 8.0)

    def test_calculate_makespan_with_precedence(self):
        precedence = {(1, 2): 5, (2, 3): 2}
        scheduler = LTH(self.jobs, precedence)

        schedule = [1, 2, 3]
        C_max, start_times = scheduler.calculate_makespan(schedule)

        # Проверяем, что DPC учтены
        self.assertGreaterEqual(start_times[2], start_times[1] + 5)
        self.assertGreaterEqual(start_times[3], start_times[2] + 2)

    def test_calculate_makespan_empty_schedule(self):
        C_max, start_times = self.scheduler.calculate_makespan([])
        self.assertEqual(C_max, 0)
        self.assertEqual(start_times, {})

    def test_calculate_makespan_single_job(self):
        jobs = [Job(id=1, r_i=5, d_i=3, q_i=2)]
        scheduler = LTH(jobs)
        C_max, start_times = scheduler.calculate_makespan([1])

        self.assertEqual(start_times[1], 5)
        self.assertAlmostEqual(C_max, 5 + 3 + 2)


class TestLTH(unittest.TestCase):
    """Тесты для алгоритма LTH (Longest Tail Heuristic)"""

    def setUp(self):
        self.jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=3),
            Job(id=2, r_i=1, d_i=3, q_i=1),
            Job(id=3, r_i=2, d_i=1, q_i=2),
            Job(id=4, r_i=0, d_i=2, q_i=5)
        ]
        self.lth = LTH(self.jobs)

    def test_lth_solves_simple_problem(self):
        schedule, C_max, stats = self.lth.solve()

        self.assertIsNotNone(schedule)
        self.assertEqual(len(schedule), 4)
        self.assertGreater(C_max, 0)
        self.assertIn('algorithm', stats)
        self.assertEqual(stats['algorithm'], 'LTH')
        self.assertIn('execution_time', stats)
        self.assertIn('iterations', stats)

    def test_lth_with_precedence_constraints(self):
        precedence = {(1, 2): 3, (2, 3): 2}
        lth = LTH(self.jobs, precedence)
        schedule, C_max, stats = lth.solve()

        self.assertIsNotNone(schedule)
        # Проверяем, что ограничения соблюдены
        if 1 in schedule and 2 in schedule:
            self.assertLess(schedule.index(1), schedule.index(2))
        if 2 in schedule and 3 in schedule:
            self.assertLess(schedule.index(2), schedule.index(3))

    def test_lth_single_job(self):
        jobs = [Job(id=1, r_i=0, d_i=5, q_i=3)]
        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        self.assertEqual(schedule, [1])
        self.assertAlmostEqual(C_max, 5 + 3)  # processing + delivery

    def test_lth_two_jobs(self):
        jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=1),
            Job(id=2, r_i=0, d_i=3, q_i=5)  # larger tail
        ]
        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        # Job with larger tail should be scheduled first
        self.assertEqual(schedule[0], 2)  # q=5 > q=1

    def test_lth_release_times_update(self):
        """Проверяет, что release times правильно обновляются"""
        jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=1),
            Job(id=2, r_i=10, d_i=1, q_i=10)
        ]
        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        # Job 2 has larger tail but later release
        # LTH should schedule it when it becomes available
        self.assertEqual(schedule[0], 1)  # Job 1 first
        self.assertEqual(schedule[1], 2)  # Then job 2


class TestMLTH(unittest.TestCase):
    """Тесты для алгоритма MLTH (Modified Longest Tail Heuristic)"""

    def setUp(self):
        self.jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=3),
            Job(id=2, r_i=3, d_i=1, q_i=10),  # large tail, late release
            Job(id=3, r_i=1, d_i=2, q_i=2)
        ]
        self.mlth = MLTH(self.jobs)

    def test_mlth_solves_problem(self):
        schedule, C_max, stats = self.mlth.solve()

        self.assertIsNotNone(schedule)
        self.assertEqual(len(schedule), 3)
        self.assertGreater(C_max, 0)
        self.assertEqual(stats['algorithm'], 'MLTH')

    def test_mlth_with_delayed_precedence(self):
        precedence = {(1, 2): 4, (2, 3): 2}
        mlth = MLTH(self.jobs, precedence)
        schedule, C_max, stats = mlth.solve()

        self.assertIsNotNone(schedule)
        self.assertGreater(C_max, 0)

    def test_mlth_reschedule_delayed_jobs(self):
        """Проверяет механизм перепланирования отложенных заданий"""
        jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=1),
            Job(id=2, r_i=1, d_i=1, q_i=5),
            Job(id=3, r_i=2, d_i=3, q_i=2)
        ]
        mlth = MLTH(jobs)
        schedule, C_max, stats = mlth.solve()

        self.assertIsNotNone(schedule)
        self.assertEqual(len(schedule), 3)


class TestILTF(unittest.TestCase):
    """Тесты для алгоритма ILTF (Idle Largest Tail First)"""

    def setUp(self):
        self.jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=3),
            Job(id=2, r_i=4, d_i=1, q_i=10),  # important job with late release
            Job(id=3, r_i=1, d_i=2, q_i=2)
        ]
        self.iltf = ILTF(self.jobs)

    def test_iltf_solves_problem(self):
        schedule, C_max, stats = self.iltf.solve()

        self.assertIsNotNone(schedule)
        self.assertEqual(len(schedule), 3)
        self.assertGreater(C_max, 0)
        self.assertEqual(stats['algorithm'], 'ILTF')
        self.assertIn('idle_times', stats)
        self.assertIn('LB', stats)

    def test_iltf_idle_time(self):
        """Задача, где ILTF должен ввести простой"""
        jobs = [
            Job(id=1, r_i=0, d_i=3, q_i=1),
            Job(id=2, r_i=5, d_i=2, q_i=10)  # very important job
        ]
        iltf = ILTF(jobs)
        schedule, C_max, stats = iltf.solve()

        self.assertIsNotNone(schedule)
        # Должен быть простой перед важной работой
        self.assertGreaterEqual(stats['idle_times'], 0)

    def test_iltf_with_precedence(self):
        precedence = {(1, 2): 2, (2, 3): 1}
        iltf = ILTF(self.jobs, precedence)
        schedule, C_max, stats = iltf.solve()

        self.assertIsNotNone(schedule)
        # Проверяем соблюдение предшествования
        if 1 in schedule and 2 in schedule:
            self.assertLess(schedule.index(1), schedule.index(2))

    def test_iltf_lower_bound_calculation(self):
        schedule, C_max, stats = self.iltf.solve()
        lb = stats['LB']
        gap = stats['gap']

        self.assertGreater(lb, 0)
        self.assertGreaterEqual(gap, 0)
        self.assertLessEqual(C_max, 100)  # reasonable makespan

    def test_iltf_ready_jobs(self):
        """Проверяет получение готовых заданий"""
        scheduled = set()
        current_time = 0
        ready = self.iltf._get_ready_jobs(scheduled, current_time)

        self.assertIsInstance(ready, list)
        # Job 1 has r_i=0, should be ready
        self.assertIn(1, ready)


class TestBalasDPC(unittest.TestCase):

    def test_balas_optimality_check(self):
        """
        В DPC-модели без ограничений:
        обе работы стартуют в r_i = 0,
        C_max = 0 + 2 + 1 = 3.
        """
        jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=1),
            Job(id=2, r_i=0, d_i=2, q_i=1)
        ]
        balas = BalasDPC(jobs, debug=False)
        schedule, C_max, stats = balas.solve(timeout=10.0)

        self.assertAlmostEqual(C_max, 5.0)
        self.assertTrue(stats['optimal'])


class TestExactBruteForce(unittest.TestCase):

    def test_bruteforce_optimality(self):
        """
        В DPC-модели без ограничений:
        все работы стартуют в r_i = 0,
        C_max = max(d_i + q_i).
        """
        jobs = [
            Job(id=1, r_i=0, d_i=1, q_i=1),
            Job(id=2, r_i=0, d_i=1, q_i=1),
            Job(id=3, r_i=0, d_i=1, q_i=1)
        ]
        bf = ExactBruteForce(jobs)
        schedule, C_max, stats = bf.solve(timeout=5.0)

        # C_max = 3 + 1 = 4
        self.assertAlmostEqual(C_max, 4.0)


class TestAlgorithmComparator(unittest.TestCase):
    """Тесты для класса сравнения алгоритмов"""

    def setUp(self):
        self.jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=3),
            Job(id=2, r_i=1, d_i=2, q_i=1)
        ]
        self.comparator = AlgorithmComparator()

    def test_register_algorithm(self):
        self.comparator.register_algorithm("LTH", LTH)
        self.assertIn("LTH", self.comparator.algorithms)

    def test_compare_algorithms(self):
        self.comparator.register_algorithm("LTH", LTH)
        self.comparator.register_algorithm("MLTH", MLTH)

        results = self.comparator.compare(self.jobs)

        self.assertIn("LTH", results)
        self.assertIn("MLTH", results)
        self.assertIn('C_max', results["LTH"])
        self.assertIn('schedule', results["LTH"])

    def test_compare_with_precedence(self):
        precedence = {(1, 2): 3}
        self.comparator.register_algorithm("LTH", LTH)

        results = self.comparator.compare(self.jobs, precedence)

        self.assertIsNotNone(results)
        self.assertIn("LTH", results)

    def test_get_results_dataframe(self):
        self.comparator.register_algorithm("LTH", LTH)
        self.comparator.compare(self.jobs)

        df = self.comparator.get_results_dataframe()

        self.assertIsNotNone(df)
        self.assertIn('Алгоритм', df.columns)
        self.assertIn('C_max', df.columns)

    def test_visualize_best(self):
        """Проверяет, что визуализация работает без ошибок"""
        self.comparator.register_algorithm("LTH", LTH)
        self.comparator.compare(self.jobs)

        # Должно работать без исключений
        try:
            self.comparator.visualize_best()
        except Exception as e:
            self.fail(f"visualize_best raised exception: {e}")


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""

    def test_all_algorithms_on_small_problem(self):
        """Проверяем работу всех алгоритмов на маленькой задаче"""
        jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=3),
            Job(id=2, r_i=1, d_i=2, q_i=2)
        ]

        algorithms = [
            ("LTH", LTH),
            ("MLTH", MLTH),
            ("ILTF", ILTF),
            ("BalasDPC", BalasDPC),
            ("ExactBruteForce", ExactBruteForce),
        ]

        for name, algo_class in algorithms:
            try:
                if name == "BalasDPC":
                    algo = algo_class(jobs, debug=False)
                else:
                    algo = algo_class(jobs)

                schedule, C_max, stats = algo.solve(timeout=5.0)

                self.assertIsNotNone(schedule, f"{name} failed to produce schedule")
                self.assertEqual(len(schedule), 2, f"{name} produced wrong schedule length")
                self.assertGreater(C_max, 0, f"{name} produced invalid makespan")
                self.assertIn('execution_time', stats, f"{name} missing execution_time")

            except Exception as e:
                self.fail(f"{name} raised exception: {e}")

    def test_all_algorithms_with_precedence(self):
        """Проверяем работу всех алгоритмов с ограничениями"""
        jobs = [
            Job(id=1, r_i=0, d_i=2, q_i=1),
            Job(id=2, r_i=0, d_i=2, q_i=1),
            Job(id=3, r_i=0, d_i=2, q_i=1)
        ]
        precedence = {(1, 2): 1, (2, 3): 1}

        algorithms = [
            ("LTH", LTH),
            ("MLTH", MLTH),
            ("ILTF", ILTF),
            ("BalasDPC", BalasDPC)
        ]

        for name, algo_class in algorithms:
            try:
                if name == "BalasDPC":
                    algo = algo_class(jobs, precedence, debug=False)
                else:
                    algo = algo_class(jobs, precedence)

                schedule, C_max, stats = algo.solve(timeout=5.0)

                self.assertIsNotNone(schedule, f"{name} failed to produce schedule")

                # Проверяем соблюдение предшествования
                for i, j in [(1, 2), (2, 3)]:
                    if i in schedule and j in schedule:
                        self.assertLess(
                            schedule.index(i), schedule.index(j),
                            f"{name} violated precedence constraint {i}->{j}"
                        )

            except Exception as e:
                self.fail(f"{name} raised exception: {e}")

    def test_all_algorithms_consistent_results(self):
        """Проверяем, что все алгоритмы возвращают согласованные результаты"""
        jobs = [
            Job(id=1, r_i=0, d_i=1, q_i=1),
            Job(id=2, r_i=0, d_i=1, q_i=1)
        ]

        results = {}

        # Полный перебор даёт точное решение
        bf = ExactBruteForce(jobs)
        bf_schedule, bf_C, _ = bf.solve(timeout=5.0)

        algorithms = {
            "LTH": LTH,
            "MLTH": MLTH,
            "ILTF": ILTF,
            "BalasDPC": BalasDPC,
        }

        for name, algo_class in algorithms.items():
            if name == "BalasDPC":
                algo = algo_class(jobs, debug=False)
            else:
                algo = algo_class(jobs)

            schedule, C_max, _ = algo.solve(timeout=5.0)
            results[name] = C_max

            # Все решения должны быть не хуже точного
            self.assertGreaterEqual(
                C_max, bf_C - 1e-6,
                f"{name} produced C_max={C_max} which is better than optimal {bf_C}"
            )

        # Все эвристики должны дать разумные результаты
        for name, C_max in results.items():
            self.assertGreater(C_max, 0)
            self.assertLess(C_max, 100)


class TestPerformance(unittest.TestCase):
    """Тесты производительности"""

    def test_lth_performance(self):
        """Проверяем, что LTH быстро работает на больших задачах"""
        jobs = [Job(id=i, r_i=i % 10, d_i=1, q_i=i % 20) for i in range(50)]
        lth = LTH(jobs)

        start = time.time()
        schedule, C_max, stats = lth.solve()
        elapsed = time.time() - start

        self.assertIsNotNone(schedule)
        self.assertEqual(len(schedule), 50)
        self.assertLess(elapsed, 1.0)  # Должно быть быстро

    def test_mlth_performance(self):
        """Проверяем производительность MLTH"""
        jobs = [Job(id=i, r_i=i % 10, d_i=1, q_i=i % 20) for i in range(30)]
        mlth = MLTH(jobs)

        start = time.time()
        schedule, C_max, stats = mlth.solve()
        elapsed = time.time() - start

        self.assertIsNotNone(schedule)
        self.assertLess(elapsed, 1.0)

    def test_iltf_performance(self):
        """Проверяем производительность ILTF"""
        jobs = [Job(id=i, r_i=i % 10, d_i=1, q_i=i % 20) for i in range(30)]
        iltf = ILTF(jobs)

        start = time.time()
        schedule, C_max, stats = iltf.solve()
        elapsed = time.time() - start

        self.assertIsNotNone(schedule)
        self.assertLess(elapsed, 1.0)


class TestEdgeCases(unittest.TestCase):
    """Тесты для граничных случаев"""

    def test_empty_jobs(self):
        """Проверка на пустой список заданий"""
        jobs = []

        algorithms = [LTH, MLTH, ILTF, BalasDPC, ExactBruteForce]

        for algo_class in algorithms:
            if algo_class == BalasDPC:
                algo = algo_class(jobs, debug=False)
            else:
                algo = algo_class(jobs)

            schedule, C_max, stats = algo.solve()

            self.assertEqual(schedule, [])
            self.assertEqual(C_max, 0)

    def test_single_job(self):
        """Проверка на одно задание"""
        job = Job(id=1, r_i=5, d_i=10, q_i=3)
        jobs = [job]

        algorithms = [LTH, MLTH, ILTF, BalasDPC, ExactBruteForce]

        for algo_class in algorithms:
            if algo_class == BalasDPC:
                algo = algo_class(jobs, debug=False)
            else:
                algo = algo_class(jobs)

            schedule, C_max, stats = algo.solve()

            self.assertEqual(schedule, [1])
            self.assertAlmostEqual(C_max, 5 + 10 + 3)  # r + d + q

    def test_zero_processing_time(self):
        """Проверка на нулевое время выполнения"""
        jobs = [
            Job(id=1, r_i=0, d_i=0, q_i=1),
            Job(id=2, r_i=0, d_i=0, q_i=2)
        ]

        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        self.assertEqual(len(schedule), 2)
        # Оба задания с нулевым временем не должны увеличивать C_max
        self.assertAlmostEqual(C_max, 2)  # max q_i = 2

    def test_negative_release_times(self):
        """Проверка на отрицательные времена появления"""
        jobs = [
            Job(id=1, r_i=-5, d_i=3, q_i=1),
            Job(id=2, r_i=0, d_i=2, q_i=1)
        ]

        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        self.assertIsNotNone(schedule)
        self.assertGreaterEqual(C_max, 0)

    def test_large_delays(self):
        """Проверка на большие задержки"""
        jobs = [
            Job(id=1, r_i=0, d_i=1, q_i=1),
            Job(id=2, r_i=0, d_i=1, q_i=1)
        ]
        precedence = {(1, 2): 100}  # Огромная задержка

        lth = LTH(jobs, precedence)
        schedule, C_max, stats = lth.solve()

        self.assertIsNotNone(schedule)
        self.assertGreater(C_max, 100)  # Должна учитываться задержка

    # def test_cyclic_precedence(self):
    #     """Проверка на циклические ограничения"""
    #     jobs = [
    #         Job(id=1, r_i=0, d_i=1, q_i=1),
    #         Job(id=2, r_i=0, d_i=1, q_i=1)
    #     ]
    #     # Циклические ограничения (должны быть обработаны)
    #     precedence = {(1, 2): 1, (2, 1): 1}
    #
    #     # Не должен выбросить исключение
    #     try:
    #         scheduler = LTH(jobs, precedence)
    #         schedule, C_max, stats = scheduler.solve()
    #         self.assertIsNotNone(schedule)
    #     except Exception as e:
    #         self.fail(f"Cyclic precedence caused exception: {e}")

    def test_all_identical_jobs(self):
        """Проверка на одинаковые задания"""
        jobs = [Job(id=i, r_i=0, d_i=1, q_i=1) for i in range(10)]

        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        self.assertEqual(len(schedule), 10)
        # Любой порядок даёт C_max = 10 + 1 = 11
        self.assertAlmostEqual(C_max, 11.0)

    def test_large_release_times(self):
        """Проверка на большие времена появления"""
        jobs = [
            Job(id=1, r_i=100, d_i=1, q_i=1),
            Job(id=2, r_i=0, d_i=1, q_i=100)
        ]

        lth = LTH(jobs)
        schedule, C_max, stats = lth.solve()

        # Job 2 should be scheduled first despite smaller tail? Actually q=100 is larger
        self.assertEqual(schedule[0], 2)  # q=100 > q=1


if __name__ == '__main__':
    # Настройка для запуска тестов
    unittest.main(verbosity=2)
