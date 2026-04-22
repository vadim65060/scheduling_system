"""
Модуль для сравнения алгоритмов
"""
import math
from typing import List, Dict, Tuple, Optional

import pandas as pd
from colorama import Fore

from core.job import Job
from core.utils import Visualizer


class AlgorithmComparator:
    """Класс для сравнения алгоритмов планирования"""

    def __init__(self):
        self.results = {}
        self.algorithms = {}

    def register_algorithm(self, name: str, algorithm_class: type, **kwargs):
        """
        Регистрирует алгоритм для сравнения.

        Args:
            name: Имя алгоритма
            algorithm_class: Класс алгоритма
            **kwargs: Аргументы для конструктора
        """
        self.algorithms[name] = (algorithm_class, kwargs)

    def compare(self,
                jobs: List[Job],
                precedence_constraints: Optional[Dict[Tuple[int, int], float]] = None,
                **kwargs) -> Dict[str, Dict]:
        """
        Сравнивает все зарегистрированные алгоритмы.

        Returns:
            Словарь с результатами
        """
        self.results = {}

        Visualizer.print_header("СРАВНЕНИЕ АЛГОРИТМОВ")
        Visualizer.print_info("Количество заданий", len(jobs))

        if precedence_constraints:
            Visualizer.print_info("Ограничений предшествования", len(precedence_constraints))

        print()

        # Запуск каждого алгоритма
        for algo_name, (algo_class, algo_kwargs) in self.algorithms.items():
            try:
                Visualizer.print_section(f"Запуск: {algo_name}")

                # Создание экземпляра алгоритма
                combined_kwargs = {**algo_kwargs, **kwargs}
                scheduler = algo_class(jobs, precedence_constraints)

                # Запуск алгоритма
                schedule, C_max, stats = scheduler.solve(**combined_kwargs)

                # Сохранение результатов
                self.results[algo_name] = {
                    'schedule': schedule,
                    'C_max': C_max,
                    'stats': stats,
                    'scheduler': scheduler
                }

                # Вывод краткой информации
                if schedule:
                    Visualizer.print_info("C_max", f"{C_max:.2f}")
                    Visualizer.print_info("Время", Visualizer.format_time(stats['execution_time']))

                    if 'nodes_explored' in stats:
                        Visualizer.print_info("Исследовано узлов", f"{stats['nodes_explored']:,}")
                    elif 'iterations' in stats:
                        Visualizer.print_info("Итераций", f"{stats['iterations']:,}")

                Visualizer.print_success(f"Алгоритм {algo_name} завершен")

            except Exception as e:
                Visualizer.print_error(f"Ошибка в алгоритме {algo_name}: {e.with_traceback()}")
                self.results[algo_name] = {
                    'error': str(e),
                    'C_max': float('inf')
                }

        # Вывод таблицы сравнения
        self._print_comparison_table()

        # Анализ результатов
        self._analyze_results()

        return self.results

    def _print_comparison_table(self):
        """Печатает таблицу сравнения"""
        data = []
        for algo_name, result in self.results.items():
            if 'error' in result:
                row = {
                    'Алгоритм': algo_name,
                    'C_max': 'ОШИБКА',
                    'Время': '-',
                    'Оптимальность': '-'
                }
            else:
                C_max = result['C_max']
                exec_time = result['stats'].get('execution_time', 0)

                row = {
                    'Алгоритм': algo_name,
                    'C_max': f"{C_max:.2f}",
                    'Время': Visualizer.format_time(exec_time),
                    'Оптимальность': self._get_optimality_status(algo_name, C_max)
                }

            data.append(row)

        # Создание DataFrame
        df = pd.DataFrame(data)

        # Вывод красивой таблицы
        Visualizer.print_header("ТАБЛИЦА СРАВНЕНИЯ")
        print(df.to_string(index=False, justify='center'))

    def _get_optimality_status(self, algo_name: str, C_max: float) -> str:
        """Определяет статус оптимальности"""
        # Находим лучшее значение
        best_C = min(r['C_max'] for r in self.results.values()
                     if 'error' not in r)

        if abs(C_max - best_C) < 1e-6:
            return "✅ ОПТИМАЛЬНО"
        else:
            gap = ((C_max - best_C) / best_C) * 100
            return f"⚠️  +{gap:.1f}%"

    def _analyze_results(self):
        """Анализирует результаты сравнения"""
        # Находим лучший алгоритм
        valid_results = {k: v for k, v in self.results.items()
                         if 'error' not in v}

        if not valid_results:
            Visualizer.print_error("Нет валидных результатов для анализа")
            return

        best_algo = min(valid_results.items(),
                        key=lambda x: x[1]['C_max'])

        best_name, best_result = best_algo
        best_C = best_result['C_max']

        Visualizer.print_section("АНАЛИЗ РЕЗУЛЬТАТОВ")
        Visualizer.print_info("Лучший алгоритм", best_name)
        Visualizer.print_info("Лучший C_max", f"{best_C:.2f}")

        # Сравнение с другими алгоритмами
        print("\n" + Fore.YELLOW + "Сравнение с лучшим:")
        for algo_name, result in valid_results.items():
            if algo_name != best_name:
                C_max = result['C_max']
                gap = ((C_max - best_C) / best_C) * 100
                if best_result['stats'].get('execution_time', 1) == 0:
                    time_ratio = math.inf
                else:
                    time_ratio = result['stats']['execution_time'] / best_result['stats'].get('execution_time', 1)

                status = f"{algo_name}: C_max +{gap:.1f}%, время ×{time_ratio:.1f}"
                print(f"  {status}")

    def visualize_best(self):
        """Визуализирует лучшее решение"""
        valid_results = {k: v for k, v in self.results.items()
                         if 'error' not in v}

        if not valid_results:
            Visualizer.print_error("Нет решений для визуализации")
            return

        # Находим лучшее решение
        best_algo = min(valid_results.items(),
                        key=lambda x: x[1]['C_max'])

        best_name, best_result = best_algo
        scheduler = best_result['scheduler']
        schedule = best_result['schedule']
        C_max = best_result['C_max']
        stats = best_result['stats']

        # Визуализация
        scheduler.visualize(schedule, C_max, stats,
                            title=f"ЛУЧШЕЕ РЕШЕНИЕ: {best_name}")

    def get_results_dataframe(self) -> pd.DataFrame:
        """Возвращает результаты в виде DataFrame"""
        data = []
        for algo_name, result in self.results.items():
            if 'error' in result:
                row = {'Алгоритм': algo_name, 'Ошибка': result['error']}
            else:
                row = {
                    'Алгоритм': algo_name,
                    'C_max': result['C_max'],
                    'Время_сек': result['stats'].get('execution_time', 0),
                    'Время': Visualizer.format_time(result['stats'].get('execution_time', 0)),
                    'Расписание': ' → '.join(map(str, result['schedule'])) if result['schedule'] else None
                }

                # Добавляем дополнительные статистики
                for key, value in result['stats'].items():
                    if key not in ['execution_time', 'algorithm']:
                        row[key] = value

            data.append(row)

        return pd.DataFrame(data)