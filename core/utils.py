"""
Утилиты и визуализация
"""

import sys
import time
from datetime import timedelta
from typing import List, Dict, Any

from colorama import init, Fore

from core.job import Job

# Инициализация colorama для цветного вывода
init(autoreset=True)


class Visualizer:
    """Класс для визуализации результатов"""

    @staticmethod
    def print_header(title: str, width: int = 80, char: str = "="):
        """Печатает заголовок"""
        print(Fore.CYAN + char * width)
        print(Fore.CYAN + f"{title:^{width}}")
        print(Fore.CYAN + char * width)

    @staticmethod
    def print_section(title: str, width: int = 60):
        """Печатает раздел"""
        print(Fore.YELLOW + f"\n{title}")
        print(Fore.YELLOW + "-" * len(title))

    @staticmethod
    def print_info(label: str, value: Any, indent: int = 2):
        """Печатает информацию с отступом"""
        spaces = " " * indent
        print(f"{spaces}{Fore.WHITE}{label}: {Fore.GREEN}{value}")

    @staticmethod
    def print_warning(message: str):
        """Печатает предупреждение"""
        print(Fore.YELLOW + f"⚠️  {message}")

    @staticmethod
    def print_error(message: str):
        """Печатает ошибку"""
        print(Fore.RED + f"❌ {message}")

    @staticmethod
    def print_success(message: str):
        """Печатает успех"""
        print(Fore.GREEN + f"✅ {message}")

    @staticmethod
    def create_gantt_chart(schedule: List[int],
                           start_times: Dict[int, float],
                           jobs_dict: Dict[int, Job],
                           C_max: float,
                           title: str = "ГАНТ-ДИАГРАММА") -> str:
        """
        Создает красивую Гант-диаграму

        Args:
            schedule: Порядок выполнения заданий
            start_times: Времена начала заданий
            jobs_dict: Словарь заданий
            C_max: Makespan
            title: Заголовок диаграммы

        Returns:
            Строковое представление диаграммы
        """
        result = []
        width = 70
        scale = width / (C_max + 1)

        # Заголовок
        result.append(Fore.CYAN + "=" * (width + 20))
        result.append(Fore.CYAN + f"{title:^{width + 20}}")
        result.append(Fore.CYAN + "=" * (width + 20))
        result.append("")

        # Временная шкала
        time_line = Fore.WHITE + "Время: "
        for i in range(0, int(C_max) + 2, max(1, int(C_max) // 10)):
            pos = int(i * scale)
            time_line += " " * (pos - len(time_line)) + f"{i:2d}"
        result.append(time_line)

        # Линия разделителя
        result.append(Fore.WHITE + "-" * (width + 10))

        # Строка машины
        machine_line = [" "] * width
        current_time = 0

        for job_id in schedule:
            job = jobs_dict[job_id]
            start = start_times[job_id]
            end = start + job.d_i

            # Заполняем интервал выполнения
            start_pos = int(start * scale)
            end_pos = int(end * scale)

            for pos in range(start_pos, min(end_pos, width)):
                machine_line[pos] = "█"

            current_time = end

        result.append(Fore.BLUE + "Машина: " + "".join(machine_line))

        # Номера заданий над диаграммой
        jobs_line = [" "] * width
        for job_id in schedule:
            job = jobs_dict[job_id]
            start = start_times[job_id]
            end = start + job.d_i

            # Размещаем номер задания в середине интервала
            mid_pos = int((start + job.d_i / 2) * scale)
            if 0 <= mid_pos < width - 2:
                jobs_line[mid_pos] = f"J{job_id}"

        result.append(Fore.BLUE + "Задания: " + "".join(jobs_line))
        result.append(Fore.WHITE + "-" * (width + 10))

        # Детали по каждому заданию
        for job_id in schedule:
            job = jobs_dict[job_id]
            start = start_times[job_id]
            end = start + job.d_i
            delivery_end = end + job.q_i

            # Создаем строку для задания
            job_line = [" "] * width

            # Выполнение (синий)
            start_pos = int(start * scale)
            end_pos = int(end * scale)
            for pos in range(start_pos, min(end_pos, width)):
                job_line[pos] = "█"

            # Доставка (зеленый)
            delivery_start = end_pos
            delivery_end_pos = int(delivery_end * scale)
            for pos in range(delivery_start, min(delivery_end_pos, width)):
                if pos < width and job_line[pos] == " ":
                    job_line[pos] = "░"

            # Формируем строку
            job_str = f"J{job_id}:".ljust(6)

            # Цвет для критического задания
            if delivery_end == C_max:
                job_str = Fore.RED + job_str
            else:
                job_str = Fore.WHITE + job_str

            line = job_str + "".join(job_line)

            # Добавляем временную информацию
            time_info = f" [{start:.1f}-{end:.1f}]"
            if delivery_end == C_max:
                time_info += Fore.RED + " ⭐ КРИТИЧЕСКОЕ"

            result.append(line + Fore.WHITE + time_info)

        result.append(Fore.WHITE + "-" * (width + 10))

        # Легенда
        result.append(Fore.YELLOW + "\nЛЕГЕНДА:")
        result.append(Fore.BLUE + "  █ - Выполнение на машине")
        result.append(Fore.GREEN + "  ░ - Время доставки (q_i)")
        result.append(Fore.RED + "  ⭐ - Критическое задание (определяет C_max)")

        return "\n".join(result)

    @staticmethod
    def format_time(seconds: float) -> str:
        """Форматирует время в читаемый вид"""
        if seconds < 0.001:
            return f"{seconds * 1e6:.0f}µs"
        elif seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            return str(timedelta(seconds=int(seconds)))

    @staticmethod
    def print_comparison_table(results: Dict[str, Dict], metric: str = "C_max"):
        """Печатает таблицу сравнения алгоритмов"""
        print(Fore.CYAN + "\n" + "=" * 70)
        print(Fore.CYAN + "СРАВНЕНИЕ АЛГОРИТМОВ".center(70))
        print(Fore.CYAN + "=" * 70)

        headers = ["Алгоритм", "C_max", "Время", "Узлов", "Оптимальность"]
        print(Fore.YELLOW + f"{headers[0]:<20} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>12}")
        print(Fore.YELLOW + "-" * 70)

        # Находим лучшее значение
        best_value = min(r[metric] for r in results.values())

        for algo_name, result in results.items():
            C_max = result.get('C_max', float('inf'))
            exec_time = result.get('execution_time', 0)
            nodes = result.get('nodes_explored', result.get('iterations', 0))

            # Определяем оптимальность
            if abs(C_max - best_value) < 1e-6:
                optimality = Fore.GREEN + "ОПТИМАЛЬНО"
            else:
                gap = ((C_max - best_value) / best_value) * 100
                optimality = Fore.YELLOW + f"+{gap:.1f}%"

            # Форматируем время
            time_str = Visualizer.format_time(exec_time)

            print(f"{Fore.WHITE}{algo_name:<20} {Fore.CYAN}{C_max:>10.2f} "
                  f"{Fore.MAGENTA}{time_str:>10} {Fore.BLUE}{nodes:>10,} "
                  f"{optimality}")


class Timer:
    """Контекстный менеджер для измерения времени"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

    def get_elapsed(self) -> float:
        return self.elapsed


class ProgressBar:
    """Класс для отображения прогресса"""

    def __init__(self, total: int, width: int = 50, desc: str = "Прогресс"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Обновляет прогресс"""
        self.current += n
        self.display()

    def display(self):
        """Отображает прогресс-бар"""
        progress = min(self.current / self.total, 1.0)
        bar_width = int(self.width * progress)
        bar = "█" * bar_width + "░" * (self.width - bar_width)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f"ETA: {Visualizer.format_time(eta)}"
        else:
            eta_str = "ETA: --"

        percent = progress * 100
        sys.stdout.write(f"\r{Fore.CYAN}{self.desc}: [{bar}] {percent:6.2f}% "
                         f"({self.current:,}/{self.total:,}) {eta_str}")
        sys.stdout.flush()

    def finish(self):
        """Завершает отображение прогресс-бара"""
        self.update(self.total - self.current)
        print()