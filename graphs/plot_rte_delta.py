import matplotlib.pyplot as plt
import numpy as np

# Данные из четырёх таблиц (L25, L50, L75, L100)
# Значения Δ % для каждой эвристики
data = {
    'BalasB&BDPC': [4.1, 11.6, 20.6, 30.2],
    'ILTF': [5.5, 14.2, 24.4, 34.9],
    'BestOfHeuristics': [5.9, 15.3, 26.4, 37.7],
    'LTH': [6.3, 16.8, 28.8, 40.6],
    'MLTH': [6.8, 17.4, 30.0, 41.5],
}

# Уровни L (по оси X)
L_levels = ['L25', 'L50', 'L75', 'L100']
x_pos = np.arange(len(L_levels))

# Цвета для каждой эвристики
colors = {
    'BalasB&BDPC': '#1f77b4',
    'ILTF': '#ff7f0e',
    'BestOfHeuristics': '#2ca02c',
    'LTH': '#d62728',
    'MLTH': '#9467bd',
}

# Создаём график
fig, ax = plt.subplots(figsize=(14, 7))

bar_width = 0.15

for idx, (algo_name, values) in enumerate(data.items()):
    bars = ax.bar(
        x_pos + idx * bar_width - 2 * bar_width,
        values,
        bar_width,
        label=algo_name,
        color=colors[algo_name],
        edgecolor='white',
        linewidth=0.5,
    )
    # Подписи над столбцами
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.5,
            f'{val}%',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

ax.set_xlabel('Максимальная задержка DPC', fontsize=12)
ax.set_ylabel('Увеличение C_max относительно relaxed, Δ %', fontsize=12)
ax.set_title('Влияние отложенных ограничений предшествования (DPC) на качество расписания', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(L_levels, fontsize=11)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9, edgecolor='gray')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, max(max(v) for v in data.values()) * 1.12)

# Линии сетки по горизонтали
for y_val in [10, 20, 30, 40]:
    ax.axhline(y=y_val, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig('img/rte_delta_comparison.png', dpi=400, bbox_inches='tight')
plt.show()