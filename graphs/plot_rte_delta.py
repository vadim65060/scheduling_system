import matplotlib.pyplot as plt
import numpy as np

# Данные из четырёх таблиц (L25, L50, L75, L100)
# Значения Δ % для каждой эвристики
data = {
    'BalasB&BDPC': [1.8, 5.5, 12.8, 19.6],
    'ILTF': [5.1, 13.4, 23.3, 33.1],
    'BestOfHeuristics': [5.1, 13.5, 23.4, 33.2],
    'LTH': [5.8, 15.8, 27.2, 38.5],
    'MLTH': [6.5, 17.0, 29.0, 40.3],
}

# Уровни L (по оси X)
L_levels = ['L25', 'L50', 'L75', 'L100']
x = np.arange(len(L_levels))  # [0, 1, 2, 3]

# Цвета и маркеры для каждой эвристики
styles = {
    'BalasB&BDPC': {'color': '#1f77b4', 'marker': 'o', 'linewidth': 2.5, 'markersize': 10},
    'ILTF': {'color': '#ff7f0e', 'marker': 's', 'linewidth': 2.0, 'markersize': 9},
    'BestOfHeuristics': {'color': '#2ca02c', 'marker': 'D', 'linewidth': 2.0, 'markersize': 9},
    'LTH': {'color': '#d62728', 'marker': '^', 'linewidth': 1.5, 'markersize': 8},
    'MLTH': {'color': '#9467bd', 'marker': 'v', 'linewidth': 1.5, 'markersize': 8},
}

# =========================================================================
# Общее полотно с двумя подграфиками
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ---- Левый график: линии ----
for algo_name, values in data.items():
    style = styles[algo_name]
    ax1.plot(
        x, values,
        marker=style['marker'],
        color=style['color'],
        linewidth=style['linewidth'],
        markersize=style['markersize'],
        label=algo_name,
    )
    # Подписи значений
    for i, val in enumerate(values):
        offset_y = 0.8 if algo_name == 'BalasB&BDPC' else -1.2
        ax1.text(
            x[i] + 0.05, val + offset_y,
            f'{val}%',
            fontsize=8,
            color=style['color'],
            ha='left',
            va='bottom',
            fontweight='bold' if algo_name == 'BalasB&BDPC' else 'normal',
        )

ax1.set_xlabel('Максимальная задержка DPC', fontsize=12)
ax1.set_ylabel('Увеличение C_max относительно relaxed, Δ %', fontsize=12)
ax1.set_title('Линейный график', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(L_levels, fontsize=10)
ax1.set_yticks(np.arange(0, 46, 5))
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='gray')

for y_val in [10, 20, 30, 40]:
    ax1.axhline(y=y_val, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)

# ---- Правый график: столбцы ----
bar_width = 0.15
x_pos = np.arange(len(L_levels))

for idx, (algo_name, values) in enumerate(data.items()):
    style = styles[algo_name]
    bars = ax2.bar(
        x_pos + idx * bar_width - 2 * bar_width,
        values,
        bar_width,
        label=algo_name,
        color=style['color'],
        edgecolor='white',
        linewidth=0.5,
    )
    # Подписи над столбцами
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.5,
            f'{val}%',
            ha='center',
            va='bottom',
            fontsize=7,
            fontweight='bold',
        )

ax2.set_xlabel('Максимальная задержка DPC', fontsize=12)
ax2.set_ylabel('Увеличение C_max относительно relaxed, Δ %', fontsize=12)
ax2.set_title('Столбчатая диаграмма', fontsize=13)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(L_levels, fontsize=10)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='gray')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_ylim(0, max(max(v) for v in data.values()) * 1.12)

# Общий заголовок
fig.suptitle(
    'Влияние отложенных ограничений предшествования (DPC) на качество расписания',
    fontsize=15,
    fontweight='bold',
    y=1.0,
)

plt.tight_layout()
plt.savefig('img/rte_delta_comparison.png', dpi=800, bbox_inches='tight')
plt.show()
