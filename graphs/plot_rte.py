import matplotlib.pyplot as plt
import numpy as np

data = {
    'RTE(BalasB&BDPC)': [1.02, 1.03, 1.05, 1.07],
    'BestH': [1.08, 1.07, 1.05, 1.04],
    'LTH': [1.09, 1.07, 1.06, 1.05]
}

# Ось X (наборы данных L1, L2, L3, L4)
L = np.arange(1, 5)  # [1, 2, 3, 4]

# Настройки графика
plt.figure(figsize=(10, 6))
plt.plot(L, data['RTE(BalasB&BDPC)'], marker='o', linewidth=2, markersize=8, label='RTE(BalasB&BDPC)')
plt.plot(L, data['BestH'], marker='s', linewidth=2, markersize=8, label='BestH')
plt.plot(L, data['LTH'], marker='^', linewidth=2, markersize=8, label='LTH')

# Настройка осей и заголовков
plt.xlabel('L max', fontsize=12)
plt.ylabel('С_max относительно BalasB&BDPC, %', fontsize=12)
plt.title('Сравнение алгоритмов RTE(BalasB&BDPC), BestH и LTH', fontsize=14)
plt.xticks(L, [f'L{i*25}' for i in L])
plt.grid(True, linestyle='--', alpha=0.7)

# Добавление подписей значений
for i in range(len(L)):
    plt.text(L[i] + 0.02, data['RTE(BalasB&BDPC)'][i] + 0.002, f"{data['RTE(BalasB&BDPC)'][i]:.2f}", ha='left', va='bottom', fontsize=9)
    plt.text(L[i] + 0.02, data['BestH'][i] + 0.002, f"{data['BestH'][i]:.2f}", ha='left', va='bottom', fontsize=9)
    plt.text(L[i] + 0.02, data['LTH'][i] + 0.002, f"{data['LTH'][i]:.2f}", ha='left', va='bottom', fontsize=9)

plt.legend()
plt.tight_layout()
plt.savefig('img/rte_comparison.png', dpi=800, bbox_inches='tight')
plt.show()