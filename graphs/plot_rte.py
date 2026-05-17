import matplotlib.pyplot as plt
import numpy as np

data = {
    'RTE(BalasB&BDPC)': [1.03, 1.08, 1.13, 1.17],
    'BestH': [1.02, 1.02, 1.02, 1.01],
    'ILTF': [1.05, 1.09, 1.11, 1.13]
}

# Ось X (наборы данных L1, L2, L3, L4)
L = np.arange(1, 5)  # [1, 2, 3, 4]

# Настройки графика
plt.figure(figsize=(10, 6))
plt.plot(L, data['RTE(BalasB&BDPC)'], marker='o', linewidth=2, markersize=8, label='RTE(BalasB&BDPC)')
plt.plot(L, data['BestH'], marker='s', linewidth=2, markersize=8, label='BestH')
plt.plot(L, data['ILTF'], marker='^', linewidth=2, markersize=8, label='ILTF')

# Настройка осей и заголовков
plt.xlabel('L max', fontsize=12)
plt.ylabel('С_max относительно BalasB&BDPC, %', fontsize=12)
plt.title('Сравнение алгоритмов RTE(BalasB&BDPC), BestH и ILTF', fontsize=14)
plt.xticks(L, [f'L{i*25}' for i in L])
plt.grid(True, linestyle='--', alpha=0.7)

# Добавление подписей значений
for i in range(len(L)):
    plt.text(L[i] + 0.02, data['RTE(BalasB&BDPC)'][i] + 0.002, f"{data['RTE(BalasB&BDPC)'][i]:.2f}", ha='left', va='bottom', fontsize=9)
    plt.text(L[i] + 0.02, data['BestH'][i] + 0.002, f"{data['BestH'][i]:.2f}", ha='left', va='bottom', fontsize=9)
    plt.text(L[i] + 0.02, data['ILTF'][i] + 0.002, f"{data['ILTF'][i]:.2f}", ha='left', va='bottom', fontsize=9)

plt.legend()
plt.tight_layout()
plt.savefig('img/rte_comparison.png', dpi=400, bbox_inches='tight')
plt.show()