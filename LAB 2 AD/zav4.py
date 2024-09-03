import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Встановлюємо фіксоване зерно для відтворюваності результатів
np.random.seed(42)

# Параметри для генерації спіралей
num_samples = 500  # Кількість точок для кожного класу
noise = 0.05  # Параметр шуму для варіативності
n_turns = 3  # Кількість обертів спіралі

# Генерація точок для першої спіралі (клас 1)
theta1 = np.linspace(0, n_turns * 2 * np.pi, num_samples)
r1 = theta1
x1 = r1 * np.cos(theta1) + np.random.normal(0, noise, num_samples)
y1 = r1 * np.sin(theta1) + np.random.normal(0, noise, num_samples)
class1 = np.column_stack((x1, y1, np.ones(num_samples)))

# Генерація точок для другої спіралі (клас 2) в протилежному напрямку
theta2 = np.linspace(0, n_turns * 2 * np.pi, num_samples)
r2 = -theta2  # Інвертуємо радіус для створення спіралі в протилежний бік
x2 = r2 * np.cos(theta2) + np.random.normal(0, noise, num_samples)
y2 = r2 * np.sin(theta2) + np.random.normal(0, noise, num_samples)
class2 = np.column_stack((x2, y2, np.full(num_samples, 2)))

# Об'єднання даних в один масив
data = np.vstack((class1, class2))

# Перетворення в DataFrame для зручності роботи
df = pd.DataFrame(data, columns=["x", "y", "class"])

# Збереження даних у CSV-файл
df.to_csv('file4.csv', index=False)

# Візуалізація даних (подвійна спіраль)
plt.figure(figsize=(8, 8))
plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'], color='magenta', label='1-st Class')
plt.scatter(df[df['class'] == 2]['x'], df[df['class'] == 2]['y'], color='green', label='2-nd Class')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Точки у формі подвійної спіралі')
plt.grid(True)
plt.show()
