import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Встановлюємо фіксоване зерно для відтворюваності результатів
np.random.seed(42)

# Параметри для генерації тора
num_samples = 500  # Кількість точок для кожного класу
inner_radius = 5  # Внутрішній радіус внутрішнього тора
outer_radius = 10  # Внутрішній радіус зовнішнього тора
width_inner = 1  # Ширина внутрішнього тора
width_outer = 1  # Ширина зовнішнього тора
noise = 0.2  # Параметр шуму для варіативності

# Генерація точок для внутрішнього тора (клас 1)
theta_inner = np.random.uniform(0, 2 * np.pi, num_samples)
r_inner = np.random.normal(inner_radius, width_inner, num_samples)
x_inner = r_inner * np.cos(theta_inner) + np.random.normal(0, noise, num_samples)
y_inner = r_inner * np.sin(theta_inner) + np.random.normal(0, noise, num_samples)
class1 = np.column_stack((x_inner, y_inner, np.ones(num_samples)))

# Генерація точок для зовнішнього тора (клас 2)
theta_outer = np.random.uniform(0, 2 * np.pi, num_samples)
r_outer = np.random.normal(outer_radius, width_outer, num_samples)
x_outer = r_outer * np.cos(theta_outer) + np.random.normal(0, noise, num_samples)
y_outer = r_outer * np.sin(theta_outer) + np.random.normal(0, noise, num_samples)
class2 = np.column_stack((x_outer, y_outer, np.full(num_samples, 2)))

# Об'єднання даних в один масив
data = np.vstack((class1, class2))

# Перетворення в DataFrame для зручності роботи
df = pd.DataFrame(data, columns=["x", "y", "class"])

# Збереження даних у CSV-файл
df.to_csv("file3.csv", index=False)

# Візуалізація даних (тор в торі)
plt.figure(figsize=(8, 8))
plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'], color='blue', label='Class 1 (Inner Torus)')
plt.scatter(df[df['class'] == 2]['x'], df[df['class'] == 2]['y'], color='red', label='Class 2 (Outer Torus)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Точки у формі "тор в торі"')
plt.grid(True)
plt.show()

# Окремі графіки для кожного тора
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'], color='blue', label='Class 1 (Inner Torus)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Внутрішній тор')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df[df['class'] == 2]['x'], df[df['class'] == 2]['y'], color='red', label='Class 2 (Outer Torus)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Зовнішній тор')
plt.grid(True)

plt.tight_layout()
plt.show()

# Радіальні гістограми для кожного тора
plt.figure(figsize=(10, 6))
r_inner_dist = np.sqrt(df[df['class'] == 1]['x']**2 + df[df['class'] == 1]['y']**2)
r_outer_dist = np.sqrt(df[df['class'] == 2]['x']**2 + df[df['class'] == 2]['y']**2)

plt.hist(r_inner_dist, bins=20, alpha=0.5, label='Class 1 (Inner Torus)')
plt.hist(r_outer_dist, bins=20, alpha=0.5, label='Class 2 (Outer Torus)')
plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.legend()
plt.title('Радіальні гістограми розподілу для кожного тора')
plt.grid(True)
plt.show()
