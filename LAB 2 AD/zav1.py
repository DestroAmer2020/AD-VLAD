import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Встановлюємо фіксоване зерно для відтворюваності результатів
np.random.seed(42)

# Параметри генерації даних
num_samples = 500  # Кількість точок для кожного класу

# Генерація точок для класу 1
class1_x = np.random.normal(loc=2, scale=1.0, size=num_samples)
class1_y = np.random.normal(loc=2, scale=1.0, size=num_samples)
class1 = np.column_stack((class1_x, class1_y, np.ones(num_samples)))

# Генерація точок для класу 2
class2_x = np.random.normal(loc=-2, scale=1.0, size=num_samples)
class2_y = np.random.normal(loc=-2, scale=1.0, size=num_samples)
class2 = np.column_stack((class2_x, class2_y, np.zeros(num_samples)))

# Об'єднання даних в один масив
data = np.vstack((class1, class2))

# Перетворення в DataFrame для зручності роботи
df = pd.DataFrame(data, columns=["x", "y", "class"])

# Збереження даних у CSV-файл
df.to_csv("file1.csv", index=False)

# Візуалізація даних
plt.figure(figsize=(8, 6))
plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'], color='blue', label='Class 1')
plt.scatter(df[df['class'] == 0]['x'], df[df['class'] == 0]['y'], color='red', label='Class 2')

# Лінія розділу (наприклад, y = x)
x_vals = np.linspace(-5, 5, 100)
y_vals = x_vals
plt.plot(x_vals, y_vals, '--', color='black', label='Decision Boundary')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Лінійно роздільні класи')
plt.grid(True)
plt.show()

# Гістограма розподілу для класу 1
plt.figure(figsize=(8, 4))
plt.hist(df[df['class'] == 1]['x'], bins=20, alpha=0.7, label='Class 1 (x)')
plt.hist(df[df['class'] == 1]['y'], bins=20, alpha=0.7, label='Class 1 (y)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Розподіл даних для класу 1')
plt.grid(True)
plt.show()

# Гістограма розподілу для класу 2
plt.figure(figsize=(8, 4))
plt.hist(df[df['class'] == 0]['x'], bins=20, alpha=0.7, label='Class 2 (x)')
plt.hist(df[df['class'] == 0]['y'], bins=20, alpha=0.7, label='Class 2 (y)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Розподіл даних для класу 2')
plt.grid(True)
plt.show()
