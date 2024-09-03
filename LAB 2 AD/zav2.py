import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Встановлюємо фіксоване зерно для відтворюваності результатів
np.random.seed(42)

# Параметри генерації даних
num_samples = 300  # Кількість точок для кожного класу
noise = 1.5  # Параметр шуму для збільшення перекриття

# Генерація точок для класу 1
class1_x = np.random.normal(loc=2, scale=noise, size=num_samples)
class1_y = np.random.normal(loc=2, scale=noise, size=num_samples)
class1 = np.column_stack((class1_x, class1_y, np.ones(num_samples)))

# Генерація точок для класу 2
class2_x = np.random.normal(loc=-2, scale=noise, size=num_samples)
class2_y = np.random.normal(loc=2, scale=noise, size=num_samples)
class2 = np.column_stack((class2_x, class2_y, np.full(num_samples, 2)))

# Генерація точок для класу 3
class3_x = np.random.normal(loc=0, scale=noise, size=num_samples)
class3_y = np.random.normal(loc=-2, scale=noise, size=num_samples)
class3 = np.column_stack((class3_x, class3_y, np.full(num_samples, 3)))

# Об'єднання даних в один масив
data = np.vstack((class1, class2, class3))

# Перетворення в DataFrame для зручності роботи
df = pd.DataFrame(data, columns=["x", "y", "class"])

# Збереження даних у CSV-файл
df.to_csv("file2.csv", index=False)

# Візуалізація даних
plt.figure(figsize=(8, 6))
plt.scatter(df[df['class'] == 1]['x'], df[df['class'] == 1]['y'], color='blue', label='Class 1')
plt.scatter(df[df['class'] == 2]['x'], df[df['class'] == 2]['y'], color='red', label='Class 2')
plt.scatter(df[df['class'] == 3]['x'], df[df['class'] == 3]['y'], color='green', label='Class 3')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Точки трьох класів з перекриттям')
plt.grid(True)
plt.show()

# Визначення центроїдів класів
centroids = {
    1: (2, 2),
    2: (-2, 2),
    3: (0, -2)
}

# Функція для перевірки належності до зони перекриття
def is_in_overlap(point, class_label, radius=3):
    other_classes = [key for key in centroids if key != class_label]
    for oc in other_classes:
        if distance.euclidean(point, centroids[oc]) < radius:
            return True
    return False

# Додавання колонки про перекриття
df['overlap'] = df.apply(lambda row: is_in_overlap((row['x'], row['y']), row['class']), axis=1)

# Частка точок у зоні перекриття
overlap_percentage = df['overlap'].mean() * 100
print(f"Частка точок, що належать до зон перекриття: {overlap_percentage:.2f}%")

# Гістограма розподілу для кожного класу
plt.figure(figsize=(10, 6))
for i in range(1, 4):
    plt.hist(df[df['class'] == i]['x'], bins=20, alpha=0.5, label=f'Class {i} (x)')
    plt.hist(df[df['class'] == i]['y'], bins=20, alpha=0.5, label=f'Class {i} (y)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Розподіл даних для кожного класу')
plt.grid(True)
plt.show()
