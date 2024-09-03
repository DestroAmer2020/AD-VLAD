import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Функція, що описує систему Лоренца
def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Параметри системи Лоренца
parameters = [
    {"sigma": 10, "rho": 28, "beta": 8/3},
    {"sigma": 14, "rho": 46.92, "beta": 4},
    {"sigma": 7, "rho": 60, "beta": 2},
]

# Початкові умови
initial_state = [1.0, 1.0, 1.0]

# Часові параметри
t_start = 0.0
t_end = 50.0
t_points = 10000
t_eval = np.linspace(t_start, t_end, t_points)

# Генерація та візуалізація траєкторій для кожного набору параметрів
fig = plt.figure(figsize=(18, 6))

for i, params in enumerate(parameters):
    # Розв'язок системи Лоренца для даних параметрів
    sol = solve_ivp(lorenz_system, [t_start, t_end], initial_state, args=(params["sigma"], params["rho"], params["beta"]), t_eval=t_eval, method='RK45')

    # Візуалізація траєкторії в 3D 
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter(sol.y[0], sol.y[1], sol.y[2], c=t_eval, cmap='viridis', s=0.5)
    ax.set_title(f"σ={params['sigma']}, ρ={params['rho']}, β={params['beta']}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()

# Аналіз впливу параметрів
for params in parameters:
    print(f"Для параметрів: σ={params['sigma']}, ρ={params['rho']}, β={params['beta']}")
    print("Спостереження: зміна параметрів значно впливає на траєкторію.")
    print("------------------------------------------------------")
