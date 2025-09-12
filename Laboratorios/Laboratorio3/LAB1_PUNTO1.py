import random
import numpy as np
import matplotlib.pyplot as plt

# --- Definir función objetivo ---
def f(x):
    return x**2 - 3*x + 4

# --- Algoritmo Genético continuo (mínimo global) ---
def genetic_algorithm(f, bounds=(-10, 10), pop_size=50, generations=200, mutation_rate=0.1):
    L, R = bounds
    # Población inicial aleatoria en [L, R]
    population = [random.uniform(L, R) for _ in range(pop_size)]
    
    for _ in range(generations):
        # Evaluar fitness como el negativo de f (para minimizar)
        fitness = [-f(x) for x in population]
        
        # Selección (torneo de 3)
        def tournament():
            contenders = random.sample(population, 3)
            return min(contenders, key=f)  # elige el de menor f(x)
        
        new_population = []
        for _ in range(pop_size):
            # Cruce: promedio entre padres
            p1, p2 = tournament(), tournament()
            child = (p1 + p2) / 2
            # Mutación
            if random.random() < mutation_rate:
                child += random.uniform(-1, 1)
            # Mantener en rango
            child = max(L, min(R, child))
            new_population.append(child)
        
        population = new_population
    
    # Mejor individuo final (mínimo global)
    best = min(population, key=f)
    return best, f(best)

# --- Ejecución ---
best_x, best_y = genetic_algorithm(f, bounds=(-10, 10), generations=200)
print(f"Mínimo encontrado: x = {best_x:.4f}, f(x) = {best_y:.4f}")

# --- Graficar ---
xs = np.linspace(-10, 10, 400)
ys = f(xs)

plt.plot(xs, ys, label="f(x)")
plt.scatter(best_x, best_y, color="red", s=80, label="Mínimo encontrado")

# Mostrar el valor mínimo en la gráfica
plt.text(best_x, best_y + 1, f"({best_x:.2f}, {best_y:.2f})",
         ha="center", fontsize=9, color="black", weight="bold")

# Mostrar la función en la ventana
plt.text(-9, max(ys)-10, r"$f(x) = x^2 - 3x + 4$", fontsize=12, color="blue")

plt.title("Algoritmo Genético - Mínimo Global")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
