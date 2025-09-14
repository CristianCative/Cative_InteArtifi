import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -------------------------
# Definir manualmente las ciudades (x,y)
# -------------------------
cities = [
    (30, 100),  # Ciudad 0
    (40, 180),  # Ciudad 1
    (600, 260),  # Ciudad 2
    (100, 340),  # Ciudad 3
    (10, 420),  # Ciudad 4
    (200, 40),  # Ciudad 5
    (120, 240),  # Ciudad 5
   (50, 340),  # Ciudad 5
]

# -------------------------
# Algoritmo Genético (TSP)
# -------------------------
def euclidean(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def total_distance(coords, route):
    dist = 0.0
    for i in range(len(route)-1):
        dist += euclidean(coords[route[i]], coords[route[i+1]])
    dist += euclidean(coords[route[-1]], coords[route[0]])
    return dist

def fitness(coords, route):
    return 1.0 / (total_distance(coords, route) + 1e-9)

def ordered_crossover(parent1, parent2):
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None]*n
    child[a:b] = parent1[a:b]
    pos = b % n
    for g in parent2:
        if g not in child:
            child[pos] = g
            pos = (pos+1)%n
    return child

def swap_mutation(route, rate=0.1):
    if random.random() < rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def run_ga(coords, pop_size=60, generations=100, mutation_rate=0.15):
    n = len(coords)
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    best_route = None
    best_fit = -1
    history = []
    for gen in range(generations):
        fits = [fitness(coords, r) for r in population]
        gen_best_index = int(np.argmax(fits))
        gen_best_fit = fits[gen_best_index]
        gen_best_route = population[gen_best_index]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_route = gen_best_route.copy()
        history.append(gen_best_fit)

        new_pop = []
        for _ in range(pop_size):
            i,j = random.sample(range(pop_size), 2)
            parent1 = population[i] if fits[i] > fits[j] else population[j]
            i,j = random.sample(range(pop_size), 2)
            parent2 = population[i] if fits[i] > fits[j] else population[j]
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            new_pop.append(child)
        population = new_pop
    return best_route, best_fit, history

# -------------------------
# Graficar ruta encontrada
# -------------------------
def plot_route(coords, route, dist):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('equal')
    ax.axis('off')

    n = len(route)
    for i in range(n):
        a = coords[route[i]]
        b = coords[route[(i+1)%n]]
        arrow = FancyArrowPatch((a[0], a[1]), (b[0], b[1]),
                                arrowstyle='->', mutation_scale=15,
                                lw=2, color='red', shrinkA=10, shrinkB=10)
        ax.add_patch(arrow)

    for idx,(x,y) in enumerate(coords):
        ax.scatter([x], [y], facecolors='none', edgecolors='black', s=300, linewidths=1.5)
        ax.text(x+10, y, str(idx), fontsize=12, color='black', weight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    start = coords[route[0]]
    ax.scatter([start[0]], [start[1]], s=140, facecolors='green', edgecolors='black', zorder=10)
    ax.text(start[0]+15, start[1]+15, "START", fontsize=10, color='darkgreen', weight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # Agregar información del mejor recorrido
    info_text = f"Mejor ruta: {route}\nDistancia total: {dist:.2f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.show()

# -------------------------
# Ejecutar
# -------------------------
best_route, best_fit, history = run_ga(cities)
dist = total_distance(cities, best_route)

print("Mejor ruta:", best_route)
print("Distancia total:", dist)

plot_route(cities, best_route, dist)

