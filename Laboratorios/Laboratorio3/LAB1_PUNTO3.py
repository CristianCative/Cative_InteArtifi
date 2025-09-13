import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# -------------------------
# CONFIGURACIÓN
# -------------------------
GROUPS = ["G1", "G2", "G3"]
TIME_SLOTS = ["Lun", "Mar", "Mié", "Jue", "Vie"]
SUBJECTS = ["Matemáticas", "Física", "Historia", "Inglés", "Arte"]

TEACHERS = {
    "T_A": {"subjects": ["Matemáticas"], "available": ["Lun", "Mié"]},
    "T_B": {"subjects": ["Física"], "available": ["Mar", "Jue"]},
    "T_C": {"subjects": ["Historia"], "available": ["Vie"]},
    "T_D": {"subjects": ["Inglés", "Arte"], "available": ["Lun","Mar","Mié","Jue","Vie"]}
}

PREFERENCES = {
    "Matemáticas": {"Lun"},
    "Física": {"Mar"},
    "Historia": {"Mié"},
    "Inglés": {"Jue"},
    "Arte": {"Vie"}
}

# Parámetros del GA ajustados
POP_SIZE = 120         # antes 80
GENERATIONS = 800      # antes 500
TOURNAMENT_SIZE = 5    # antes 3
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------
# FUNCIONES GA
# -------------------------
def create_random_individual():
    individual = {}
    for g in GROUPS:
        schedule = []
        for slot in TIME_SLOTS:
            subj = random.choice(SUBJECTS)
            possible_teachers = [t for t, info in TEACHERS.items() if subj in info["subjects"] and slot in info["available"]]
            if not possible_teachers:
                possible_teachers = [t for t, info in TEACHERS.items() if subj in info["subjects"]]
                if not possible_teachers:
                    possible_teachers = list(TEACHERS.keys())
            teacher = random.choice(possible_teachers)
            schedule.append((subj, teacher))
        individual[g] = schedule
    return individual

def evaluate_individual(ind):
    penalty = 0.0
    # Penalización por choques de profesores
    for slot_idx, slot in enumerate(TIME_SLOTS):
        teach_count = {}
        for g in GROUPS:
            subj, teacher = ind[g][slot_idx]
            teach_count[teacher] = teach_count.get(teacher, 0) + 1
        for teacher, count in teach_count.items():
            if count > 1:
                penalty += (count - 1) * 5.0   # antes 15.0
    
    # Penalización por disponibilidad y preferencias
    for g in GROUPS:
        for slot_idx, slot in enumerate(TIME_SLOTS):
            subj, teacher = ind[g][slot_idx]
            if slot not in TEACHERS[teacher]["available"]:
                penalty += 4.0   # antes 8.0
            pref = PREFERENCES.get(subj, set())
            if pref and slot not in pref:
                penalty += 1.0   # antes 3.0
    
    # Balance de carga entre profesores
    load = {t: 0 for t in TEACHERS}
    for g in GROUPS:
        for slot_idx in range(len(TIME_SLOTS)):
            _, teacher = ind[g][slot_idx]
            load[teacher] += 1
    penalty += np.array(list(load.values())).std() * 0.5  # antes 1.5
    
    fitness = 1.0 / (1.0 + penalty)
    return fitness, penalty

def tournament_selection(pop, fitnesses, k=TOURNAMENT_SIZE):
    selected = random.sample(range(len(pop)), k)
    best = selected[0]
    for idx in selected[1:]:
        if fitnesses[idx] > fitnesses[best]:
            best = idx
    return deepcopy(pop[best])

def crossover(parent1, parent2):
    child = {}
    for g in GROUPS:
        if random.random() < 0.5:
            child[g] = deepcopy(parent1[g])
        else:
            child[g] = deepcopy(parent2[g])
    return child

def mutate(individual, mutation_rate):
    for g in GROUPS:
        for slot_idx, slot in enumerate(TIME_SLOTS):
            if random.random() < mutation_rate:
                if random.random() < 0.6:
                    subj = random.choice(SUBJECTS)
                    candidates = [t for t, info in TEACHERS.items() if subj in info["subjects"] and slot in info["available"]]
                    if not candidates:
                        candidates = [t for t, info in TEACHERS.items() if subj in info["subjects"]]
                        if not candidates:
                            candidates = list(TEACHERS.keys())
                    teacher = random.choice(candidates)
                    individual[g][slot_idx] = (subj, teacher)
                else:
                    subj, _ = individual[g][slot_idx]
                    candidates = [t for t, info in TEACHERS.items() if subj in info["subjects"] and slot in info["available"]]
                    if not candidates:
                        candidates = [t for t, info in TEACHERS.items() if subj in info["subjects"]]
                        if not candidates:
                            candidates = list(TEACHERS.keys())
                    teacher = random.choice(candidates)
                    individual[g][slot_idx] = (subj, teacher)
    return individual

def run_genetic(pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=0.1):
    population = [create_random_individual() for _ in range(pop_size)]
    best_overall, best_overall_fit, best_overall_penalty = None, -1, float('inf')
    best_history, avg_history = [], []
    for _ in range(generations):
        fitnesses, penalties = [], []
        for ind in population:
            fit, pen = evaluate_individual(ind)
            fitnesses.append(fit); penalties.append(pen)
        avg = float(np.mean(fitnesses))
        best_idx = int(np.argmax(fitnesses))
        best_fit, best_pen = fitnesses[best_idx], penalties[best_idx]
        if best_fit > best_overall_fit:
            best_overall_fit, best_overall_penalty = best_fit, best_pen
            best_overall = deepcopy(population[best_idx])
        best_history.append(best_fit); avg_history.append(avg)
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate=mutation_rate)
            new_pop.append(child)
        population = new_pop
    return best_overall, best_overall_fit, best_overall_penalty, best_history, avg_history

# -------------------------
# VISUALIZACIÓN DEL HORARIO
# -------------------------
def show_schedule_table(best, best_fit, best_penalty):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    col_labels = ["Grupo"] + TIME_SLOTS
    table_data = []
    for g in GROUPS:
        row = [g] + [f"{subj}\n{teacher}" for (subj, teacher) in best[g]]
        table_data.append(row)
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    plt.title(f"Mejor Horario - Fitness={best_fit:.4f}, Penalización={best_penalty:.2f}")
    plt.show()

# -------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------
if __name__ == "__main__":
    # Comparación con distintas tasas de mutación
    mutation_rates = [0.04, 0.1, 0.2]
    plt.figure(figsize=(8,4))
    for mr in mutation_rates:
        _, _, _, best_hist, avg_hist = run_genetic(mutation_rate=mr)
        plt.plot(best_hist, label=f"Best MR={mr}")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Comparación de Convergencia con Diferentes Tasas de Mutación")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Ejecutar con la mejor tasa (0.1 por defecto) y mostrar horario
    best, best_fit, best_penalty, _, _ = run_genetic(mutation_rate=0.1)
    show_schedule_table(best, best_fit, best_penalty)
