# Laboratorio 3 - Algoritmos de Optimización con Búsqueda y Genéticos

## Introducción

Este laboratorio aborda la resolución de problemas mediante **algoritmos de búsqueda local** y **algoritmos genéticos (GA)**.  
Se trabajaron tres puntos principales: la búsqueda de máximos con Hill Climbing, la resolución del **Problema del Viajante (TSP)** y la **optimización de horarios académicos**.  

Cada punto incluye la formulación del problema, la implementación del algoritmo correspondiente y el análisis de los resultados obtenidos.

---

## Punto 1: Búsqueda Local (Hill Climbing)

En este punto se implementó el algoritmo de **Hill Climbing** para encontrar el máximo de la función cuadrática:

\[
f(x) = -(x - 3)^2 + 9
\]

La función presenta un máximo global en \(x=3\). El algoritmo explora el espacio de soluciones mediante pequeños desplazamientos aleatorios y conserva aquellas que mejoran la solución actual.

### Código relevante

```python
def f(x):
    return -(x - 3) ** 2 + 9   # máximo global en x=3

def hill_climbing(iteraciones=1000):
    x = random.uniform(-10, 10)
    for _ in range(iteraciones):
        nuevo_x = x + random.uniform(-0.1, 0.1)
        if f(nuevo_x) > f(x):
            x = nuevo_x
    return x, f(x)
```

### Ejecución

```bash
python LAB1_PUNTO1.py
```

---

## Punto 2: Problema del Viajante (TSP) con Algoritmo Genético

Se abordó el clásico **Travelling Salesman Problem (TSP)**, cuyo objetivo es encontrar la ruta más corta que pase exactamente una vez por cada ciudad y regrese al punto inicial.  

Se empleó un **algoritmo genético** con las siguientes operaciones:
- **Función de aptitud:** inversa de la distancia total.
- **Crossover ordenado (OX).**
- **Mutación por intercambio (Swap Mutation).**

Se probaron dos configuraciones de ciudades:  
- Caso con **6 ciudades**.
- Caso con **8 ciudades**.

### Código relevante

```python
def total_distance(coords, route):
    dist = 0.0
    for i in range(len(route)-1):
        dist += euclidean(coords[route[i]], coords[route[i+1]])
    dist += euclidean(coords[route[-1]], coords[route[0]])
    return dist
```

### Ejecución

```bash
python LAB1_PUNTO2.py
```

---

## Punto 3: Generación de Horarios con Algoritmos Genéticos

El último punto consistió en la **generación automática de horarios académicos** para distintos grupos de estudiantes, asignando materias y docentes bajo restricciones de disponibilidad y preferencias.

- Se definieron **materias, profesores, días y grupos**.  
- Se aplicó un algoritmo genético que minimiza penalizaciones por:
  - Choques de profesores.
  - Violación de disponibilidad.
  - Incumplimiento de preferencias.
  - Carga desequilibrada de docentes.  

El resultado final muestra:
- **Curvas de convergencia** con distintas tasas de mutación.
- Un **horario optimizado** con **penalización mínima** y su valor de *fitness*.

### Código relevante

```python
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
                penalty += (count - 1) * 5.0
    ...
    fitness = 1.0 / (1.0 + penalty)
    return fitness, penalty
```

### Ejecución

```bash
python LAB1_PUNTO3.py
```

---

## Conclusiones

- El **Hill Climbing** permitió observar cómo los algoritmos de búsqueda local exploran el espacio de soluciones y convergen hacia máximos, aunque con riesgo de quedar atrapados en óptimos locales.  
- En el **TSP**, el uso de algoritmos genéticos proporcionó soluciones eficientes para instancias pequeñas y medianas, demostrando la capacidad de los operadores evolutivos para optimizar rutas complejas.  
- En la **generación de horarios**, los algoritmos genéticos mostraron ser una herramienta flexible para problemas de asignación con múltiples restricciones, logrando horarios viables y penalizaciones reducidas.

---
