import numpy as np
import random

# Пример данных задачи
num_cities = 10
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)  # Запрет маршрутов до самого себя

# Параметры алгоритма
population_size = 100
num_generations = 100
mutation_rate = 0.2
stagnation_limit = 9  # Количество поколений без улучшений для остановки

# Генерация начальной популяции
def create_individual():
    return list(np.random.permutation(num_cities))

def create_population():
    return [create_individual() for _ in range(population_size)]

# Функция оценки
def calculate_cost(individual):
    cost = 0
    for i in range(len(individual) - 1):
        cost += distance_matrix[individual[i], individual[i + 1]]
    cost += distance_matrix[individual[-1], individual[0]]  # Замыкаем маршрут
    return cost

# Оператор скрещивания
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child

# Оператор мутации
def mutate(individual):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

# Основной алгоритм
population = create_population()
best_cost = float('inf')
best_solution = None
no_improvement_count = 0

for generation in range(num_generations):
    # Оценка популяции
    population = sorted(population, key=calculate_cost)
    current_best_cost = calculate_cost(population[0])

    if current_best_cost < best_cost:
        best_cost = current_best_cost
        best_solution = population[0]
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    print(f"Generation {generation}: Best Cost = {best_cost}")

    if no_improvement_count >= stagnation_limit:
        print(f"Остановка на поколении {generation} из-за отсутствия улучшений.")
        break

    # Селекция
    next_generation = population[:population_size // 2]

    # Генерация потомков
    while len(next_generation) < population_size:
        parents = random.sample(population[:50], 2)
        child = crossover(parents[0], parents[1])
        mutate(child)
        next_generation.append(child)

    population = next_generation

print("Best Solution:", best_solution)
print("Best Cost:", best_cost)
