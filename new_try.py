import numpy as np
import random
import math
import matplotlib.pyplot as plt
from dataset_B import all_B_set
# from main_2 import demands, coordinates

# from main_2 import coordinates

# Устанавливаем фиксированный seed для воспроизводимости результатов
random.seed(42)
np.random.seed(42)

population_size = 300
generations = 600
elite_size = 100
mutation_rate = 0.3

# Исходные данные из задачи B-n44-k7
# coordinates = [
#     (35, 31), (77, 13), (95, 51), (91, 15), (65, 93), (51, 17), (41, 57), (39, 39),
#     (70, 102), (74, 94), (81, 97), (77, 95), (84, 14), (48, 60), (98, 54), (92, 18),
#     (42, 42), (52, 18), (44, 42), (48, 44), (80, 22), (42, 58), (73, 105), (81, 101),
#     (102, 58), (82, 18), (96, 58), (98, 18), (84, 16), (44, 44), (73, 103), (96, 16),
#     (46, 48), (92, 16), (44, 64), (98, 22), (96, 52), (46, 64), (77, 103), (104, 54),
#     (58, 20), (46, 64), (73, 103), (94, 16)
# ]
# demands = [
#     0, 26, 11, 22, 18, 10, 9, 12, 10, 12, 19, 5, 18, 7, 14, 14, 20, 7, 4, 11, 3,
#     22, 15, 69, 26, 20, 12, 22, 9, 23, 7, 17, 3, 18, 18, 9, 4, 20, 24, 3, 4, 4, 23, 17
# ]

# TODO B-34k-k5
# coordinates = [
#     (28, 57), (76, 46), (67, 5), (84, 22), (73, 6), (67, 72), (68, 74), (68, 6), (76, 7),
#     (91, 30), (80, 0), (0, 25), (73, 13), (76, 81), (92, 30), (69, 80), (90, 30), (71, 77),
#     (83, 47), (79, 47), (74, 6), (68, 6), (72, 0), (80, 0), (91, 25), (71, 73), (78, 10),
#     (85, 24), (75, 80), (87, 24), (0, 7), (71, 78), (74, 0), (76, 0)
# ]
# demands = [
#     0, 6, 12, 2, 24, 3, 18, 21, 14, 69, 1, 13, 2, 2, 7, 7, 1, 23, 19, 14, 8, 11, 4, 8,
#     24, 12, 9, 4, 19, 15, 2, 2, 15, 66
# ]

# TODO B-n50-k8
# coordinates = [
#     (8, 12), (63, 77), (19, 10), (39, 49), (86, 5), (96, 64), (34, 25), (44, 15), (58, 90),
#     (49, 21), (43, 57), (52, 28), (47, 22), (40, 50), (44, 34), (0, 70), (56, 30), (60, 98),
#     (103, 0), (105, 70), (45, 25), (64, 82), (44, 50), (44, 53), (59, 27), (54, 16), (92, 11),
#     (46, 22), (98, 74), (103, 72), (44, 29), (39, 30), (97, 71), (49, 22), (22, 13), (44, 58),
#     (50, 25), (46, 21), (44, 50), (51, 66), (28, 12), (50, 59), (61, 92), (52, 25), (21, 16),
#     (51, 61), (46, 51), (91, 13), (25, 13), (53, 22)
# ]
# demands = [
#     0, 14, 3, 5, 9, 69, 13, 25, 17, 12, 12, 10, 2, 23, 15, 16, 26, 12, 12, 26, 10, 22, 4, 16,
#     8, 23, 2, 24, 12, 24, 4, 19, 21, 7, 15, 14, 18, 7, 20, 18, 2, 21, 21, 3, 5, 20, 16, 25, 3, 10
# ]

# TODO B-n78-k10

# coordinates = [
#     (46, 12), (51, 4), (52, 30), (80, 70), (18, 90), (59, 39), (23, 59), (77, 48), (82, 30), (18, 82),
#     (11, 41), (7, 9), (88, 33), (23, 88), (0, 76), (85, 34), (17, 46), (52, 10), (13, 45), (19, 85),
#     (86, 77), (54, 6), (83, 32), (15, 10), (53, 5), (14, 42), (13, 10), (57, 32), (20, 85), (65, 46),
#     (61, 42), (87, 52), (79, 51), (25, 91), (89, 34), (26, 100), (0, 88), (63, 43), (55, 10), (23, 86),
#     (8, 18), (0, 74), (20, 44), (56, 7), (14, 10), (88, 40), (96, 38), (59, 31), (22, 87), (59, 36),
#     (24, 83), (83, 37), (53, 5), (0, 37), (84, 78), (27, 93), (61, 12), (69, 43), (54, 9), (20, 98),
#     (18, 50), (25, 84), (31, 69), (58, 36), (0, 11), (61, 36), (18, 49), (57, 8), (0, 49), (56, 8),
#     (62, 45), (83, 32), (53, 10), (82, 53), (21, 85), (64, 41), (80, 50), (16, 10)
# ]
#
# demands = [
#     0, 14, 17, 17, 16, 19, 17, 5, 12, 4, 2, 2, 26, 2, 7, 18, 6, 6, 18, 2, 14, 5, 9, 4, 3, 15, 4, 23, 7,
#     21, 4, 1, 6, 16, 4, 20, 5, 14, 14, 26, 5, 2, 14, 11, 21, 20, 18, 2, 19, 12, 22, 14, 23, 25, 8, 3, 9,
#     21, 3, 22, 6, 2, 22, 20, 5, 13, 6, 14, 16, 12, 23, 5, 12, 15, 21, 4, 23, 19
# ]
# TODO P-n16-k8 (n=15, Q=35)
# coordinates = [
#     (30, 40), (37, 52), (49, 49), (52, 64), (31, 62), (52, 33),
#     (42, 41), (52, 41), (57, 58), (62, 42), (42, 57), (27, 68),
#     (43, 67), (58, 48), (58, 27), (37, 69)
# ]
#
# # Спрос (деманд) для каждого узла
# demands = [0, 19, 30, 16, 23, 11, 31, 15, 28, 8, 8, 7, 14, 6, 19, 11]

#TODO P-n19-k2
# coordinates = [
#     (30, 40), (37, 52), (49, 43), (52, 64), (31, 62), (52, 33),
#     (42, 41), (52, 41), (57, 58), (62, 42), (42, 57), (27, 68),
#     (43, 67), (58, 27), (37, 69), (61, 33), (62, 63), (63, 69), (45, 35)
# ]
#
# # Спрос (деманд) для каждого узла
# demands = [0, 19, 30, 16, 23, 11, 31, 15, 28, 14, 8, 7, 14, 19, 11, 26, 17, 6, 15]

# # TODO P-n76-k5
# coordinates = [
#     (40, 40), (22, 22), (36, 26), (21, 45), (45, 35), (55, 20),
#     (33, 34), (50, 50), (55, 45), (26, 59), (40, 66), (55, 65),
#     (35, 51), (62, 35), (62, 57), (62, 24), (21, 36), (33, 44),
#     (9, 56), (62, 48), (66, 14), (44, 13), (26, 13), (11, 28),
#     (7, 43), (17, 64), (41, 46), (55, 34), (35, 16), (52, 26),
#     (43, 26), (31, 76), (22, 53), (26, 29), (50, 40), (55, 50),
#     (54, 10), (60, 15), (47, 66), (30, 60), (30, 50), (12, 17),
#     (15, 14), (16, 19), (21, 48), (50, 30), (51, 42), (50, 15),
#     (48, 21), (12, 38), (15, 56), (29, 39), (54, 38), (55, 57),
#     (67, 41), (10, 70), (6, 25), (65, 27), (40, 60), (70, 64),
#     (64, 4), (36, 6), (30, 20), (20, 30), (15, 5), (50, 70),
#     (57, 72), (45, 42), (38, 33), (50, 4), (66, 8), (59, 5),
#     (35, 60), (27, 24), (40, 20), (40, 37)
# ]
#
# # Спрос (деманд) для каждого узла
# demands = [
#     0, 18, 26, 11, 30, 21, 19, 15, 16, 29, 26, 37, 16, 12, 31, 8, 19, 20,
#     13, 15, 22, 28, 12, 6, 27, 14, 18, 17, 29, 13, 22, 25, 28, 27, 19, 10,
#     12, 14, 24, 16, 33, 15, 11, 18, 17, 21, 27, 19, 20, 5, 22, 12, 19, 22,
#     16, 7, 26, 14, 21, 24, 13, 15, 18, 11, 28, 9, 37, 30, 10, 8, 11, 3, 1,
#     6, 10, 20
# ]

# Расчёт евклидова расстояния между двумя городами
def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Генерация начального индивида
def create_individual(coordinates):
    cities = list(range(2, len(coordinates) + 1))  # Исключаем депо (город 1)
    random.shuffle(cities)
    return cities

# Генерация начальной популяции
def create_population(coordinates):
    return [create_individual(coordinates) for _ in range(population_size)]

# Оценка маршрутов (фитнес-функция)
def calculate_cost(individual, demands, coordinates, capacity):
    routes = []
    route = []
    total_cost = 0
    current_load = 0

    for city in individual:
        city_demand = demands[city - 1]
        if current_load + city_demand <= capacity:
            route.append(city)
            current_load += city_demand
        else:
            routes.append(route)
            route = [city]
            current_load = city_demand

    routes.append(route)

    for route in routes:
        if not route:
            continue
        route_cost = euclidean_distance(coordinates[0], coordinates[route[0] - 1])
        for i in range(len(route) - 1):
            route_cost += euclidean_distance(coordinates[route[i] - 1], coordinates[route[i + 1] - 1])
        route_cost += euclidean_distance(coordinates[route[-1] - 1], coordinates[0])
        total_cost += route_cost

    return total_cost, routes

# Селекция
def select_parents(population, demands, coordinates, capacity):
    return sorted(population, key=lambda x: calculate_cost(x, demands, coordinates, capacity)[0])[:elite_size]

# Кроссовер (порядковое скрещивание)
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

# Мутация
def mutate(individual):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

# Построение маршрутов
def plot_routes(routes, coordinates):
    plt.figure(figsize=(10, 8))
    for i, route in enumerate(routes):
        route_coords = [coordinates[0]] + [coordinates[city - 1] for city in route] + [coordinates[0]]
        x, y = zip(*route_coords)
        plt.plot(x, y, marker='o', label=f"Route #{i + 1}")
    plt.scatter(*zip(*coordinates), color='red', s=50, label='Cities')
    plt.title("Routes")
    plt.legend()
    plt.show()

# Генетический алгоритм
def genetic_algorithm(demands, coordinates, capacity):
    population = create_population(coordinates)
    best_cost = float('inf')
    best_solution = None

    for generation in range(generations):
        population = sorted(population, key=lambda x: calculate_cost(x, demands, coordinates, capacity)[0])
        current_best_cost, current_best_routes = calculate_cost(population[0], demands, coordinates, capacity)
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = population[0]
            best_routes = current_best_routes

        print(f"Generation {generation}: Best Cost = {best_cost:.2f}")

        next_generation = select_parents(population, demands, coordinates, capacity)
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            next_generation.append(child)

        population = next_generation

    return best_cost, best_routes

# Основной запуск
check_disp = []
# capacity = 280
# num_trucks = 8
for test in all_B_set():
    coordinates, demands, capacity, car, answer = test
    best_cost, best_routes = genetic_algorithm(demands, coordinates, capacity)
    print("\nBest Solution:")
    for i, route in enumerate(best_routes):
        route_cost = calculate_cost(route, demands, coordinates, capacity)[0]
        print(f"Route #{i + 1}: {' -> '.join(map(str, route))} | Cost = {route_cost:.2f}")

    print(f"\nTotal Cost: {best_cost:.2f}")
    check_disp.append((best_cost - answer) / best_cost)
    # plot_routes(best_routes)

print(sum(check_disp) / len(check_disp))