import numpy as np
import random
import math
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# Входные данные
# coordinates = [
#     (58, 90), (5, 66), (34, 98), (93, 17), (23, 67), (71, 91), (20, 39), (20, 86),
#     (7, 99), (72, 78), (27, 90), (14, 72), (72, 101), (72, 96), (19, 73), (76, 95),
#     (30, 92), (30, 87), (40, 104), (7, 73), (7, 76), (79, 86), (82, 105), (12, 104),
#     (21, 93), (77, 85), (37, 101), (28, 91), (72, 95), (22, 92), (98, 18), (42, 107),
#     (16, 101), (41, 100), (23, 41), (8, 70), (22, 45), (15, 103), (27, 88), (25, 47),
#     (26, 42), (29, 93), (25, 45), (29, 91), (26, 44), (78, 109), (27, 69), (40, 99),
#     (74, 81), (8, 69), (18, 74), (24, 96), (82, 107), (27, 46), (81, 95), (79, 95),
#     (76, 95), (97, 19), (74, 98), (21, 94), (40, 0), (100, 23), (99, 24), (73, 85),
#     (26, 40), (16, 103), (33, 94)
# ]
coordinates = [
    (35, 31), (77, 13), (95, 51), (91, 15), (65, 93), (51, 17), (41, 57), (39, 39),
    (70, 102), (74, 94), (81, 97), (77, 95), (84, 14), (48, 60), (98, 54), (92, 18),
    (42, 42), (52, 18), (44, 42), (48, 44), (80, 22), (42, 58), (73, 105), (81, 101),
    (102, 58), (82, 18), (96, 58), (98, 18), (84, 16), (44, 44), (73, 103), (96, 16),
    (46, 48), (92, 16), (44, 64), (98, 22), (96, 52), (46, 64), (77, 103), (104, 54),
    (58, 20), (46, 64), (73, 103), (94, 16)
]
demands = [
    0, 26, 11, 22, 18, 10, 9, 12, 10, 12, 19, 5, 18, 7, 14, 14, 20, 7, 4, 11,
    3, 22, 15, 69, 26, 20, 12, 22, 9, 23, 7, 17, 3, 18, 18, 9, 4, 20, 24, 3,
    4, 4, 23, 17
]
num_cities = 44
num_trucks = 7
capacity = 100

# Предварительное вычисление расстояний
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        dist = math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2)
        distance_matrix[i][j] = distance_matrix[j][i] = dist

# Генерация начальной популяции
def create_individual():
    return list(np.random.permutation(num_cities - 1))  # исключаем депо (город 0)

def create_population(pop_size=100):
    return [create_individual() for _ in range(pop_size)]

# Функция оценки маршрутов
def calculate_cost(individual):
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
        route_cost = 0
        for i in range(len(route) - 1):
            route_cost += distance_matrix[route[i] - 1][route[i + 1] - 1]
        route_cost += distance_matrix[route[-1] - 1][route[0] - 1]
        total_cost += route_cost

    return total_cost, routes

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
    if random.random() < 0.2:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

def plot_routes(routes, coordinates):
    plt.figure(figsize=(10, 8))

    for city_id, (x, y) in enumerate(coordinates, start=1):
        plt.scatter(x, y, c='blue', label=f"City {city_id}" if city_id % 10 == 0 else "")
        plt.text(x, y, f" {city_id}", fontsize=9, ha='right')

    for i, route in enumerate(routes):
        full_route = [1] + route + [1]
        route_coords = [coordinates[city - 1] for city in full_route]
        route_coords.append(route_coords[0])
        route_x, route_y = zip(*route_coords)

        plt.plot(route_x, route_y, marker='o', markersize=5, label=f"Route #{i + 1}")

    plt.title("Vehicle Routing Problem (CVRP) Routes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Основной алгоритм
population = create_population()
best_cost = float('inf')
best_solution = None
best_routes = None
no_improvement_count = 0

for generation in range(100):
    population = sorted(population, key=lambda x: calculate_cost(x)[0])
    current_best_cost, current_best_routes = calculate_cost(population[0])

    if current_best_cost < best_cost:
        best_cost = current_best_cost
        best_solution = population[0]
        best_routes = current_best_routes
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    print(f"Generation {generation}: Best Cost = {best_cost}")

    if no_improvement_count >= 9:
        print(f"Stopping at generation {generation} due to no improvements.")
        break

    next_generation = population[:50]

    while len(next_generation) < 100:
        parents = random.sample(population[:50], 2)
        child = crossover(parents[0], parents[1])
        mutate(child)
        next_generation.append(child)

    population = next_generation

# Вывод лучшего решения
print("\nBest Solution:")
for i, route in enumerate(best_routes):
    print(f"Route #{i + 1}: {' '.join(map(str, route))}")

print(f"Total Cost: {best_cost:.2f}")

plot_routes(best_routes, coordinates)
