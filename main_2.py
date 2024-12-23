import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Устанавливаем фиксированный seed для повторяемости результатов
random.seed(42)
np.random.seed(42)
# Входные данные
coordinates = [
    (58, 90), (5, 66), (34, 98), (93, 17), (23, 67), (71, 91), (20, 39), (20, 86),
    (7, 99), (72, 78), (27, 90), (14, 72), (72, 101), (72, 96), (19, 73), (76, 95),
    (30, 92), (30, 87), (40, 104), (7, 73), (7, 76), (79, 86), (82, 105), (12, 104),
    (21, 93), (77, 85), (37, 101), (28, 91), (72, 95), (22, 92), (98, 18), (42, 107),
    (16, 101), (41, 100), (23, 41), (8, 70), (22, 45), (15, 103), (27, 88), (25, 47),
    (26, 42), (29, 93), (25, 45), (29, 91), (26, 44), (78, 109), (27, 69), (40, 99),
    (74, 81), (8, 69), (18, 74), (24, 96), (82, 107), (27, 46), (81, 95), (79, 95),
    (76, 95), (97, 19), (74, 98), (21, 94), (40, 0), (100, 23), (99, 24), (73, 85),
    (26, 40), (16, 103), (33, 94)
]
demands = [
    0, 12, 12, 10, 2, 23, 15, 16, 26, 12, 12, 26, 10, 22, 4, 16, 8, 23, 2, 24, 12,
    24, 4, 19, 21, 7, 15, 14, 18, 7, 20, 18, 2, 21, 21, 3, 5, 20, 16, 25, 3, 10, 18,
    4, 14, 6, 13, 25, 17, 15, 19, 21, 15, 25, 7, 4, 16, 24, 16, 10, 15, 14, 25, 6, 16
]
num_cities = len(coordinates)
num_trucks = 10
capacity = 100


# Функция для вычисления евклидова расстояния
def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# Генерация начальной популяции
def create_individual():
    return list(np.random.permutation(num_cities - 1))  # исключаем депо (город 0)


def create_population():
    return [create_individual() for _ in range(100)]


# Функция оценки маршрутов
def calculate_cost(individual):
    routes = []
    route = []
    total_cost = 0
    current_load = 0

    # Разбиение на маршруты для каждого грузовика
    for city in individual:
        city_demand = demands[city - 1]  # Индексы городов начинаются с 1, поэтому требуемый индекс - city - 1
        if current_load + city_demand <= capacity:  # Если грузовик не перегружен
            route.append(city)
            current_load += city_demand
        else:
            routes.append(route)
            route = [city]  # Новый маршрут
            current_load = city_demand

    routes.append(route)  # Добавление последнего маршрута

    # Считаем стоимость всех маршрутов
    for route in routes:
        route_cost = 0
        # Считаем стоимость маршрута (расстояние между городами)
        for i in range(len(route) - 1):
            route_cost += euclidean_distance(coordinates[route[i] - 1],
                                             coordinates[route[i + 1] - 1])  # Индексы городов сдвигаем на 1
        route_cost += euclidean_distance(coordinates[route[-1] - 1], coordinates[route[0] - 1])  # Замыкаем маршрут
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
    # Создаем график
    plt.figure(figsize=(10, 8))

    # Отображаем все города
    for city_id, (x, y) in enumerate(coordinates, start=1):
        plt.scatter(x, y, c='blue', label=f"City {city_id}" if city_id % 10 == 0 else "")
        plt.text(x, y, f" {city_id}", fontsize=9, ha='right')

    # Отображаем маршруты
    for i, route in enumerate(routes):
        # Убедимся, что каждый маршрут начинается и заканчивается в городе 1
        full_route = [1] + route + [1]
        route_coords = [coordinates[city - 1] for city in full_route]
        route_coords.append(route_coords[0])  # Замыкаем маршрут

        route_x, route_y = zip(*route_coords)

        # Для каждого маршрута (грузовика) добавляем уникальную метку в легенду
        plt.plot(route_x, route_y, marker='o', markersize=5, label=f"Route #{i + 1}")

    plt.title("Vehicle Routing Problem (CVRP) Routes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()  # Легенда с маршрутами для каждой машины
    plt.grid(True)
    plt.show()

# Основной алгоритм
population = create_population()
best_cost = float('inf')
best_solution = None
best_routes = None
no_improvement_count = 0

for generation in range(100):
    # Оценка популяции
    population = sorted(population, key=lambda x: calculate_cost(x)[0])  # Сортируем по стоимости
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
        print(f"Остановка на поколении {generation} из-за отсутствия улучшений.")
        break

    # Селекция
    next_generation = population[:50]

    # Генерация потомков
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
