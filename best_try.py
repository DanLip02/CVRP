import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from dataset_B import all_B_set
from dataset_P import all_P_set
from dataset_E import all_E_set
import time
# from main_2 import demands, coordinates

# from main_2 import coordinates


random.seed(42)
np.random.seed(42)

population_size = 450
generations = 800
elite_size = 250
mutation_rate = 0.35


def plot(dict_: dict):
    df_avat = pd.DataFrame().from_dict(dict_).T
    for row in df_avat.iterrows():
        print(row[1].index)
        plt.plot(row[1].index, row[1])
    plt.legend(labels=df_avat.index)
    plt.title("Optimization O3")
    plt.xlabel("Num. of nodes")
    plt.ylabel("derivation")
    plt.show()

def generate_dict(dict_: dict):
    res = {}
    for key, value in dict_.items():
        if key not in res:
            res[key] = []
        res[key] = sum(value) / len(value)

    return res

def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def create_individual(coordinates):
    cities = list(range(2, len(coordinates) + 1))
    random.shuffle(cities)
    return cities

def create_population(coordinates):
    return [create_individual(coordinates) for _ in range(population_size)]

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

def select_parents(population, demands, coordinates, capacity):
    return sorted(population, key=lambda x: calculate_cost(x, demands, coordinates, capacity)[0])[:elite_size]

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

def mutate(individual):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
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
timer_E = []
timer_P = []
timer_B = []

dict_plot = {}
local_E = {}
local_B = {}
local_P = {}
# capacity = 280
# num_trucks = 8
counter = 1
for test in all_E_set():
    print(f'Test № {counter}')
    start_time = time.perf_counter()  # Более точное время
    coordinates, demands, capacity, car, answer = test
    best_cost, best_routes = genetic_algorithm(demands, coordinates, capacity)
    print("\nBest Solution:")
    for i, route in enumerate(best_routes):
        route_cost = calculate_cost(route, demands, coordinates, capacity)[0]
        print(f"Route #{i + 1}: {' -> '.join(map(str, route))} | Cost = {route_cost:.2f}")

    print(f"\nTotal Cost: {best_cost:.2f}")
    end_time = time.perf_counter()
    timer_E.append(end_time - start_time)
    if len(coordinates) not in local_E:
        local_E[len(coordinates)] = []
    local_E[len(coordinates)].append((best_cost - answer) / best_cost)
    check_disp.append((best_cost - answer) / best_cost)
    counter += 1
    # plot_routes(best_routes)
dict_plot['Set_E'] = generate_dict(local_E)

for test in all_P_set():
    start_time = time.perf_counter()  # Более точное время
    print(f'Test № {counter}')
    coordinates, demands, capacity, car, answer = test
    best_cost, best_routes = genetic_algorithm(demands, coordinates, capacity)
    print("\nBest Solution:")
    for i, route in enumerate(best_routes):
        route_cost = calculate_cost(route, demands, coordinates, capacity)[0]
        print(f"Route #{i + 1}: {' -> '.join(map(str, route))} | Cost = {route_cost:.2f}")

    print(f"\nTotal Cost: {best_cost:.2f}")
    end_time = time.perf_counter()
    timer_P.append(end_time - start_time)
    if len(coordinates) not in local_P:
        local_P[len(coordinates)] = []
    local_P[len(coordinates)].append((best_cost - answer) / best_cost)
    check_disp.append((best_cost - answer) / best_cost)
    counter += 1

dict_plot['Set_P'] = generate_dict(local_P)

for test in all_B_set():
    print(f'Test № {counter}')
    start_time = time.perf_counter()
    coordinates, demands, capacity, car, answer = test
    best_cost, best_routes = genetic_algorithm(demands, coordinates, capacity)
    print("\nBest Solution:")
    for i, route in enumerate(best_routes):
        route_cost = calculate_cost(route, demands, coordinates, capacity)[0]
        print(f"Route #{i + 1}: {' -> '.join(map(str, route))} | Cost = {route_cost:.2f}")

    print(f"\nTotal Cost: {best_cost:.2f}")
    end_time = time.perf_counter()
    timer_B.append(end_time - start_time)
    if len(coordinates) not in local_B:
        local_B[len(coordinates)] = []
    local_B[len(coordinates)].append((best_cost - answer) / best_cost)
    check_disp.append((best_cost - answer) / best_cost)
    counter += 1
dict_plot['Set_B'] = generate_dict(local_B)
print("Avarage deviation: ", sum(check_disp) / len(check_disp))
print("Time for all test from set B: ", sum(timer_B) / len(timer_B))
print("Time for all test from set P: ", sum(timer_P) / len(timer_P))
print("Time for all test from set E: ", sum(timer_E) / len(timer_E))

plot(dict_plot)
