
def test_1():
    # Исходные данные из задачи B-n31-k5
    # Координаты узлов
    coordinates = [
        (17, 76), (24, 6), (96, 29), (14, 19), (14, 32), (0, 34), (16, 22),
        (20, 26), (22, 28), (17, 23), (98, 30), (30, 8), (23, 27), (19, 23),
        (34, 7), (31, 7), (0, 37), (19, 23), (0, 36), (26, 7), (98, 32),
        (5, 40), (17, 26), (21, 26), (28, 8), (1, 35), (27, 28), (99, 30),
        (26, 28), (17, 29), (20, 26)
    ]

    # Спрос (деманд) для каждого узла
    demands = [
        0, 25, 3, 13, 17, 16, 9, 22, 10, 16, 8, 3, 16, 16, 10, 24, 16, 15,
        14, 5, 12, 2, 18, 20, 15, 8, 22, 15, 10, 13, 19
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 672
    car = 5
    return coordinates, demands, capacity, car, answer

def test_2():
    # Исходные данные из задачи B-n31-k5
    # Координаты узлов
    # Координаты узлов
    coordinates = [
        (28, 57), (76, 46), (67, 5), (84, 22), (73, 6), (67, 72), (68, 74),
        (68, 6), (76, 7), (91, 30), (80, 0), (0, 25), (73, 13), (76, 81),
        (92, 30), (69, 80), (90, 30), (71, 77), (83, 47), (79, 47), (74, 6),
        (68, 6), (72, 0), (80, 0), (91, 25), (71, 73), (78, 10), (85, 24),
        (75, 80), (87, 24), (0, 7), (71, 78), (74, 0), (76, 0)
    ]

    # Спрос (деманд) для каждого узла
    demands = [
        0, 6, 12, 2, 24, 3, 18, 21, 14, 69, 1, 13, 2, 2, 7, 7, 1, 23,
        19, 14, 8, 11, 4, 8, 24, 12, 9, 4, 19, 15, 2, 2, 15, 66
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 788
    car = 5
    return coordinates, demands, capacity, car, answer

def test_3():
    # Исходные данные из задачи B-n35-k5

    # Координаты узлов и спрос
    coordinates = [
        (78, 95), (93, 43), (57, 4), (2, 80), (10, 17), (31, 8), (10, 87),
        (97, 50), (16, 93), (98, 48), (103, 47), (38, 9), (100, 51), (60, 11),
        (15, 19), (39, 15), (102, 47), (103, 59), (10, 82), (39, 9), (97, 52),
        (18, 97), (32, 13), (96, 45), (11, 21), (15, 96), (10, 81), (13, 24),
        (0, 8), (103, 59), (33, 11), (13, 94), (63, 5), (3, 87), (14, 25)
    ]

    demands = [
        0, 12, 3, 2, 13, 17, 12, 1, 26, 13, 15, 20, 20, 3, 3, 12, 25, 2, 15, 24,
        2, 7, 15, 2, 13, 9, 12, 26, 17, 26, 9, 14, 9, 25, 13
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 955
    car = 5
    return coordinates, demands, capacity, car, answer

def test_4():
    # Исходные данные из задачи B-n38-k6

    # Координаты узлов и спрос
    coordinates = [
        (64, 75), (16, 97), (2, 79), (28, 79), (16, 3), (39, 20), (35, 65), (43, 21), (44, 0),
        (37, 67), (17, 100), (34, 80), (45, 7), (11, 83), (35, 88), (21, 98), (41, 72), (51, 0),
        (17, 9), (36, 66), (20, 105), (35, 85), (37, 75), (17, 11), (46, 6), (40, 23), (43, 73),
        (40, 21), (29, 85), (45, 6), (0, 24), (42, 66), (30, 81), (17, 9), (46, 4), (0, 85),
        (51, 2), (5, 80)
    ]

    demands = [
        0, 18, 10, 16, 12, 21, 23, 15, 25, 3, 6, 7, 4, 20, 25, 20, 12, 3, 12, 14, 26, 9, 22,
        20, 9, 13, 21, 10, 8, 5, 14, 7, 21, 20, 1, 21, 17, 2
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 805
    car = 6

    return coordinates, demands, capacity, car, answer


def test_5():
    # Исходные данные из задачи B-n39-k5

    # Координаты узлов и спрос
    coordinates = [
        (37, 21), (77, 57), (97, 79), (39, 33), (45, 47), (85, 23), (7, 1), (16, 6), (21, 13),
        (21, 7), (12, 6), (92, 24), (92, 32), (10, 8), (19, 15), (86, 24), (106, 84), (48, 48),
        (8, 2), (14, 2), (14, 2), (98, 82), (106, 80), (98, 86), (98, 82), (80, 62), (82, 62),
        (52, 52), (42, 42), (25, 11), (104, 86), (44, 38), (78, 64), (98, 80), (16, 4), (46, 48),
        (44, 42), (100, 82), (52, 54)
    ]

    demands = [
        0, 14, 16, 18, 20, 1, 12, 13, 18, 9, 15, 8, 10, 7, 18, 14, 1, 15, 7, 25, 2, 3, 4, 16,
        15, 23, 11, 21, 10, 12, 9, 7, 8, 4, 23, 8, 6, 2, 15
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 549
    car = 5

    return coordinates, demands, capacity, car, answer


def test_6():
    # Исходные данные из задачи B-n41-k6

    # Координаты узлов и спрос
    coordinates = [
        (37, 35), (61, 78), (18, 33), (10, 7), (48, 26), (96, 58), (16, 97), (102, 67), (97, 67),
        (20, 17), (110, 77), (26, 101), (102, 74), (98, 64), (71, 82), (101, 60), (104, 69), (70, 85),
        (17, 9), (56, 34), (105, 69), (20, 8), (103, 63), (105, 76), (25, 35), (28, 43), (58, 36),
        (56, 32), (65, 87), (70, 80), (100, 65), (102, 61), (99, 73), (24, 103), (26, 107), (21, 39),
        (99, 71), (20, 16), (56, 30), (51, 31), (106, 70)
    ]

    demands = [
        0, 6, 11, 14, 7, 12, 16, 6, 18, 7, 20, 7, 23, 16, 10, 9, 18, 8, 15, 9, 23, 7, 15, 14,
        21, 25, 14, 11, 19, 12, 8, 25, 17, 22, 11, 16, 16, 8, 23, 15, 13
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 829
    car = 6

    return coordinates, demands, capacity, car, answer


def test_7():
    # Исходные данные из задачи B-n43-k6

    # Координаты узлов и спрос
    coordinates = [
        (74, 34), (9, 22), (13, 28), (47, 63), (74, 43), (71, 10), (11, 61), (23, 33), (14, 29),
        (31, 34), (22, 29), (50, 66), (16, 30), (48, 70), (16, 0), (18, 33), (20, 34), (26, 36),
        (49, 0), (19, 0), (80, 17), (75, 13), (18, 68), (10, 27), (12, 66), (80, 50), (29, 43),
        (79, 48), (14, 0), (76, 51), (54, 68), (17, 30), (78, 50), (72, 19), (23, 37), (78, 45),
        (14, 34), (0, 15), (16, 31), (24, 37), (84, 47), (80, 44), (24, 41)
    ]

    demands = [
        0, 25, 19, 8, 10, 1, 14, 11, 20, 23, 3, 4, 6, 23, 7, 15, 13, 3, 11, 24, 8, 14, 8, 9,
        24, 10, 19, 10, 5, 7, 23, 11, 12, 4, 11, 16, 22, 24, 12, 5, 11, 7, 9
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 742
    car = 6

    return coordinates, demands, capacity, car, answer

def test_8():
    # Исходные данные из задачи B-n44-k7
    coordinates = [
        (35, 31), (77, 13), (95, 51), (91, 15), (65, 93), (51, 17), (41, 57), (39, 39),
        (70, 102), (74, 94), (81, 97), (77, 95), (84, 14), (48, 60), (98, 54), (92, 18),
        (42, 42), (52, 18), (44, 42), (48, 44), (80, 22), (42, 58), (73, 105), (81, 101),
        (102, 58), (82, 18), (96, 58), (98, 18), (84, 16), (44, 44), (73, 103), (96, 16),
        (46, 48), (92, 16), (44, 64), (98, 22), (96, 52), (46, 64), (77, 103), (104, 54),
        (58, 20), (46, 64), (73, 103), (94, 16)
    ]
    demands = [
        0, 26, 11, 22, 18, 10, 9, 12, 10, 12, 19, 5, 18, 7, 14, 14, 20, 7, 4, 11, 3,
        22, 15, 69, 26, 20, 12, 22, 9, 23, 7, 17, 3, 18, 18, 9, 4, 20, 24, 3, 4, 4, 23, 17
    ]

    capacity = 100
    car = 7
    answer = 909
    return coordinates, demands, capacity, car, answer

def test_9():
    # Исходные данные из задачи B-n45-k5

    # Координаты узлов и спрос
    coordinates = [
        (53, 22), (34, 28), (2, 5), (40, 85), (88, 38), (74, 20), (82, 21), (0, 46), (84, 31),
        (11, 12), (42, 37), (90, 44), (85, 22), (9, 8), (4, 51), (3, 10), (90, 40), (41, 33),
        (10, 50), (96, 45), (48, 90), (87, 31), (79, 26), (39, 32), (0, 91), (89, 45), (91, 46),
        (3, 53), (44, 0), (89, 41), (40, 32), (42, 86), (0, 13), (97, 45), (1, 50), (45, 94),
        (36, 33), (4, 15), (42, 88), (42, 29), (92, 0), (75, 26), (78, 0), (77, 29), (5, 47)
    ]

    demands = [
        0, 1, 19, 19, 22, 20, 11, 2, 5, 20, 22, 2, 2, 11, 22, 19, 3, 1, 2, 16, 2, 13, 7, 8,
        16, 14, 4, 14, 7, 20, 14, 7, 9, 7, 5, 10, 13, 25, 1, 22, 9, 3, 8, 10, 19
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 751
    car = 5

    return coordinates, demands, capacity, car, answer

def test_10():
    # Исходные данные из задачи B-n45-k6

    # Координаты узлов и спрос
    coordinates = [
        (49, 64), (60, 38), (38, 21), (98, 27), (69, 60), (59, 40), (82, 28), (86, 36), (76, 65),
        (102, 29), (43, 26), (45, 28), (76, 68), (64, 45), (0, 44), (89, 31), (60, 44), (0, 44),
        (84, 29), (83, 66), (86, 36), (103, 32), (76, 65), (86, 74), (0, 48), (66, 41), (67, 40),
        (77, 61), (78, 66), (66, 46), (61, 41), (105, 31), (78, 68), (91, 37), (83, 29), (91, 45),
        (42, 25), (83, 69), (101, 36), (74, 65), (93, 37), (107, 34), (63, 40), (61, 47), (90, 43)
    ]

    demands = [
        0, 21, 13, 24, 10, 22, 16, 9, 9, 18, 15, 3, 7, 14, 22, 10, 13, 3, 4, 14, 2, 16, 10, 5,
        11, 19, 14, 18, 12, 21, 21, 24, 6, 7, 13, 15, 5, 10, 19, 6, 21, 19, 17, 26, 8
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 678
    car = 6

    return coordinates, demands, capacity, car, answer

def test_11():
    # Исходные данные из задачи B-n50-k7

    # Координаты узлов и спрос
    coordinates = [
        (49, 53), (59, 1), (17, 83), (85, 57), (47, 21), (1, 21), (25, 69), (75, 63), (3, 7),
        (56, 26), (86, 58), (8, 8), (59, 27), (64, 2), (86, 64), (28, 72), (88, 58), (18, 90),
        (82, 64), (22, 92), (6, 10), (10, 24), (59, 29), (52, 24), (94, 62), (76, 68), (66, 2),
        (90, 58), (20, 84), (50, 22), (76, 64), (63, 33), (20, 84), (59, 31), (32, 74), (48, 24),
        (2, 30), (10, 8), (57, 27), (68, 6), (28, 74), (63, 35), (86, 58), (90, 62), (22, 90),
        (6, 28), (62, 8), (59, 35), (18, 88), (30, 76)
    ]

    demands = [
        0, 21, 8, 11, 7, 21, 5, 13, 10, 9, 20, 7, 12, 23, 2, 4, 14, 12, 3, 5, 13, 5, 12, 2,
        3, 18, 24, 4, 63, 19, 2, 9, 4, 9, 23, 6, 3, 12, 7, 17, 22, 26, 14, 9, 2, 16, 24, 4,
        19, 11
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 741
    car = 7

    return coordinates, demands, capacity, car, answer

def test_12():
    # Исходные данные из задачи B-n50-k8

    # Координаты узлов и спрос
    coordinates = [
        (8, 12), (63, 77), (19, 10), (39, 49), (86, 5), (96, 64), (34, 25), (44, 15), (58, 90),
        (49, 21), (43, 57), (52, 28), (47, 22), (40, 50), (44, 34), (0, 70), (56, 30), (60, 98),
        (103, 0), (105, 70), (45, 25), (64, 82), (44, 50), (44, 53), (59, 27), (54, 16), (92, 11),
        (46, 22), (98, 74), (103, 72), (44, 29), (39, 30), (97, 71), (49, 22), (22, 13), (44, 58),
        (50, 25), (46, 21), (44, 50), (51, 66), (28, 12), (50, 59), (61, 92), (52, 25), (21, 16),
        (51, 61), (46, 51), (91, 13), (25, 13), (53, 22)
    ]

    demands = [
        0, 14, 3, 5, 9, 69, 13, 25, 17, 12, 12, 10, 2, 23, 15, 16, 26, 12, 12, 26, 10, 22, 4, 16,
        8, 23, 2, 24, 12, 24, 4, 19, 21, 7, 15, 14, 18, 7, 20, 18, 2, 21, 21, 3, 5, 20, 16, 25,
        3, 10
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1312
    car = 8

    return coordinates, demands, capacity, car, answer


def test_13():
    # Исходные данные из задачи B-n51-k7

    # Координаты узлов и спрос
    coordinates = [
        (53, 55), (1, 90), (94, 85), (13, 19), (97, 45), (78, 69), (92, 35), (98, 36), (99, 0), (2, 94),
        (85, 77), (94, 42), (101, 3), (103, 51), (106, 38), (9, 95), (88, 71), (83, 77), (100, 55), (17, 23),
        (3, 91), (6, 91), (9, 102), (9, 95), (8, 104), (9, 99), (98, 39), (101, 8), (99, 38), (103, 54), (3, 103),
        (101, 7), (103, 8), (20, 25), (97, 93), (0, 51), (81, 70), (103, 7), (95, 87), (102, 54), (83, 72), (101, 10),
        (105, 48), (96, 40), (5, 91), (9, 96), (10, 100), (9, 96), (22, 21), (2, 95), (9, 95)
    ]

    demands = [
        0, 9, 10, 14, 5, 8, 10, 15, 16, 23, 4, 22, 16, 12, 20, 18, 9, 17, 42, 9, 17, 7, 4, 7, 13, 6, 22, 6,
        13, 21, 16, 20, 11, 18, 24, 26, 9, 21, 3, 22, 7, 10, 17, 8, 10, 12, 10, 24, 5, 10, 6
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1032
    car = 7

    return coordinates, demands, capacity, car, answer

def test_14():
    # Исходные данные из задачи B-n52-k7

    # Координаты узлов и спрос
    coordinates = [
        (29, 33), (41, 11), (31, 87), (91, 27), (53, 87), (7, 19), (27, 19), (1, 41), (8, 20), (40, 92),
        (14, 20), (92, 32), (62, 94), (45, 97), (96, 36), (47, 101), (40, 92), (49, 97), (48, 12), (45, 101),
        (46, 16), (92, 34), (49, 95), (62, 92), (96, 28), (28, 22), (47, 95), (9, 21), (92, 30), (12, 22), (48, 12),
        (94, 28), (43, 99), (8, 42), (43, 99), (8, 42), (30, 20), (16, 24), (36, 92), (98, 34), (36, 28), (28, 26),
        (42, 18), (2, 50), (16, 20), (56, 90), (45, 97), (17, 23), (34, 88), (47, 99), (62, 96), (8, 24)
    ]

    demands = [
        0, 22, 8, 3, 6, 10, 18, 8, 13, 13, 3, 10, 6, 23, 8, 4, 13, 6, 5, 10, 22, 13, 13, 18, 14, 14, 26, 23,
        4, 25, 9, 8, 16, 3, 1, 17, 9, 15, 2, 19, 7, 6, 22, 22, 10, 7, 4, 23, 13, 10, 8, 14
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 747
    car = 7

    return coordinates, demands, capacity, car, answer


def test_15():
    # Исходные данные из задачи B-n56-k7

    # Координаты узлов и спрос
    coordinates = [
        (87, 45), (93, 41), (75, 95), (89, 55), (89, 19), (21, 9), (81, 31), (79, 25), (30, 18), (5, 5),
        (31, 21), (84, 98), (10, 8), (90, 58), (24, 16), (80, 26), (96, 42), (28, 18), (37, 23), (39, 27),
        (78, 104), (98, 26), (8, 12), (10, 12), (82, 30), (82, 38), (94, 24), (94, 44), (30, 16), (84, 28),
        (76, 96), (76, 102), (98, 58), (90, 64), (76, 96), (100, 44), (94, 56), (84, 26), (6, 12), (90, 24),
        (26, 12), (84, 30), (86, 28), (31, 27), (24, 16), (84, 30), (33, 19), (90, 28), (94, 50), (92, 60),
        (8, 12), (24, 16), (94, 42), (82, 32), (76, 96), (80, 100)
    ]

    demands = [
        0, 10, 15, 7, 25, 16, 8, 15, 23, 8, 2, 14, 7, 15, 14, 7, 14, 11, 7, 6, 1, 22, 26, 5, 18, 18, 4, 10,
        3, 2, 12, 6, 15, 12, 5, 7, 12, 24, 6, 12, 5, 18, 3, 7, 15, 23, 20, 5, 24, 6, 19, 5, 10, 3, 4, 5
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 707
    car = 7

    return coordinates, demands, capacity, car, answer

def test_16():
    # Исходные данные из задачи B-n57-k7

    # Координаты узлов и спрос
    coordinates = [
        (11, 83), (37, 61), (77, 81), (35, 21), (1, 93), (21, 39), (63, 7), (97, 95), (3, 7), (5, 3),
        (36, 22), (64, 10), (6, 8), (24, 40), (86, 82), (68, 14), (28, 46), (10, 8), (8, 14), (10, 10),
        (104, 104), (38, 24), (80, 82), (22, 46), (26, 40), (4, 10), (10, 12), (26, 44), (4, 10), (4, 98),
        (78, 82), (64, 8), (98, 98), (24, 40), (10, 8), (8, 96), (8, 8), (84, 90), (8, 8), (86, 86), (6, 94),
        (10, 10), (80, 82), (102, 96), (36, 26), (44, 28), (22, 46), (100, 100), (28, 42), (30, 40), (78, 84),
        (10, 4), (78, 84), (6, 8), (10, 94), (40, 66), (10, 16)
    ]

    demands = [
        0, 2, 1, 3, 9, 13, 21, 6, 22, 10, 23, 2, 15, 10, 23, 8, 5, 14, 6, 15, 18, 15, 10, 17, 1, 1, 18,
        7, 12, 9, 26, 60, 12, 17, 9, 7, 13, 24, 10, 17, 24, 25, 1, 20, 14, 6, 9, 11, 2, 11, 17, 3, 1, 8,
        12, 9, 13
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1153
    car = 7

    return coordinates, demands, capacity, car, answer


def test_17():
    # Исходные данные из задачи B-n57-k9

    # Координаты узлов и спрос
    coordinates = [
        (19, 1), (83, 61), (59, 95), (73, 25), (51, 53), (23, 49), (63, 19), (51, 81), (1, 89), (97, 41),
        (74, 30), (60, 88), (58, 54), (63, 95), (65, 89), (6, 94), (61, 97), (75, 31), (106, 48), (69, 93),
        (56, 84), (26, 54), (66, 24), (60, 98), (92, 66), (8, 92), (64, 102), (104, 48), (104, 42), (30, 52),
        (52, 54), (77, 33), (68, 20), (56, 62), (65, 91), (54, 60), (26, 52), (52, 90), (56, 56), (66, 102),
        (72, 24), (28, 56), (58, 82), (84, 66), (63, 93), (88, 64), (74, 28), (26, 50), (58, 58), (52, 56),
        (76, 26), (8, 92), (90, 68), (32, 50), (84, 66), (78, 34), (64, 24)
    ]

    demands = [
        0, 18, 16, 23, 20, 18, 19, 2, 4, 10, 2, 26, 21, 10, 8, 20, 21, 4, 5, 15, 18, 2, 21, 20, 23, 13, 22,
        17, 11, 7, 13, 13, 16, 14, 25, 18, 2, 19, 6, 19, 9, 24, 2, 13, 17, 20, 24, 8, 15, 7, 7, 4, 22, 11,
        10, 25, 24
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1598
    car = 9

    return coordinates, demands, capacity, car, answer


def test_18():
    # Исходные данные из задачи B-n63-k10

    # Координаты узлов и спрос
    coordinates = [
        (73, 89), (97, 22), (19, 31), (23, 95), (18, 89), (60, 94), (9, 26), (71, 7), (10, 80), (27, 64),
        (46, 32), (26, 0), (8, 8), (63, 101), (16, 10), (30, 69), (26, 103), (20, 0), (99, 32), (16, 33),
        (79, 13), (18, 30), (27, 41), (34, 4), (19, 0), (12, 33), (11, 28), (101, 0), (69, 98), (69, 102),
        (15, 81), (26, 90), (65, 100), (14, 29), (21, 40), (10, 29), (81, 10), (53, 37), (31, 9), (21, 36),
        (24, 98), (15, 14), (28, 36), (19, 0), (30, 102), (81, 15), (12, 86), (30, 65), (66, 99), (17, 89),
        (102, 29), (16, 31), (17, 33), (32, 8), (11, 11), (61, 102), (16, 88), (29, 9), (15, 9), (14, 85),
        (63, 95), (29, 32), (19, 32)
    ]

    demands = [
        0, 18, 11, 18, 17, 21, 6, 12, 6, 21, 14, 2, 5, 12, 23, 23, 19, 18, 8, 21, 7, 19, 18, 7, 6, 12, 21,
        6, 21, 20, 18, 6, 20, 20, 23, 4, 21, 4, 6, 9, 21, 5, 20, 11, 16, 48, 2, 15, 15, 22, 25, 23, 5, 24,
        11, 24, 20, 21, 10, 24, 3, 9, 5
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1496
    car = 10

    return coordinates, demands, capacity, car, answer


def test_19():
    # Исходные данные из задачи B-n56-k7

    # Координаты узлов и спрос
    coordinates = [
        (87, 45), (93, 41), (75, 95), (89, 55), (89, 19), (21, 9), (81, 31), (79, 25), (30, 18), (5, 5),
        (31, 21), (84, 98), (10, 8), (90, 58), (24, 16), (80, 26), (96, 42), (28, 18), (37, 23), (39, 27),
        (78, 104), (98, 26), (8, 12), (10, 12), (82, 30), (82, 38), (94, 24), (94, 44), (30, 16), (84, 28),
        (76, 96), (76, 102), (98, 58), (90, 64), (76, 96), (100, 44), (94, 56), (84, 26), (6, 12), (90, 24),
        (26, 12), (84, 30), (86, 28), (31, 27), (24, 16), (84, 30), (33, 19), (90, 28), (94, 50), (92, 60),
        (8, 12), (24, 16), (94, 42), (82, 32), (76, 96), (80, 100)
    ]

    demands = [
        0, 10, 15, 7, 25, 16, 8, 15, 23, 8, 2, 14, 7, 15, 14, 7, 14, 11, 7, 6, 1, 22, 26, 5, 18, 18, 4, 10,
        3, 2, 12, 6, 15, 12, 5, 7, 12, 24, 6, 12, 5, 18, 3, 7, 15, 23, 20, 5, 24, 6, 19, 5, 10, 3, 4, 5
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 707
    car = 7

    return coordinates, demands, capacity, car, answer

def test_20():
    # Исходные данные из задачи B-n64-k9

    # Координаты узлов и спрос
    coordinates = [
        (59, 73), (79, 65), (73, 27), (41, 81), (11, 93), (13, 89), (11, 15), (71, 81), (77, 89), (41, 87),
        (44, 94), (51, 101), (42, 92), (58, 104), (14, 94), (18, 90), (60, 102), (72, 86), (45, 95), (20, 90),
        (74, 32), (78, 90), (18, 98), (78, 36), (16, 90), (78, 36), (80, 74), (78, 92), (52, 108), (49, 97),
        (72, 82), (72, 88), (42, 82), (50, 92), (82, 94), (16, 94), (18, 94), (42, 88), (78, 94), (80, 32),
        (84, 96), (76, 82), (78, 30), (88, 68), (18, 20), (18, 20), (74, 82), (47, 95), (18, 94), (86, 68),
        (45, 95), (14, 102), (82, 66), (14, 16), (76, 86), (18, 102), (58, 106), (76, 34), (18, 18), (14, 96),
        (18, 94), (46, 96), (82, 70), (76, 28)
    ]

    demands = [
        0, 24, 15, 2, 3, 24, 17, 9, 4, 2, 15, 20, 17, 15, 20, 8, 6, 2, 6, 18, 5, 16, 2, 18, 1, 13, 17, 5,
        14, 19, 3, 22, 6, 9, 22, 6, 18, 23, 24, 4, 20, 16, 15, 24, 16, 9, 4, 11, 3, 54, 36, 19, 11, 21, 12,
        9, 17, 16, 14, 14, 21, 19, 6, 17
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 861
    car = 9

    return coordinates, demands, capacity, car, answer

def test_21():
    # Исходные данные из задачи B-n66-k9

    # Координаты узлов и спрос
    coordinates = [
        (41, 88), (53, 83), (87, 28), (42, 11), (22, 1), (34, 36), (84, 33), (81, 63), (11, 29), (62, 27),
        (90, 30), (84, 69), (69, 33), (89, 72), (45, 16), (28, 6), (89, 35), (58, 88), (62, 87), (88, 67),
        (57, 90), (54, 0), (92, 71), (63, 86), (70, 34), (91, 35), (93, 30), (39, 44), (86, 36), (43, 13),
        (89, 41), (85, 70), (0, 72), (18, 39), (93, 76), (90, 65), (88, 0), (91, 36), (87, 66), (45, 14),
        (0, 33), (54, 89), (35, 46), (91, 73), (95, 35), (85, 71), (94, 31), (88, 37), (91, 35), (66, 34),
        (86, 38), (92, 77), (56, 93), (58, 91), (60, 91), (61, 84), (90, 70), (85, 67), (0, 69), (94, 35),
        (28, 5), (89, 37), (90, 34), (94, 32), (95, 35), (16, 32)
    ]

    demands = [
        0, 5, 20, 7, 2, 19, 19, 9, 15, 22, 14, 15, 18, 4, 6, 4, 10, 16, 14, 1, 23, 19, 16, 22, 15, 4, 21,
        16, 22, 6, 2, 11, 5, 2, 12, 6, 17, 15, 11, 4, 22, 13, 21, 13, 20, 6, 15, 22, 11, 18, 8, 22, 16, 7,
        16, 21, 16, 17, 16, 23, 3, 20, 13, 6, 12, 15
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1316
    car = 9

    return coordinates, demands, capacity, car, answer

def test_22():
    # Исходные данные из задачи B-n68-k9

    # Координаты узлов и спрос
    coordinates = [
        (87, 39), (63, 45), (5, 71), (71, 13), (59, 63), (45, 95), (29, 13), (51, 79), (7, 67), (25, 85),
        (1, 7), (72, 14), (28, 86), (52, 86), (46, 100), (66, 46), (28, 90), (16, 70), (2, 14), (79, 21),
        (14, 72), (32, 88), (6, 72), (73, 15), (2, 12), (73, 21), (12, 76), (66, 72), (2, 16), (38, 16),
        (30, 88), (68, 54), (32, 14), (64, 46), (26, 88), (62, 68), (34, 16), (8, 68), (77, 15), (73, 15),
        (75, 15), (8, 68), (36, 20), (56, 82), (26, 88), (28, 86), (52, 84), (60, 70), (26, 86), (8, 72),
        (76, 16), (6, 76), (14, 72), (78, 14), (34, 92), (74, 14), (64, 46), (10, 74), (6, 8), (28, 88),
        (60, 64), (6, 78), (46, 102), (77, 23), (68, 66), (58, 80), (58, 84), (36, 14)
    ]

    demands = [
        0, 10, 3, 7, 18, 2, 2, 23, 8, 9, 22, 12, 20, 21, 9, 6, 11, 18, 23, 10, 11, 3, 10, 23, 8, 16,
        26, 19, 9, 18, 5, 23, 1, 2, 21, 19, 5, 14, 10, 2, 16, 17, 5, 16, 6, 11, 14, 17, 8, 48, 12, 9,
        6, 10, 19, 22, 5, 12, 5, 9, 8, 21, 9, 3, 8, 9, 16, 17
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1272
    car = 9

    return coordinates, demands, capacity, car, answer

def test_23():
    # Исходные данные из задачи B-n78-k10

    # Координаты узлов и спрос
    coordinates = [
        (46, 12), (51, 4), (52, 30), (80, 70), (18, 90), (59, 39), (23, 59), (77, 48), (82, 30), (18, 82),
        (11, 41), (7, 9), (88, 33), (23, 88), (0, 76), (85, 34), (17, 46), (52, 10), (13, 45), (19, 85),
        (86, 77), (54, 6), (83, 32), (15, 10), (53, 5), (14, 42), (13, 10), (57, 32), (20, 85), (65, 46),
        (61, 42), (87, 52), (79, 51), (25, 91), (89, 34), (26, 100), (0, 88), (63, 43), (55, 10), (23, 86),
        (8, 18), (0, 74), (20, 44), (56, 7), (14, 10), (88, 40), (96, 38), (59, 31), (22, 87), (59, 36),
        (24, 83), (83, 37), (53, 5), (0, 37), (84, 78), (27, 93), (61, 12), (69, 43), (54, 9), (20, 98),
        (18, 50), (25, 84), (31, 69), (58, 36), (0, 11), (61, 36), (18, 49), (57, 8), (0, 49), (56, 8),
        (62, 45), (83, 32), (53, 10), (82, 53), (21, 85), (64, 41), (80, 50), (16, 10)
    ]

    demands = [
        0, 14, 17, 17, 16, 19, 17, 5, 12, 4, 2, 2, 26, 2, 7, 18, 6, 6, 18, 2, 14, 5, 9, 4, 3, 15,
        4, 23, 7, 21, 4, 1, 6, 16, 4, 20, 5, 14, 14, 26, 5, 2, 14, 11, 21, 20, 18, 2, 19, 12, 22, 14,
        23, 25, 8, 3, 9, 21, 3, 22, 6, 2, 22, 20, 5, 13, 6, 14, 16, 12, 23, 5, 12, 15, 21, 4, 23, 19
    ]

    # Вместимость грузовика
    capacity = 100
    answer = 1221
    car = 10

    return coordinates, demands, capacity, car, answer

def all_B_set():
    return [test_1(), test_2(), test_3(), test_4(), test_5(), test_6(),
            test_7(), test_8(), test_9(), test_10(), test_11(), test_12(),
            test_13(), test_14(), test_15(), test_16(), test_17(), test_18(), test_19(),
            test_20(), test_21(), test_22(), test_23()]