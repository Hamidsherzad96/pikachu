import math     #Har använt mig av lecture notes, w3school,  och exercisen som vi har övat på. Även bollat lite med klasskamraterna.
                #Har även bett om förklaringar från Chat-GPT och CO-Pilot för att förstå implementationen av koden/strukturen.
                #Fick hjälp med strukturen, bantningen och korrigering av koden, av en vän till mig som jobbar inom denna bransch-
                #och han har kunskaper inom python-programmering.            

import numpy as np
import matplotlib.pyplot as plt
import re

TEST_POINTS_FILENAME = "data/testpoints.txt"
DATA_POINTS_FILENAME = 'data/datapoints.txt'

def read_test_points(filename):
    #Öppnar filen och läser allt som en enda String.
    with open(filename, 'r') as f:
        text = f.read()

    matches = re.findall(r"\(([^)]*)\)", text)
    return [(float(match[0].strip()), float(match[1].strip())) for match in map(lambda m: m.split(','), matches)]


def distance(p1, p2) -> float:
    #Här tillämpar jag Euclidean distance formeln, för att beräkna avståndet mellan punkterna.
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def split_data(data):
    #splittar på datan här så de blir 2 grupper.
    pichu = [(x, y) for (x, y, n) in data if n == 0]
    pikachu = [(x, y) for (x, y, n) in data if n == 1]

    return pichu, pikachu


def plot_points(pichu, pikachu):
    # Unpack tuples into x and y listor
    x1, y1 = zip(*pichu)
    x2, y2 = zip(*pikachu)

    # Scatter plot
    plt.scatter(x1, y1, color="blue", label="Pichu", marker="o")
    plt.scatter(x2, y2, color="red", label="Pikachu", marker="x")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Two Datasets as Points")
    plt.legend()
    plt.show()


def classify(point, pichu, pikachu) -> str:
    """
    :param point: The point to classify
    :param pichu: Pichu points
    :param pikachu: Pikachu points
    :return: Either 'pichu' or 'pikachu'
    """
    d_pichu = min(map(lambda x: distance(x, point), pichu))
    d_pikachu = min(map(lambda x: distance(x, point), pikachu))

    return 'Pichu' if d_pichu < d_pikachu else 'Pikachu'

def classify_knn(point, pichu, pikachu):
    d_pichu = list(map(lambda x: distance(x, point), pichu))
    d_pikachu = list(map(lambda x: distance(x, point), pikachu))

    count = 0
    for i in range(10):
        min_d_pichu = min(d_pichu)
        min_d_pikachu = min(d_pikachu)

        count += 1 if min_d_pichu < min_d_pikachu else -1

        d_pichu.remove(min_d_pichu)
        d_pikachu.remove(min_d_pikachu)

    return 'Pichu' if count >= 0 else 'Pikachu'


def classify_point(test_points, pichu, pikachu):
    for p in test_points:
        character_type = classify(p, pichu, pikachu)
        print(f'Sample with (width, height): {p} classified as {character_type}')


def read_point(message) -> tuple:
    """
    Attempts to read a point from the user.
    On error, the message will be set as the second component of the resulting tuple. Otherwise, it will be None.
    On success, the point will be set as the first component of the resulting tuple. Otherwise, it will be None.
    When both are None, it means that the user typed a blank line.
    :param message: The prompt message for the user.
    :return: a tuple which first component is a tuple (width, height) and second component is an error message.
    """
    answer = input(message)

    if answer.strip() == '':
        return None, None

    parts = answer.split(',')

    if len(parts) != 2:
        return None, 'Missing component'

    [width, height] = parts

    try:
        width = float(width)
        height = float(height)
    except ValueError:
        return None, 'Invalid point'

    if width < 0 or height < 0:
        return None, 'Values must be non-negative'

    return (width, height), None


def main():
    """
    Runs exercise 1
    """
    data = np.genfromtxt(DATA_POINTS_FILENAME, delimiter=',', names=True)
    test_points = read_test_points(TEST_POINTS_FILENAME)

    pichu, pikachu = split_data(data)
    plot_points(pichu, pikachu)
    classify(test_points, pichu, pikachu)


def main_2():
    """
    Runs exercise 2
    """
    data = np.genfromtxt(DATA_POINTS_FILENAME, delimiter=',', names=True)
    pichu, pikachu = split_data(data)   

    while True:
        p, error = read_point('Type a point (width, height): ')

        if error:
            print(error)
            continue

        if p is None:
            break

        character_type = classify_knn(p, pichu, pikachu)
        print(character_type, end="\n")


main()