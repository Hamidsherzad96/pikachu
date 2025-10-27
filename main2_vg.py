import math
import random

import numpy as np
from matplotlib import pyplot as plt

#Definerar Pichu och Pikachu.
PICHU = 0
PIKACHU = 1

# Läser in data från filen datapoints.txt.
#kör klassifieringen 10 gågner och samlar nogrannheten
# från varje körning och plottar resultaten.
def main():
    data_raw = read_data('datapoints.txt')

    runs = 10
    accuracies = [compute_accuracy(data_raw, runs) for _ in range(runs)]

    plot_accuracies(accuracies)

#delar upp datan i tränings och test mängder.
#klassifierar testdata med KNN och returnerar nogrannheten.
def compute_accuracy(data_raw, k):
    data = split_data(data_raw)
    stats = evaluate(data['train'], data['test'], k)
    return stats['acc']

#skapar en scatter-plot, ritar ett medelvärde som en blå linje.
def plot_accuracies(accuracies):
    x = list(range(1, len(accuracies) + 1))
    y = accuracies

    mean_accuracy = sum(accuracies) / len(accuracies)

    plt.scatter(x, y, color="red", label="Accuracy", marker="x")
    plt.axhline(y=mean_accuracy, color='blue', linestyle='--', linewidth=1, label='Mean Accuracy')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

#klassifierar varje testpunkt med KNN och räknar tp, fp, tn, fn.
#Beräknar nogrannheten.
def evaluate(training_data, test_data, k=10):
    tp = fp = tn = fn = 0
    for p in test_data:
        predicted_label, _ = classify_knn(p, training_data, k)

        if predicted_label == PIKACHU and p['label'] == PIKACHU:
            tp += 1
        elif predicted_label == PIKACHU and p['label'] == PICHU:
            fp += 1
        elif predicted_label == PICHU and p['label'] == PIKACHU:
            fn += 1
        elif predicted_label == PICHU and p['label'] == PICHU:
            tn += 1
        else:
            raise RuntimeError('Unreachable')

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'acc': accuracy,
    }

#Beröknar avstånd till alla träningspunkter och sorterar efter avstånd.
def classify_knn(point, training_data, k=10):
    distances = [(distance(point, p), p['label']) for p in training_data]

    by_distance = lambda p: p[0]
    distances.sort(key=by_distance)

    k_nearest = distances[:k]

    votes = {PICHU: 0, PIKACHU: 0}

    for _, label in k_nearest:
        votes[label] += 1
#Tar de k närmaste grannarna här och "röstar" på etikett, därefter returnerar etikett.
    label_of_nearest = k_nearest[0][1]
    predicted_label = label_of_nearest if votes[PICHU] == votes[PIKACHU] else (
        PIKACHU if votes[PIKACHU] > votes[PICHU] else PICHU)

    confidence = votes[predicted_label] / k
    return predicted_label, confidence

#Läser CSV data med width, height, label.
def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',', names=True)
    return list(map(create_point, data))


def create_point(row):
    width, height, label = row
#Skapar en lsita med dicts.
    return {
        'width': width,
        'height': height,
        'label': int(label)
    }


def random_take(collection, count):
    """
    Tar ett angivet antal slumpmässiga objekt från samlingen och returnerar en lista som innehåller dem.
    Samlingen ändras direkt (på plats).
    """
    result = []

    for i in range(min(count, len(collection))):
        index = random.randint(0, len(collection) - 1)
        result.append(collection.pop(index))

    return result


def distance(p0, p1):
    return math.hypot(p0['width'] - p1['width'], p0['height'] - p1['height'])

#delar upp datan i två grupper. Tar bort 50 slumpmässiga från varje träningsdata
#och resten används som testdata.
def split_data(data):
    test_pichu = [p for p in data if p['label'] == PICHU]
    training_pichu = random_take(test_pichu, 50)

    test_pikachu = [p for p in data if p['label'] == PIKACHU]
    training_pikachu = random_take(test_pikachu, 50)

    return {
        "train": training_pichu + training_pikachu,
        "test": test_pichu + test_pikachu,
    }


if __name__ == '__main__':
    main()
