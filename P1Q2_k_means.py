import numpy as np
import matplotlib.pyplot as plt


def k_means(dataset, k=3):
    centroids = dataset[np.random.choice(dataset.shape[0], k, replace=False)]

    while True:
        distances = np.linalg.norm(dataset[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([dataset[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def calculate_accuracy_and_distance(dataset):
    labels, centroids = k_means(dataset)

    accuracy = 0
    distance = 0

    for point in range(len(labels)):
        if labels[point] == 0:
            distance += np.linalg.norm(centroids[0] - dataset[point])
            if 0 <= point < 100:
                accuracy += 1
        elif labels[point] == 1:
            distance += np.linalg.norm(centroids[1] - dataset[point])
            if 100 <= point < 200:
                accuracy += 1
        elif labels[point] == 2:
            distance += np.linalg.norm(centroids[2] - dataset[point])
            if 200 <= point < 300:
                accuracy += 1

    return accuracy / 300, distance


def main():
    np.random.seed(0)
    n = 100
    sigmas = [0.5, 1, 2, 4, 8]
    datasets = []

    dists = {
        'mean_a': np.array([-1, -1]),
        'mean_b': np.array([1, -1]),
        'mean_c': np.array([0, 1]),
        'cov_a': np.array([[2, 0.5], [0.5, 1]]),
        'cov_b': np.array([[1, -0.5], [-0.5, 2]]),
        'cov_c': np.array([[1, 0], [0, 2]])
    }

    for sigma in sigmas:
        dataset_a = np.random.multivariate_normal(dists['mean_a'], sigma * dists['cov_a'], n)
        dataset_b = np.random.multivariate_normal(dists['mean_b'], sigma * dists['cov_b'], n)
        dataset_c = np.random.multivariate_normal(dists['mean_c'], sigma * dists['cov_c'], n)

        dataset = np.vstack((dataset_a, dataset_b, dataset_c))
        datasets.append(dataset)

    accuracies = []
    distances = []
    for dataset in datasets:
        accuracy, distance = calculate_accuracy_and_distance(dataset)
        accuracies.append(accuracy)
        distances.append(distance)

    plt.figure(1)
    plt.plot(sigmas, distances)
    plt.xlabel("σ")
    plt.ylabel("Total distance from points to their assigned clusters centers")
    plt.figure(2)
    plt.plot(sigmas, accuracies)
    plt.xlabel("σ")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    main()
