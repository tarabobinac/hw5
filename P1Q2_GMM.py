import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def gmm(dataset, k=3):
    weights = np.ones(k) / k
    means = dataset[np.random.choice(dataset.shape[0], k, replace=False)]
    covariances = [np.cov(np.transpose(dataset))] * k
    log_likelihood_prev = 0

    while True:
        probs = np.zeros((dataset.shape[0], k))
        for mean in range(k):
            probs[:, mean] = weights[mean] * multivariate_normal.pdf(dataset, mean=means[mean], cov=covariances[mean])
        probs /= probs.sum(axis=1)[:, np.newaxis]

        for mean in range(k):
            sum = probs[:, mean].sum()
            weights[mean] = sum / dataset.shape[0]
            means[mean] = (1 / sum) * np.sum(probs[:, mean, np.newaxis] * dataset, axis=0)
            covariances[mean] = (1 / sum) * np.transpose(probs[:, mean, np.newaxis] * (dataset - means[mean])) @ (dataset - means[mean])

        log_likelihood = 0
        for point in range(dataset.shape[0]):
            log_likelihood_per_point = 0
            for mean in range(k):
                log_likelihood_per_point += weights[mean] * multivariate_normal.pdf(dataset[point], mean=means[mean], cov=covariances[mean])
            log_likelihood += np.log(log_likelihood_per_point)

        if log_likelihood - log_likelihood_prev == 0:
            break

        log_likelihood_prev = log_likelihood

    labels = np.argmax(probs, axis=1)
    return labels, means, log_likelihood


def calculate_accuracy_and_log_likelihood(dataset):
    labels, means, log_likelihood = gmm(dataset)

    accuracy = 0

    for point in range(len(labels)):
        if labels[point] == 0:
            if 0 <= point < 100:
                accuracy += 1
        elif labels[point] == 1:
            if 100 <= point < 200:
                accuracy += 1
        elif labels[point] == 2:
            if 200 <= point < 300:
                accuracy += 1

    return accuracy / 300, log_likelihood


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
    negative_log_likelihoods = []
    for dataset in datasets:
        accuracy, log_likelihood = calculate_accuracy_and_log_likelihood(dataset)
        accuracies.append(accuracy)
        negative_log_likelihoods.append(log_likelihood * -1)

    plt.figure(1)
    plt.plot(sigmas, negative_log_likelihoods)
    plt.xlabel("σ")
    plt.ylabel("Negative log likelihood")
    plt.figure(2)
    plt.plot(sigmas, accuracies)
    plt.xlabel("σ")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    main()
