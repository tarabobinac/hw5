import numpy as np
import matplotlib.pyplot as plt


def pca(X, d, type=1):
    X_i = X
    if type == 2:
        X_i = X - np.mean(X, axis=0)
    elif type == 3:
        X_i = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    U, S, V_T = np.linalg.svd(X_i, full_matrices=False)
    Z = np.dot(X_i, V_T[:d, :].T)
    reconstructions = np.dot(Z, V_T[:d, :])

    if type == 2:
        reconstructions += np.mean(X, axis=0)
    elif type == 3:
        reconstructions = (reconstructions * np.std(X, axis=0)) + np.mean(X, axis=0)

    return U, S, V_T, Z, reconstructions


def dro(X, d):
    X_i = X - np.mean(X, axis=0)
    U, S, V_T = np.linalg.svd(X_i, full_matrices=False)
    print(S)  # Find "knee point"

    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Z = U_d @ S_d

    Z_centered = Z - np.mean(Z, axis=0)
    Z = Z_centered / np.sqrt(np.mean(Z_centered ** 2, axis=0))

    A = np.transpose(X_i) @ Z @ np.linalg.inv(np.transpose(Z) @ Z)
    b = np.mean(X, axis=0) - A @ np.mean(Z, axis=0)

    reconstructions = (Z @ np.transpose(A)) + np.tile(b, (X.shape[0], 1))

    return U, S, V_T, Z, reconstructions


def plot(X, reconstructions, i, title):
    plt.figure(i)
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    plt.scatter(reconstructions[:, 0], reconstructions[:, 1], marker='x')
    plt.xlim(-3, 10)
    plt.ylim(-3, 10)
    plt.title(title)


def main():
    d = 1
    X = np.genfromtxt('./data/data2D.csv', delimiter=',', dtype=float)
    X_1000 = np.genfromtxt('./data/data1000D.csv', delimiter=',', dtype=float)

    b_U, b_S, b_V, b_Z, b_reconstructions = pca(X, d)
    d_U, d_S, d_V, d_Z, d_reconstructions = pca(X, d, 2)
    n_U, n_S, n_V, n_Z, n_reconstructions = pca(X, d, 3)
    dro_U, dro_S, dro_V, dro_Z, dro_reconstructions = dro(X, d)

    b2_U, b2_S, b2_V, b2_Z, b2_reconstructions = pca(X_1000, 31)
    d2_U, d2_S, d2_V, d2_Z, d2_reconstructions = pca(X_1000, 31, 2)
    n2_U, n2_S, n2_V, n2_Z, n2_reconstructions = pca(X_1000, 31, 3)
    dro2_U, dro2_S, dro2_V, dro2_Z, dro2_reconstructions = dro(X_1000, 31)

    print("Average buggy PCA reconstruction error on data2D: " + str(np.sum(np.square(X - b_reconstructions)) / 50))
    print("Average demeaned PCA reconstruction error on data 2D: " + str(np.sum(np.square(X - d_reconstructions)) / 50))
    print("Average normalized PCA reconstruction error on data 2D: " + str(np.sum(np.square(X - n_reconstructions)) / 50))
    print("Average DRO reconstruction error on data 2D: " + str(np.sum(np.square(X - dro_reconstructions)) / 50))

    print("Average buggy PCA reconstruction error on data1000D: " + str(np.sum(np.square(X_1000 - b2_reconstructions)) / 500))
    print("Average demeaned PCA reconstruction error on data 1000D: " + str(np.sum(np.square(X_1000 - d2_reconstructions)) / 500))
    print("Average normalized PCA reconstruction error on data 1000D: " + str(np.sum(np.square(X_1000 - n2_reconstructions)) / 500))
    print("Average DRO reconstruction error on data 1000D: " + str(np.sum(np.square(X_1000 - dro2_reconstructions)) / 500))

    plot(X, b_reconstructions, 1, "Buggy PCA")
    plot(X, d_reconstructions, 2, "Demeaned PCA")
    plot(X, n_reconstructions, 3, "Normalized PCA")
    plot(X, dro_reconstructions, 4, "DRO")
    plt.show()


if __name__ == '__main__':
    main()
