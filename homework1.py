import sklearn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from itertools import combinations


def euclidean_distance(x_vect, m_vect):
    return np.linalg.norm(x_vect - m_vect)


def plot_clusters(data, clusters, centroids, iteration):
    COLORS = ('green', 'blue', 'brown', 'black')
    plt.figure(figsize=(6, 5))

    for i, points in enumerate(clusters):
        points = np.array(points)
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1], s=10, color=COLORS[i])

    mx = [m[0] for m in centroids]
    my = [m[1] for m in centroids]
    plt.scatter(mx, my, s=100, color='red', marker='x', label='Centroids')

    plt.title(f'Iteration {iteration}')
    plt.legend()
    plt.show()


def main():
    # Шаг 1. Определение оптимального числа кластеров
    irises = load_iris()
    data = irises.data[:, :2]
    sse = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init=10)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма квадратов расстояний (SSE)')
    plt.title('Метод локтя')
    plt.show()

    # Оптимальное число кластеров
    optimal_k = 3

    # Шаг 2. Алгоритм K-Means
    centroids = [data[i] for i in np.random.choice(len(data), optimal_k, replace=False)]

    iteration = 0
    while iteration < 10:
        clusters = [[] for _ in range(optimal_k)]

        # Назначение точек к ближайшему центроиду
        for x_vect in data:
            distances = [euclidean_distance(x_vect, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(x_vect)

        prev_centroids = centroids.copy()
        centroids = [np.mean(cluster, axis=0) if len(cluster) > 0 else prev_centroids[i] for i, cluster in
                     enumerate(clusters)]


        plot_clusters(data, clusters, centroids, iteration + 1)

        iteration += 1

    # Шаг 3. Финальная визуализация всех проекций
    labels = np.zeros(len(data), dtype=int)
    for i, cluster in enumerate(clusters):
        for point in cluster:
            labels[np.where((data == point).all(axis=1))[0][0]] = i

    feature_combinations = list(combinations(range(irises.data.shape[1]), 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    COLORS = ('green', 'blue', 'brown', 'black')

    for ax, (i, j) in zip(axes.flatten(), feature_combinations):
        for cluster_idx in range(optimal_k):
            cluster_points = irises.data[labels == cluster_idx]
            ax.scatter(cluster_points[:, i], cluster_points[:, j], color=COLORS[cluster_idx], alpha=0.6,
                       label=f'Cluster {cluster_idx + 1}')

        ax.set_xlabel(irises.feature_names[i])
        ax.set_ylabel(irises.feature_names[j])
        ax.set_title(f'Projection: {irises.feature_names[i]} vs {irises.feature_names[j]}')

    plt.legend()
    plt.tight_layout()
    plt.show()


main()
