import random
import numpy as np
from sklearn.cluster import KMeans


def _generate_points(xLimit, yLimit, clusters, num_points):
    points = {}
    xypairs = []
    targets = []
    for n in range(num_points):
        x = random.randint(0, xLimit)
        y = random.randint(0, yLimit)
        t = random.randint(0, clusters)
        xypairs.append([x, y])
        targets.append(t)
    points['data'] = np.array(xypairs)

    print(points['data'])

    return points


def _closest_cluster_index(feature_x_j, centroids):
    closest_dist = np.inf
    closest_cluster_index = 0

    for index, centroid_i in enumerate(centroids):
        euclidean_distance = np.linalg.norm(feature_x_j - centroid_i)

        if euclidean_distance < closest_dist:
            closest_dist = euclidean_distance
            closest_cluster_index = index

    return closest_cluster_index


def my_kmeans(Data, k, e=0.001):
    """
    :param Data: Raw return from _generate_points()
    :param k: Number of clusters to create
    :param e: Maximum error threshold
    :return: Final centroids and a NumPy array of features.
    """

    # we start with the first k points given as the initial centroids
    last_centroids = []
    centroids = np.copy(Data['data'][:k])



    i = 0
    while True:
        clusters = [[] for __ in range(k)]

        # Cluster assignment step
        for feature_x_j in Data['data']:
            cci = _closest_cluster_index(feature_x_j, centroids)
            clusters[cci].append(feature_x_j)


        # Centroid update step
        for index, cluster in enumerate(clusters):
            centroids[index] = np.average(clusters[index], axis = 0)

        # Check if within error
        for centroid, last_centroid in zip(centroids, last_centroids):
            if len(last_centroids) > 0 and np.sum((centroid - last_centroid)**2) <= e:
                for i in range(0, len(clusters)):
                    print(clusters[i])
                return centroids  # Optimal clustering achieved.

        # Save current to t-1
        # Current (t) will be overwritten in next loop (unless loop not entered)
        last_centroids = np.copy(centroids)
    


def get_scikit_kmeans_centroids(num_clusters, points, tolerance):
    return KMeans(
            n_clusters=num_clusters,
            n_init=1,
            init=points['data'][:num_clusters],
            random_state=0,
            tol=tolerance
    ).fit(points['data']).cluster_centers_ #we overwrite the foggy method




if __name__ == '__main__':
    num_clusters = 3
    tolerance = 0.001
    points = _generate_points(xLimit = 100, yLimit = 100,
                              clusters = num_clusters, num_points = 20)

    scikit_kmeans_centroids = get_scikit_kmeans_centroids(num_clusters, points, tolerance)
    print('Scikit-learn final centroid coordinates:')
    print(scikit_kmeans_centroids)

    clutter = []
    #for i in range(0, 1):
    my_kmeans_centroids = my_kmeans(points, num_clusters, tolerance)
    clutter.append(my_kmeans_centroids)
    
    
    #print('\n My final centroid coordinates:')
    print(np.mean(clutter, axis=0))