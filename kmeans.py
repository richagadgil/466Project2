import sys
import numpy as np
import re
import nltk
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 


class Record:

    cid = None
    c_name = None
    c_house = None
    hid = None
    pid = None
    diarization_id = None
    text = None
    vector = None
    def __init__(self):
        pass

    def add_cid(self, cid):
        self.cid = cid
    
    def add_text(self, text):
        self.text = text
    
    def add_vector(self, vector):
        self.vector = vector

def main():
    args = sys.argv[1:]
    if not args or len(args) != 1:
        print("usage: main filename")
        sys.exit(1)
    filename = args[0]
    records = []
    vectors = []
    found_cids = set()
    with open(filename, 'r') as f:
        num_records = 0
        for line in f:
            if num_records == 0:
                num_records += 1
                continue 
            words = line.split('\t')
            r = Record()
            r.add_cid(words[2])
            found_cids.add(words[2])
            r.add_text(words[14])
            records.append(r)
            #filtered = [word.strip() for word in words]
            #filtered[:] = [x for x in filtered if x != '']
            vector = get_features(words[14])
            vectors.append(vector)
            records[num_records-1].add_vector(vector) 
            num_records += 1
            if(num_records == 3000):
                break  
    #print(data['data'])
    k = len(found_cids)
    print("My Kmeans labels:")
    my_kmeans2(np.array(vectors), k, 0.01)
    print("Sklearn kmeans labels:")
    get_scikit_kmeans_centroids(k,np.array(vectors),0.01)
            
def get_features(text):
    features = {}
    words = re.sub('[^A-Za-z0-9 ]+', '', words[14]).lower().split()
    words = [word for word in words not in stopwords.words('english')]
	if len(words) < 3:
        return {}
    elif len(words) == 3:
        featureName = ' '.join(words)
        features[featureName] = 1 
    else:
        for window in range(len(words)-2):
            featureName = ' '.join(words[window:window+3])
            features[featureName] = 1
    return features    

def _closest_cluster_index(feature_x_j, centroids):
    closest_dist = np.inf
    closest_cluster_index = 0

    for index, centroid_i in enumerate(centroids):
        euclidean_distance = np.linalg.norm(feature_x_j.vector - centroid_i)

        if euclidean_distance < closest_dist:
            closest_dist = euclidean_distance
            closest_cluster_index = index

    return closest_cluster_index


def my_kmeans(Records, k, e=0.001):
    """
    :param Data: Raw return from _generate_points()
    :param k: Number of clusters to create
    :param e: Maximum error threshold
    :return: Final centroids and a NumPy array of features.
    """

    # we start with the first k points given as the initial centroids
    last_centroids = []
    centroids = []
    for i in range(k):
        last_centroids.append(Records[i].vector)
        centroids.append(Records[i].vector)    
    i = 0 
    while True:
        labels = [0] * len(Records)
        clusters = [[] for __ in range(k)]
        # Cluster assignment step
        counter = 0
        for feature_x_j in Records:    
            cci = _closest_cluster_index(feature_x_j, centroids)
            clusters[cci].append(feature_x_j)
            labels[counter] = cci
            counter+=1


        # Centroid update step
        for index, cluster in enumerate(clusters):
            clusters_vec = []
            for i in range(len(cluster)):
                clusters_vec.append(cluster[i].vector)
            centroids[index] = np.average(clusters_vec, axis = 0)

        # Check if within error
        for centroid, last_centroid in zip(centroids, last_centroids):
            if len(last_centroids) > 0 and np.sum((centroid- last_centroid)**2) <= e:
                print(labels)
                return clusters  # Optimal clustering achieved.

        # Save current to t-1
        last_centroids = np.copy(centroids)

def get_scikit_kmeans_centroids(num_clusters, vectors, tolerance):
    print(list(KMeans(
            n_clusters=num_clusters,
            n_init=1,
            init=vectors[:num_clusters],
            random_state=0,
            tol=tolerance
    ).fit(vectors).labels_)[:30]) #we overwrite the foggy method


def _closest_cluster_index(feature_x_j, centroids):
    closest_dist = np.inf
    closest_cluster_index = 0

    for index, centroid_i in enumerate(centroids):
        euclidean_distance = np.linalg.norm(feature_x_j - centroid_i)

        if euclidean_distance < closest_dist:
            closest_dist = euclidean_distance
            closest_cluster_index = index

    return closest_cluster_index


def my_kmeans2(Data, k, e=0.001):
    """
    :param Data: Raw return from _generate_points()
    :param k: Number of clusters to create
    :param e: Maximum error threshold
    :return: Final centroids and a NumPy array of features.
    """

    # we start with the first k points given as the initial centroids
    last_centroids = []
    centroids = np.copy(Data[:k])



    i = 0
    while True:
        clusters = [[] for __ in range(k)]

        labels = [0] * len(Data)
        # Cluster assignment step
        counter = 0
        for feature_x_j in Data:
            cci = _closest_cluster_index(feature_x_j, centroids)
            clusters[cci].append(feature_x_j)
            labels[counter] = cci 
            counter += 1

        # Centroid update step
        for index, cluster in enumerate(clusters):
            centroids[index] = np.average(clusters[index], axis = 0)

        # Check if within error
        for centroid, last_centroid in zip(centroids, last_centroids):
            if len(last_centroids) > 0 and np.sum((centroid - last_centroid)**2) <= e:
                print(labels[:30])
                return centroids  # Optimal clustering achieved.

        # Save current to t-1
        # Current (t) will be overwritten in next loop (unless loop not entered)
        last_centroids = np.copy(centroids)
    

if __name__ == '__main__':
    main()
    
