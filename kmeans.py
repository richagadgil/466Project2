import sys
import numpy as np
import re




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
    word_occurrences = {}
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
            text = re.sub('[^A-Za-z0-9 ]+', '', words[14]).lower().split()
            unique_by_record = set(text)
            for word in unique_by_record:
                if word in word_occurrences:
                    word_occurrences[word] += 1
                else:
                    word_occurrences[word] = 1 
            num_records += 1
            if(num_records == 3000):
                break  
    with open(filename, 'r') as f:
        counter = 0
        for line in f:
            if counter == 0:
                counter += 1
                continue
            words = line.split('\t')
            #filtered = [word.strip() for word in words]
            #filtered[:] = [x for x in filtered if x != '']
            text = re.sub('[^A-Za-z0-9 ]+', '', words[14]).lower().split()
            vector = get_features(text, num_records,word_occurrences)
            vectors.append(np.array(vector)) 
            records[counter-1].add_vector(np.array(vector))
            counter += 1
            if(counter == 3000):
                break
    #print(data['data'])
    k = len(found_cids)
    my_kmeans(records, k, 0.001)
    
            
def get_features(words,numRecords, wordOccurrences):
    bag_words = get_frequencies(words)
    tf_idf_vals = computeTF_IDF(bag_words, words, numRecords, wordOccurrences)
    vector_dict = {}
    for key in wordOccurrences.keys():
        if key in tf_idf_vals.keys():
            vector_dict[key] = tf_idf_vals[key] 
        else:
            vector_dict[key] = 0
    return list(vector_dict.values())

def get_frequencies(words):
    words_dict = {}
    for word in words:
        if word in words_dict:
            words_dict[word] += 1
        else:
            words_dict[word] = 1	
    return words_dict

def computeTF_IDF(bag_of_words, words, num_lines, occurrences_overall):
    tf_idf_values = {}
    n = len(words)
    for key in bag_of_words.keys():
        tf = bag_of_words[key] / n
        tf_idf_values[key] = tf * np.log(num_lines / occurrences_overall[key])
    return tf_idf_values
	

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
        clusters = [[] for __ in range(k)]
        # Cluster assignment step
        for feature_x_j in Records:
            cci = _closest_cluster_index(feature_x_j, centroids)
            clusters[cci].append(feature_x_j)


        # Centroid update step
        for index, cluster in enumerate(clusters):
            clusters_vec = []
            for i in range(len(cluster)):
                clusters_vec.append(cluster[i].vector)
            centroids[index] = np.average(clusters_vec, axis = 0)

        # Check if within error
        for centroid, last_centroid in zip(centroids, last_centroids):
            if len(last_centroids) > 0 and np.sum((centroid- last_centroid)**2) <= e:
                labels = [[] for __ in range(k)]
                for i in range(0, len(clusters)):
                    for j in range(len(clusters[i])):
                        labels[i].append(clusters[i][j].cid)
                print(labels[5])
                return clusters  # Optimal clustering achieved.

        # Save current to t-1
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
    main()
    
