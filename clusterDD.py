import sys
import numpy as np
import re
import nltk
import random
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 
from nltk.corpus import wordnet as wn
import time


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

    def add_c_name(self, c_name):
        self.c_name = c_name

    def add_text(self, text):
        self.text = text
    
    def add_vector(self, vector):
        self.vector = vector

def main():

    np.seterr(all='print')
    
    nltk.download('omw') 
    args = sys.argv[1:]
    if not args or len(args) != 1:
        print("usage: main filename")
        sys.exit(1)
    filename = args[0]
    overall_features = {}
    records = []
    vectors = []
    features = []

    c_names = {}
    all_c_names = {}
    with open(filename, 'r') as f:
        num_records = 0
        for line in f:
            if num_records == 0:
                num_records += 1
                continue 
            words = line.split('\t')
            r = Record()
            r.add_cid(words[2])
            r.add_c_name(words[3])
            r.add_text(words[14])
            records.append(r)
            if words[3] not in all_c_names:
                all_c_names[words[3]] = 1
    
    test_records = set()

    for i in range(8000):
        record = random.choice(records)
        while record in test_records:
            record = random.choice(records)
        test_records.add(record)
    
    test_records = list(test_records)
    for record in test_records:
        feature = get_features(record.text)
        for name in feature:
            if name not in overall_features:
                overall_features[name] = 0 
        if record.c_name not in c_names:
            c_names[record.c_name] = 1
        else:
            c_names[record.c_name] += 1
        features.append(feature)

    for i in range(len(features)):
        vector = dict.fromkeys(overall_features, 0)
        for key in features[i]:
            vector[key] = features[i][key]
        
        vectors.append(np.array(list(vector.values())))
        test_records[i].add_vector(np.array(list(vector.values())))


    print("My Kmeans labels:")
    for i in range(len(c_names) - 1, len(c_names)):
        print("K:", i+2)
        clusters = my_kmeans(test_records, i, 0.01)
        contingency_table(c_names, clusters)
        
        #print("SciKitLearn Labels:")
        #get_scikit_kmeans_centroids(34, np.array(vectors), 0.01)
        # USED TO TEST AGAINST SCIKIT LEARN RESULTS 
        #print(len([i for i, j in zip(my_labels, sklabels) if i == j]))



def contingency_table(labels, clusters):
    #print(labels)
 
    purity_total_rows = sum(labels.values())
    purity_max_sum = 0

    rows = {}
    sum_f1 = 0


    for clusterNo in range(0, len(clusters)):
        if len(clusters[clusterNo]) == 0:
            continue
        max_label = ""
        max_label_amt = 0
        row = []
        for label in labels.keys():
            items_per_label = len([x for x in clusters[clusterNo] if x.c_name == label])
            row.append(items_per_label)
            if(items_per_label > max_label_amt):
                max_label_amt = items_per_label
                max_label = label

        precision = max_label_amt / len(clusters[clusterNo])
        recall = max_label_amt / labels[max_label]
        purity_max_sum += max(row) #Add maximum label value present in cluster
        f1 = (2 * precision * recall) / (precision + recall)

        print(clusterNo , " ", row)
        print("Precision of Cluster", clusterNo, "=", precision) 
        print("Recall of Cluster", clusterNo, "=", recall) 
        print("F1 Score of Cluster", clusterNo, "=", f1, "\n\n\n") 
        sum_f1 += f1
    

    average_f1 = sum_f1/len(clusters)

    
    print("Average F1", average_f1) #F1 Avg Calculation
    print("Purity", purity_max_sum/purity_total_rows) #Purity Calculation

    
def pre_process(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    porter_stemmer = nltk.stem.porter.PorterStemmer() 

    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    words = nltk.word_tokenize(text)
    filtered_words = words
    filtered_words = [lemmatizer.lemmatize(filtered_word) for filtered_word in filtered_words]
    return filtered_words

def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None 

def get_features(text):
    filtered_words = pre_process(text) 


    features = {}
   
    tag_counts = {}
    tag_counts["N"] = 0
    tag_counts["ADJ"] = 0
    tag_counts["V"] = 0
    tag_counts["ADV"] = 0
    tag_counts["NUM"] = 0
    charLength = 0
    numSynonyms = 0 
    if len(filtered_words) == 0:
        return {}
    counter = 1
    synonyms = set()
    for word in filtered_words:

        tag = nltk.pos_tag([word])
        tag = tag[0]
        tag = tag[1]

        wordnet_tag = penn2morphy(tag)
    
        charLength += len(word)
        if tag.startswith('N') or tag.startswith('P'):
             tag_counts["N"] += 1
        elif tag.startswith('J'):
             tag_counts["ADJ"] += 1
        elif tag.startswith('V'):
            tag_counts["V"] += 1
        elif tag.startswith('R'):
            tag_counts["ADV"] += 1
        elif tag == "CD":    
            tag_counts["NUM"] += 1
        
    tag_counts["N"] = tag_counts["N"]
    tag_counts["ADJ"] = tag_counts["ADJ"]
    tag_counts["V"] = tag_counts["V"]
    tag_counts["ADV"] = tag_counts["ADV"]
    tag_counts["NUM"] = tag_counts["NUM"]

    stop = stopwords.words('english')
    features["stopWords"] = len([x for x in filtered_words if x.lower() in stop]) / len(filtered_words)
    
    features["numNouns"] = tag_counts["N"] / len(filtered_words)
    features["numVerbs"] = tag_counts["V"] / len(filtered_words)
    features["numAdj"] = tag_counts["ADJ"] / len(filtered_words)
    features["numNum"] = tag_counts["NUM"] / len(filtered_words)
            
    return features
def get_scikit_kmeans_centroids(num_clusters, vectors, tolerance):
    return list(KMeans(
            n_clusters=num_clusters,
            n_init=1,
            init=vectors[:num_clusters],
            random_state=0,
            tol=tolerance
    ).fit(vectors).labels_) #we overwrite the foggy method

def _closest_cluster_index(feature_x_j, centroids):
    closest_dist = np.inf
    closest_cluster_index = 0


    
    for index, centroid_i in enumerate(centroids):
        euclidean_distance = np.linalg.norm(feature_x_j.vector - centroid_i) ** 2
       

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
    #centroids = np.copy(Data[:k])
    centroids = [x.vector for x in Data[:k]]


    i = 0
    while True:
        clusters = [[] for __ in range(k)]

        labels = [0] * len(Data)
        # Cluster assignment step
        counter = 0
        for i in range(len(Data)):
            feature_x_j = Data[i]
            cci = _closest_cluster_index(feature_x_j, centroids)
            clusters[cci].append(feature_x_j)
            labels[counter] = cci 
            counter += 1

        # Centroid update step
        for i in range(k):
            if len(clusters[i]) != 0:
                centroids[i] = np.mean([record.vector for record in clusters[i]] , axis = 0)
            else:
                pass
        # Check if within error TESTING
        if len(last_centroids) > 0:
            sum = 0
            for centroid, last_centroid in zip(centroids, last_centroids):
                sum += np.sum(centroid - last_centroid) ** 2
            if(sum <= e):
                return clusters  # Optimal clustering achieved.

        # Save current to t-1
        # Current (t) will be overwritten in next loop (unless loop not entered)
        last_centroids = np.copy(centroids)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
