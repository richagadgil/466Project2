import sys
import numpy as np
import re
import nltk
import random
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

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
            #filtered = [word.strip() for word in words]
            #filtered[:] = [x for x in filtered if x != '']
            #feature = get_features(words[14])
            #for name in feature:
            #    if name not in overall_features:
            #        overall_features[name] = 0 
            #features.append(feature)
            #num_records += 1
            #if words[3] not in c_names:
            #    c_names[words[3]] = 1
            #else:
            #    c_names[words[3]] += 1
            #print(num_records)
            #if(num_records == 3000):
            #    break  
    #print(data['data'])

    test_records = []
    for i in range(400):
        record = random.choice(records)
        test_records.append(record)
        feature = get_features(record.text)
        for name in feature:
            if name not in overall_features:
                overall_features[name] = 0 
        if record.c_name not in c_names:
            c_names[record.c_name] = 1
        else:
            c_names[record.c_name] +=1
        features.append(feature)
              
    for i in range(len(features)):
        vector = dict.fromkeys(overall_features, 0)
        for key in features[i]:
            vector[key] = features[i][key]
        vectors.append(np.array(list(vector.values())))
        test_records[i].add_vector(np.array(list(vector.values())))


    print("My Kmeans labels:")
    for i in range(5, 6):
        print("K:", i)
        clusters = my_kmeans(test_records, i, 0.01)
        contingency_table(c_names, clusters)
        print("Sklearn kmeans labels:")
        get_scikit_kmeans_centroids(i,np.array(vectors),0.01)




def contingency_table(labels, clusters):
    #print(labels)
 
    purity_total_rows = sum(labels.values())
    purity_max_sum = 0

    rows = {}
    sum_f1 = 0


    for clusterNo in range(0, len(clusters)):
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

       # print(clusterNo , " ", row)
       # print("Precision of Cluster", clusterNo, "=", precision) 
       # print("Recall of Cluster", clusterNo, "=", recall) 
       # print("F1 Score of Cluster", clusterNo, "=", f1, "\n\n\n") 
        sum_f1 += f1
    

    average_f1 = sum_f1/len(clusters)

    
    print("Average F1", average_f1) #F1 Avg Calculation
    print("Purity", purity_max_sum/purity_total_rows) #Purity Calculation

    
def pre_process(text):
    #words = nltk.word_tokenize(text)
    #tags = nltk.pos_tag(words)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    porter_stemmer = nltk.stem.porter.PorterStemmer() 
    #target_tags = ["JJ", "JJR", "JJS", "NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    #target_tags = ["NN", "NNP", "NNPS", "NNS"]
    #filtered_words = [tag for tag in tags if tag[1] in target_tags]
    #filtered_words = [word for word in filtered_words if word[0].lower() not in stopwords.words('english')]            
    #filtered_words = [lemmatizer.lemmatize(word[0]) for word in filtered_words]
    #print(filtered_words)
    

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text=re.sub("(\\d|\\W)+"," ",text)
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_words = [porter_stemmer.stem(filtered_word) for filtered_word in filtered_words]
    
    return filtered_words

def get_features(text):
    filtered_words = pre_process(text)  
    features = {}
    # if len(filtered_words) < 2:
    #     return {}
    # elif len(filtered_words) == 2:
    #     featureName = ' '.join(filtered_words)
    #     if featureName in features:
    #         features[featureName] += 100
    #     else:
    #         features[featureName] = 100
    # else:
    #       for window in range(len(filtered_words)-1):
    #         featureName = ' '.join(filtered_words[window:window+2])
    #         if featureName in features:
    #             features[featureName] += 100
    #         else:
    #             features[featureName] = 100
    for word in filtered_words:
       if word in features:
           features[word] += 100
       else:
           features[word] = 100

    return features
def get_scikit_kmeans_centroids(num_clusters, vectors, tolerance):
    print(list(KMeans(
            n_clusters=num_clusters,
            n_init=1,
            init=vectors[:num_clusters],
            random_state=0,
            tol=tolerance
    ).fit(vectors).labels_)) #we overwrite the foggy method

    
def _closest_cluster_index(feature_x_j, centroids):
    closest_dist = np.inf
    closest_cluster_index = 0


    
    for index, centroid_i in enumerate(centroids):
        #euclidean_distance = np.linalg.norm(feature_x_j - centroid_i)
        #print("hi\n")
        #print(type(centroid_i))
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
        #for feature_x_j in Data:
        for i in range(len(Data)):
            #print(type(feature_x_j))
            feature_x_j = Data[i]
            cci = _closest_cluster_index(feature_x_j, centroids)
            clusters[cci].append(feature_x_j)
            labels[counter] = cci 
            counter += 1

        # Centroid update step
        for i in range(k):
            centroids[i] = np.mean([record.vector for record in clusters[i]] , axis = 0)
            
        # Check if within error
        for centroid, last_centroid in zip(centroids, last_centroids):
            if len(last_centroids) > 0 and np.sum((centroid - last_centroid)**2) <= e:
                print(labels)
                return clusters  # Optimal clustering achieved.

        # Save current to t-1
        # Current (t) will be overwritten in next loop (unless loop not entered)
        last_centroids = np.copy(centroids)


if __name__ == '__main__':
    main()
    
