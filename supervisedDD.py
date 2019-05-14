import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import time
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords 
from nltk.corpus import wordnet as wn
import nltk
import re
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier

def vectorize(n=0):
    complete_df = pd.DataFrame()
    df = pd.read_csv("DigitalDemocracy/committee_utterances.tsv", sep="\t")

   
    #get N most frequent speakers
    if(n > 0):
        items_counts = df['pid'].value_counts()[0:n]
        df = df[df['pid'].isin(items_counts.keys())]

    df['text'] = df['text'].str.lower().replace("[^A-Za-z\s]", "")


    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text'],df['pid'],test_size=0.2)

    stop_words = set(stopwords.words('english'))
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf.fit(df['text'])
    Train_X = tfidf.transform(Train_X)
    Test_X = tfidf.transform(Test_X)


    # Classifier SVM
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X,Train_Y)
    predictions = SVM.predict(Test_X)


    print("SVM Accuracy Score: ",accuracy_score(predictions, Test_Y)*100, "%")
    print("Average SVM Precision Score: ", np.mean(precision_score(predictions, Test_Y, average=None)) *100, "%")
    print("Average SVM Recall Score: ", np.mean(recall_score(predictions, Test_Y,average=None)) *100, "%")
    print("Average SVM F1 Score: ", np.mean(f1_score(predictions, Test_Y, average=None))*100, "%")

def main():
    if(len(sys.argv) > 1):
        n = sys.argv[1]
        vectorize(int(n))
    else:
        vectorize(0)

    return

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
