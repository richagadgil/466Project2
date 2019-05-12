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



def vectorize():
    complete_df = pd.DataFrame()
    df = pd.read_csv("DigitalDemocracy/committee_utterances.tsv", sep="\t")
    
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text'],df['pid'],test_size=0.2)

    Tfidf_vect = TfidfVectorizer(max_features=500)
    Tfidf_vect.fit(df['text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100, "%")

    # Classifier - Algorithm - SVM
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100, "%")

    # Classifier - Logistic Regression - LR
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(Train_X_Tfidf, Train_Y)
    predictions_LR = clf.predict(Test_X_Tfidf)
    print("Logistic Regression Accuracy Score -> ",accuracy_score(predictions_LR, Test_Y)*100, "%")


def main():
    vectorize()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))