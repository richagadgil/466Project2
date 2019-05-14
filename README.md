Team Members: Richa Gadgil, Anand Rajiv, Aniketh Bhat

Part 1 (Clustering):
NOTE: Comparison to Sci-kit learn has been commented out in interest of time. Please uncomment to view comparison.

This program conducts feature extraction on all the text and clusters the clean text according to committee using k-means. This program outputs the number of labels, the confusion matrix, average F1 score and average purity, as well as the runtime.

Part 2 (Decision Tree) Results for 5000 rows

This program generates the decision tree for the data. It outputs the number of labels, an overall number of input records broken down by training and test sizes, the different labels(committees), an overall accuracy, and a per committee performance listing of (precision, recall, F1 score)

Overall accuracy: 0.14
Average Precision: 0.094
Average Recall: 0.092
Average F1 Score: 0.085

Part 3 (Speaker Attribution) Results for 5000 rows

This program using the SVM machine learning algorithm to predict the name of the speaker, given their text. The test set is approximately 20% of the data and the training set is 80%. First the training data is vectorized using TD-IDF Vectorizer before being fit using the SVM Model. The model is then fit the test data and its predictions are analyzed. The program outputs accuracy, average precision, average recall, and average F1 score along with Runtime.

NOTE: Pandas throws FutureWarning: Depreciation due to future changes in conversion caused by outdated versions. Program still runs as expected.

Overall accuracy: 0.57
Average Precision: 0.49
Average Recall: 0.59
Average F1 Score: 0.51



