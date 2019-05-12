import sys
import os
import pandas as pd
import numpy as np
import random
from clusterDD import Record
from clusterDD import get_features
from clusterDD import pre_process

def split_sets(data, test_size):
    #Split data into training and testing
    if isinstance(test_size, float):
        test_size = round(test_size * len(data))
    indices = data.index.tolist()
    test_indices = random.sample(population = indices, k=test_size)

    test_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    return train_data, test_data

def check_purity(data, purity_thresh):
    if data.shape[0] == 1:
        vals, val_counts = np.unique(data[:,len(data) - 1], return_counts=True)
    else:
        vals, val_counts = np.unique(data[:,len(data[1])-1], return_counts=True)
    val_counts = sorted(val_counts)
    highest = val_counts[len(val_counts) - 1]
    purity = highest / float(np.size(data,0))
    print("Purity: {} , Purity Thresh: {}".format(purity,purity_thresh))
    if (purity >= purity_thresh):
        print("Returned")
        return True
    else:
        return False

#input: training dataframe.values
def get_potential_splits(data_values):
    potential_splits = {}
    _, n_cols = data_values.shape
    for col_index in range(n_cols - 1): # "-1 because last column is gonna be labels"
        vals = data_values[:, col_index]
        unique_vals = np.unique(vals)
        if len(unique_vals) > 1:
            potential_splits[col_index] = []
            for index in range(len(unique_vals)):
                if index != 0:
                    curr_val = unique_vals[index]
                    prev_val = unique_vals[index-1]
                    potential_split = (curr_val + prev_val)/2
                    potential_splits[col_index].append(potential_split)
    return potential_splits
        
#Input: Dataframe
def calculate_entropy(data):
    if(data.shape[0] == 1):
        label_column = data[:,len(data) - 1]
    else:
        label_column = data[:,len(data[1])-1]
    _, counts = np.unique(label_column, return_counts = True)
    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))    
    return entropy

#Input: Both sides of a split(Data frames)
def calculate_overall_entropy(data_below, data_above):
    num_points = len(data_below) + len(data_above)
    weight_below = len(data_below) / num_points
    weight_above = len(data_above) / num_points
    overall_entropy = (weight_below * calculate_entropy(data_below) + weight_above * calculate_entropy(data_above))
    return overall_entropy

#Input: Dataframe
def determine_best_split(data, potential_splits):
    overall_entropy = 1000
    for col_index in potential_splits:
        for value in potential_splits[col_index]:
            data_below, data_above = split_data(data, col_index, value)
            curr_overall_entropy = calculate_overall_entropy(data_below, data_above)
            if(curr_overall_entropy <= overall_entropy):
                overall_entropy = curr_overall_entropy
                best_split_col = col_index
                best_split_val = value
    return best_split_col, best_split_val

def split_data(data, split_col, split_value):
    split_col_values = data[:, split_col]
    data_below = data[split_col_values <= split_value]
    data_above = data[split_col_values > split_value]    

    return data_below, data_above


def getMajorityClass(classSizes, partitionSize):
    maxPurity = classSizes[0]/partitionSize
    majority_class = 0
    for i in range(len(classSizes)):
        purity = size/partitionSize
        if purity > maxPurity:
            maxPurity = purity
            majority_class = i
    return majority_class

def classify_data(data):
    if(data.shape[0] == 1):
        label_column = data[:,len(data) - 1]
    else:
        label_column = data[:,len(data[1])-1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

#Representation of Decision Tree: Dictionary
#Key: Question (petal width <= 0.8)
#Value: [yes_answer, no_answer]
def decision_tree(training, purity_thresh, min_leaves, counter=0):
    # data preparations
    if counter == 0:
        data = training.values
    else:
        data = training
    
    # base case
    if(check_purity(data, purity_thresh) or len(data) < min_leaves):
        classification = classify_data(data)
        return classification
    #recursive part
    else:
        counter += 1
        # helper functions
        potential_splits = get_potential_splits(data)
        if len(potential_splits) == 0:
            classification = classify_data(data)
            return classification
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        #instantiate subtree
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree(data_below, purity_thresh, min_leaves, counter)
        no_answer = decision_tree(data_above, purity_thresh, min_leaves, counter)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:        
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        return sub_tree

def classify_test(test_row,tree):
    if(isinstance(tree, str)):
        return tree
    question = list(tree.keys())[0]
    feature_name, comparison, val = question.split()
    if test_row[int(feature_name)] <= float(val):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    
    #base case
    if not isinstance(answer, dict):
        return answer
    #recursive part
    else:
        remaining_tree = answer
        return classify_test(test_row, remaining_tree)

def calculate_accuracy(test_df, tree):
    test_df["classification"] = test_df.apply(classify_test, axis=1, args=(tree,))    
    test_df["classification_correct"] = (test_df.classification == test_df.c_name)
    accuracy = test_df.classification_correct.mean()    

    return accuracy
    
def populateRecords(filename):
    records = []
    test_records = []
    overall_features = {}
    features = []
    counter = 0
    with open(filename, 'r') as f:
        for line in f:
            if counter == 0:
                counter += 1
                continue
            words = line.split('\t')
            r = Record()
            r.add_c_name(words[3])
            feature = get_features(words[14])
            for name in feature:
                if name not in overall_features:
                    overall_features[name] = 0
            features.append(feature)
            records.append(r)
            if(counter == 3000):
                break
            counter += 1
    for i in range(len(records)):
        vector = dict.fromkeys(overall_features, 0)
        for key in features[i]:
            vector[key] = features[i][key]
        vector = np.array(list(vector.values()))
        records[i].add_vector(vector)
    temp_records = records
    for i in range(900):
        random_record = random.choice(temp_records)
        temp_records.remove(random_record)
        test_records.append(random_record)
    return test_records

def create_df(records):
    vector_list = []
    c_name_list = []
    vector_length = len(records[0].vector)
    for record in records:
        vector_list.append(record.vector)
        c_name_list.append(record.c_name)        
    df = pd.DataFrame(vector_list, columns=range(1,vector_length+1))
    df["c_name"] = c_name_list 
    return df

def main():
    args = sys.argv[1:]
    if(args[0] == "-h"):
        entireHearing = True
    else:
        entireHearing = False
    
    if(entireHearing):
        #Handle this case
        print("Placeholder")
    else:
        filename = args[0]        
        records = populateRecords(filename)
        dataframe = create_df(records)
        num_labels = dataframe["c_name"].nunique()
        overall_num_records = dataframe.shape[0]
        train_data, test_data = split_sets(dataframe, 0.2)
        train_num = train_data.shape[0]
        test_num = test_data.shape[0]
        tree = decision_tree(train_data, 0.8, 10)
        print(tree)
        accuracy = calculate_accuracy(test_data, tree)
        committee_names = set(list(dataframe["c_name"]))
        print("Number of labels: {}".format(num_labels))
        print("Overall number of input records: {}".format(overall_num_records))
        print("Train size: {} records".format(train_num))
        print("Test size: {} records".format(test_num))
        print("Labels: ", end='')
        print(committee_names)
        print("Overall Accuracy: {}".format(accuracy)) 
        #Create dataframe with columns of each vector value and the label(c_name)
        #Split it into training and testing sets, pass in training set to decision tree algorithm        
if __name__ == '__main__':
    main()
