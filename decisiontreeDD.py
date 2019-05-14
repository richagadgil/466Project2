import sys
import os
import pandas as pd
import numpy as np
import random
from clusterDD import Record
from clusterDD import get_features
from clusterDD import pre_process

def split_flag(data, test_size, hearings_dict):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for name in hearings_dict:
        hearings = list(hearings_dict[name])
        test_size_scaled = round(test_size * len(hearings))
        if test_size_scaled == 0:
            test_size_scaled = 1
        curr_test_df = pd.DataFrame()
        curr_train_df = pd.DataFrame()
        for i in range(test_size_scaled):
            hearing_slice = data[data["hid"] == hearings[i]]
            curr_test_df = curr_test_df.append(hearing_slice)
        test_data = test_data.append(curr_test_df)
        for j in range(test_size_scaled, len(hearings)):
            hearing_slice = data[data["hid"] == hearings[j]]
            curr_train_df = curr_train_df.append(hearing_slice)
        train_data = train_data.append(curr_train_df) 
    data.drop('hid', axis=1, inplace=True)
    train_data.drop('hid', axis=1, inplace=True)
    test_data.drop('hid', axis=1, inplace=True)
    return train_data, test_data
        
def split_sets(data, test_size):
    data.drop('hid', axis=1, inplace=True)
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
    if (purity >= purity_thresh):
        return True
    else:
        return False

def get_potential_splits(data_values):
    potential_splits = {}
    _, n_cols = data_values.shape
    for col_index in range(n_cols - 1): 
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
        
def calculate_entropy(data):
    if(data.shape[0] == 1):
        label_column = data[:,len(data) - 1]
    else:
        label_column = data[:,len(data[1])-1]
    _, counts = np.unique(label_column, return_counts = True)
    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))    
    return entropy

def calculate_overall_entropy(data_below, data_above):
    num_points = len(data_below) + len(data_above)
    weight_below = len(data_below) / num_points
    weight_above = len(data_above) / num_points
    overall_entropy = (weight_below * calculate_entropy(data_below) + weight_above * calculate_entropy(data_above))
    return overall_entropy

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

def decision_tree(training, purity_thresh, min_leaves, counter=0):
    if counter == 0:
        data = training.values
    else:
        data = training
    if(check_purity(data, purity_thresh) or len(data) < min_leaves):
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data)
        if len(potential_splits) == 0:
            classification = classify_data(data)
            return classification
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}
        
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
    
    if not isinstance(answer, dict):
        return answer
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
            r.hid = words[5]
            feature = get_features(words[14])
            for name in feature:
                if name not in overall_features:
                    overall_features[name] = 0
            features.append(feature)
            records.append(r)
            if (counter == 5000):
                break
            counter += 1
    for i in range(len(records)):
        vector = dict.fromkeys(overall_features, 0)
        for key in features[i]:
            vector[key] = features[i][key]
        vector = np.array(list(vector.values()))
        records[i].add_vector(vector)
    return records

def create_df(records):
    vector_list = []
    c_name_list = []
    h_id_list = []
    vector_length = len(records[0].vector)
    for record in records:
        vector_list.append(record.vector)
        c_name_list.append(record.c_name)
        h_id_list.append(record.hid)        
    df = pd.DataFrame(vector_list, columns=range(1,vector_length+1))
    df["c_name"] = c_name_list
    df["hid"] = h_id_list 
    return df

def printPerCommittee(test_data, labels):
    for name in labels:
        filtered = test_data[test_data["classification"] == name]
        true_positive = sum(filtered["classification_correct"])
        false_positive = filtered.shape[0] - true_positive
        actuals = test_data[test_data["c_name"] == name]
        actuals = actuals[actuals["classification_correct"] == 0]
        false_negative = actuals.shape[0]
        if(true_positive + false_positive == 0):
            precision = 0
        else:
            precision = true_positive / float(true_positive + false_positive)
        if(true_positive + false_negative == 0):
            recall = 0
        else:
            recall = true_positive / float(true_positive + false_negative)
        if(precision + recall) == 0:
            f1 = 0
        else:
            f1 = (precision * recall) / float(precision + recall) 
            f1 = 2 * f1
        print("\nScores for {} committee:".format(name))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1 Score: {}".format(f1))       

def get_hearings_dict(records):
    hearings_dict = {}
    for record in records:
        if record.c_name not in hearings_dict:
            hearings_dict[record.c_name] = set(record.hid)
        else:
            hearings_dict[record.c_name].add(record.hid)
    return hearings_dict

def filter_records(records, hearings_dict):
    new_records = []
    for record in records:
        if len(hearings_dict[record.c_name]) > 1:
            new_records.append(record)
        else:
            del hearings_dict[record.c_name]
    return new_records 

def main():
    args = sys.argv[1:]
    if(args[0] == "-h"):
        filename = args[1]
        records = populateRecords(filename)
        hearings_dict = get_hearings_dict(records)
        records = filter_records(records, hearings_dict)
        dataframe = create_df(records)
        num_labels = dataframe["c_name"].nunique()
        overall_num_records = dataframe.shape[0]
        train_data, test_data = split_sets(dataframe, 0.2)
        train_num = train_data.shape[0]
        test_num = test_data.shape[0]
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
    accuracy = calculate_accuracy(test_data, tree)
    committee_names = set(list(dataframe["c_name"]))
    print("Number of labels: {}".format(num_labels))
    print("Overall number of input records: {}".format(overall_num_records))
    print("Train size: {} records".format(train_num))
    print("Test size: {} records".format(test_num))
    print("Labels: ", end='')
    print(committee_names)
    print("Overall Accuracy: {}".format(accuracy)) 
    printPerCommittee(test_data, committee_names)
    
if __name__ == '__main__':
    main()
