import sys
import os
import pandas as pd
import numpy as np
from kmeans import Record


def split_sets(data, test_size):
    #Split data into training and testing
    if isinstance(test_size, float):
        test_size = round(test_size * len(data))
    indices = data.index.tolist()
    test_indices = random.sample(population = indices, k=test_size)

    test_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    return train_data, test_data

def findPurity(data):
    label_col = 'c_name'
    return (data[label_col].value_counts(normalize=True))[0]

#input: training dataframe.values
def get_potential_splits(data_values):
    potential_splits = {}
    _, n_cols = data_values.shape
    for col_index in range(n_cols - 1): # "-1 because last column is gonna be labels"
        potential_splits[col_index] = []
        vals = data[:, col_index]
        unique_vals = np.unique(vals)
        
        for index in range(len(unique_vals)):
            if index != 0:
                curr_val = unique_vals[index]
                prev_val = unique_vals[index-1]
                potential_split = (curr_val + prev_val)/2
                potential_splits[col_index].append(potential_split)
    return potential_splits        

def calculate_entropy(data):
    label_column = data["c_name"]
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

def decisionTree(training, leaf_size, purity_thresh):
    partitionSize = len(training)
    classSizes = []
    # compute class sizes and put them in classSizes array 
    purity = findPurity(classSizes, partitionSize)
    if (partitionSize <= leaf_size) or (purity >= purity_thresh):
        majority_class = getMajorityClass(classSizes, partitionSize)
        # create leaf node labeled as majority_class
        return
    #Do Rest of code
            
def getFeatures(record):
    #Get features code from kmeans
    return[]
def pre_process(text):
    #Pre processing code from kmeans
    return []
def main():
    #IMPLEMENT USING PANDAS DATAFRAME
    print("Hello")
    args = sys.argv[1:]
    if(args[2] == "-h"):
        entireHearing = True
    else:
        entireHearing = False
    
    if(entireHearing):
        #Handle this case
    else:
        text = sys.argv[2]        
        #Process text, create Record objects and populate with feature vectors
        #Create dataframe with columns of each vector value and the label(c_name)
        
if __name__ == '__main__':
    main()
