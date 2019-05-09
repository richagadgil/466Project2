import sys
import os


def split_sets():
    #Split data into training and testing
    return []

def findPurity(classSizes, partitionSize):
    maxPurity = classSizes[0]/partitionSize
    for size in classSizes:
        purity = size/partitionSize
        if purity > maxPurity:
            maxPurity = purity
    return purity

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

def pre_process(text):
    #Pre processing code from kmeans

def main():
    args = sys.argv[1:]
    if(args[2] == "-h"):
        entireHearing = True
    else:
        entireHearing = False
    
    if(entireHearing):
        #Handle this case
    else:
        text = sys.argv[2]
        text = pre_process(text)
        getFeatures(text)        
    
if __name__ == '__main__':
    main()