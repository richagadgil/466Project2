import sys
import numpy
import re
import random
import numpy as np
from sklearn.cluster import KMeans


class Record:

    cid = None
    c_name = None
    c_house = None
    hid = None
    pid = None
    diarization_id = None
    text = None


    def __init__(self):
        pass
    
    def add_cid(self, cid):
        self.cid = cid
    
    def add_text(self, text):
        self.text = text


def main():
    args = sys.argv[1:]

    if not args or len(args) != 1:
        print("usage: main filename")
        sys.exit(1)
    
    filename = args[0]
    unique_words = []
    records = [] #total records


    with open(filename, 'r') as f:
        D = []
        counter = 0
        for line in f:
            if counter == 3:
                break
            words = line.split('\t') #split words along tabs
            print(words)
            filtered = [word.strip() for word in words] #strip whitespace 
            filtered[:] = [x for x in filtered if x != '']
            text = re.sub('[^A-Za-z0-9 ]+', '', words[14]).lower().split() #remove punctuation/special chars from words
            get_features(text) 
            unique_words += (text)
            counter += 1

            if(counter > 1): #add all records to record list EXCEPT FOR HEADER
                record = Record() #create new Record
                record.add_cid(int(words[2])) #add CID & Text
                record.add_text(words[14])

                records.append(record) # append to record list 

            


            
    print(len(records))
    
   
    unique_words = list(set(unique_words))

    print(len(unique_words))


def get_features(words):
    bag_of_words = get_frequencies(words)
    tf_values = computeTF(bag_of_words, words)
    print(tf_values)

def get_frequencies(words):
    words_dict = {}
    for word in words:
        if word in words_dict:
            words_dict[word] += 1
        else:
            words_dict[word] = 1	
    return words_dict

def computeTF(bag_of_words, words):
    tf_values = {}
    for key in bag_of_words.keys():
        tf = bag_of_words[key] / len(words)
        tf_values[key] = tf
	
    return tf_values


if __name__ == '__main__':
    main()
