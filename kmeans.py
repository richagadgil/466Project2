import sys
import numpy as np
import re




class Record:

    cid = None
    c_name = None
    c_house = None
    hid = None
    pid = None
    diarization_id = None
    text = None


def main():
    args = sys.argv[1:]
    if not args or len(args) != 1:
        print("usage: main filename")
        sys.exit(1)
    filename = args[0]
    unique_words = []
    word_occurrences = {}
    with open(filename, 'r') as f:
        num_records = 0
        for line in f:
            if num_records == 0:
                num_records += 1
                continue
            words = line.split('\t')
            filtered = [word.strip() for word in words]
            filtered[:] = [x for x in filtered if x != '']
            text = re.sub('[^A-Za-z0-9 ]+', '', words[14]).lower().split()
            unique_by_record = set(text)
            for word in unique_by_record:
                if word in word_occurrences:
                    word_occurrences[word] += 1
                else:
                    word_occurrences[word] = 1 
            num_records += 1
    with open(filename, 'r') as f:
        counter = 1
        for line in f:
            if counter == 1:
                counter += 1
                continue
            if counter == 3:
                break
            words = line.split('\t')
            filtered = [word.strip() for word in words]
            filtered[:] = [x for x in filtered if x != '']
            text = re.sub('[^A-Za-z0-9 ]+', '', words[14]).lower().split()
            print(get_features(text, num_records,word_occurrences))
            counter += 1

def get_features(words,numRecords, wordOccurrences):
	bag_words = get_frequencies(words)
	tf_idf_vals = computeTF_IDF(bag_words, words, numRecords, wordOccurrences)
	vector_dict = {}
    for key in wordOccurrences.keys():
        if key in tf_idf_vals.keys():
            vector_dict[key] = tf_idf_vals[key] 
        else:
            vector_dict[key] = 0
    return vector_dict.values()

def get_frequencies(words):
    words_dict = {}
    for word in words:
        if word in words_dict:
            words_dict[word] += 1
        else:
            words_dict[word] = 1	
    return words_dict

def computeTF_IDF(bag_of_words, words, num_lines, occurrences_overall):
    tf_idf_values = {}
    for key in bag_of_words.keys():
        tf = bag_of_words[key] / len(words)
        tf_idf_values[key] = tf * np.log(num_lines / occurrences_overall[key])
    return tf_idf_values
	
if __name__ == '__main__':
    main()
