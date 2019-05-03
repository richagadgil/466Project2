import sys
import numpy
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


    with open(filename, 'r') as f:
        D = []
        for line in f:
            words = line.split('\t')
            filtered = [word.strip() for word in words]
            filtered[:] = [x for x in filtered if x != '']
            unique_words += words[14].lower().replace("[^A-Za-z0-9]+", "").split()

    unique_words = list(set(unique_words))

    print(unique_words)


    
    #Document corpus: all the records
    #number unique words in corpus
    #for each record calculate TF and IDF
    #
    



def get_features(record):
  pass


if __name__ == '__main__':
  main()
