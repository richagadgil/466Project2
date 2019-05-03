Digital Democracy Data Sets (for Project 2 groups)
The ordering for the tsv as well as what each represents (committee_utterance file)

1)vid                   - Video Id
2)fileid                - YouTube Id
3)cid                   - Committee Id
4)c_name                - Committee Name
5)c_house               - Committee House
6)hid                   - Hearing Id
7)position              - Order of bills discussed in the hearing/video order
8)pid                   - Person Id
9)diarization_id        - Id of a speaker given perfect diarization
10)last                 - Speaker last name
11)first                - Speaker first name
12)start                - Start time of utterance
13)end                  - End time of utterance
14)utterance_order      - Order of the utterance in a given hearing
15)text                 - Text of the utterance

DDDataSet_3.tsv

Use this data set for your baseline machine learning analysis of speaker identification. There are about 10,000 randomly selected utterance records that meet our criteria (legislator only, at least 5 seconds and at least 28 words). 
It's a tab-separated file withe following fields per line (in order)

document id
last name
first name
person id (pid)
YouTube ID
start time (offset of the video in seconds)
end time (offset of the video in seconds)
text of the utterance

