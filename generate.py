import numpy as np
import sys
import time
from collections import Counter
import pickle

# python generate.py rt
# python generate.py fisher
def main():
    if len(sys.argv) != 2:
        print('Usage: python %s data_folder' % (sys.argv[0]))
        exit()
    data_folder = sys.argv[1]

# read training data
    positive = {}
    negative = {}
    positive_count = 0
    negative_count = 0
    pos_counter = Counter()
    neg_counter = Counter()
    
    with open(data_folder + '_train.txt', 'r') as f:
        for i, line in enumerate(f):
            # positive review
            if line[:1] == '1':
                positive_count += 1
                dict_str = line[2:]
                dict_ = dict(x.split(':') for x in dict_str.split(' '))
                for x in dict_:
                    dict_[x] = int(dict_[x])
                pos_counter +=  Counter(dict_)
            # negative review
            else:
                negative_count += 1
                dict_str = line[3:]
                dict_ = dict(x.split(':') for x in dict_str.split(' '))
                for x in dict_:
                    dict_[x] = int(dict_[x])
                neg_counter +=  Counter(dict_)
        
        positive = dict(pos_counter)
        negative = dict(neg_counter)


    # print Counter(positive).most_common(10)
    # print Counter(negative).most_common(10)
    # print len(positive)
    # print len(negative)
    # print positive
    # print negative
    with open('pos_'+ data_folder + '_train_dict.pkl', 'wb') as f:
        pickle.dump(positive,f)
    with open('neg_'+ data_folder + '_train_dict.pkl', 'wb') as f:
        pickle.dump(negative,f)

    # with open('pos_rt-train_count.pkl', 'wb') as f:
    #     pickle.dump(positive_count,f)
    # with open('neg_rt-train_count.pkl', 'wb') as f:
    #     pickle.dump(negative_count,f)

    
# # read test data
#     positive = {}
#     negative = {}
#     positive_count = 0
#     negative_count = 0
    
#     pos_counter = Counter()
#     neg_counter = Counter()
#     # read training data
#     with open('rt-test.txt', 'r') as f:
#         for i, line in enumerate(f):
#             # positive review
#             if line[:1] == '1':
#                 positive_count += 1
#                 dict_str = line[2:]
#                 dict_ = dict(x.split(':') for x in dict_str.split(' '))
#                 for x in dict_:
#                     dict_[x] = int(dict_[x])
#                 pos_counter +=  Counter(dict_)
#             # negative review
#             else:
#                 negative_count += 1
#                 dict_str = line[3:]
#                 dict_ = dict(x.split(':') for x in dict_str.split(' '))
#                 for x in dict_:
#                     dict_[x] = int(dict_[x])
#                 neg_counter +=  Counter(dict_)
        
#         positive = dict(pos_counter)
#         negative = dict(neg_counter)

#     with open('pos_rt-test_dict.pkl', 'wb') as f:
#         pickle.dump(positive,f)
#     with open('neg_rt-test_dict.pkl', 'wb') as f:
#         pickle.dump(negative,f)

#     with open('pos_rt-test_count.pkl', 'wb') as f:
#         pickle.dump(positive_count,f)
#     with open('neg_rt-test_count.pkl', 'wb') as f:
#         pickle.dump(negative_count,f)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))