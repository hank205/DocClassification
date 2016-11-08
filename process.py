from __future__ import print_function
import numpy as np
import sys
import time
from collections import Counter
import pickle

# python process.py rt
# python process.py fisher
def main():
    if len(sys.argv) != 2:
        print('Usage: python %s data_folder' % (sys.argv[0]))
        exit()
    data_folder = sys.argv[1]

    positive = {}
    negative = {} 
    with open('pos_' + data_folder + '_train_dict.pkl', 'rb') as f:
        positive = pickle.load(f)
    with open('neg_' + data_folder + '_train_dict.pkl', 'rb') as f:
        negative = pickle.load(f)    
    # positive_count = 0
    # negative_count = 0
    # with open('pos_rt-train_count.pkl', 'rb') as f:
    #     positive_count = pickle.load(f)        
    # with open('neg_rt-train_count.pkl', 'rb') as f:
    #     negative_count = pickle.load(f)
    
    positive_sum = 0.0
    negative_sum = 0.0
    for x in positive:
        positive_sum += positive[x]
    for x in negative:
        negative_sum += negative[x]  

    # transform number of occurence to probability
    for x in positive:
        positive[x] /= positive_sum
    for x in negative:
        negative[x] /= negative_sum

    # print positive
    # print negative

    test_count = 0
    correct_count = 0
    
    # read training data
    with open(data_folder + '_test.txt', 'r') as f:
        for i, line in enumerate(f):
            test_count+=1
            # positive review
            if line[:1] == '1':
                test_dict = dict(x.split(':') for x in line[2:].split(' '))
                for x in test_dict:
                    test_dict[x] = int(test_dict[x])

                # guess test_dict is pos or neg
                pos_prob = 0
                neg_prob = 0
                for x in test_dict:
                    if (x not in positive) or (x not in negative):
                        continue
                    pos_prob += np.log(positive[x])
                    neg_prob += np.log(negative[x])

                # evaluate correct or not
                if pos_prob > neg_prob:
                    correct_count += 1


            # negative review
            else:
                test_dict = dict(x.split(':') for x in line[3:].split(' '))
                for x in test_dict:
                    test_dict[x] = int(test_dict[x])

                # guess test_dict is pos or neg
                pos_prob = 0
                neg_prob = 0
                for x in test_dict:
                    if (x not in positive) or (x not in negative):
                        continue
                    pos_prob += np.log(positive[x])
                    neg_prob += np.log(negative[x])

                # evaluate correct or not
                if neg_prob > pos_prob:
                    correct_count += 1
        


    # print 'correct_count', correct_count
    # print 'test_count', test_count
    print ('accuracy:', 100.0*correct_count/test_count, '%')
    
    print ('class1 top 10 words:')
    for l in Counter(positive).most_common(10):
        print(l[0], end=' ')
    print()
    print ('class2 top 10 words:')
    for l in Counter(negative).most_common(10):
        print(l[0], end=' ')
    print()


    odds_ratio = {}
    for x in positive:
        if x in negative:
            odds_ratio[x] = max(positive[x]/negative[x], negative[x]/positive[x])

    print ('top 10 words regards odds_ratio:')
    for l in Counter(odds_ratio).most_common(10):
        print(l[0], end=' ')
    print()




if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))