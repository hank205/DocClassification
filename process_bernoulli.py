from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from collections import Counter
import pickle
import itertools
from sklearn.metrics import confusion_matrix

# Plot method from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# python process.py rt
# python process.py fisher
def main():
    if len(sys.argv) != 2:
        print('Usage: python %s data_folder' % (sys.argv[0]))
        exit()
    data_folder = sys.argv[1]

    # load trained dictionaries
    positive = {}
    negative = {} 
    with open('pos_' + data_folder + '_train_dict.pkl', 'rb') as f:
        positive = pickle.load(f)
    with open('neg_' + data_folder + '_train_dict.pkl', 'rb') as f:
        negative = pickle.load(f)

    with open('pos_'+data_folder+'_train_count.pkl', 'rb') as f:
        pos_count = pickle.load(f)
    with open('neg_'+data_folder+'_train_count.pkl', 'rb') as f:
        neg_count = pickle.load(f)

    pos_class_prob = float(pos_count)/(pos_count+neg_count)
    neg_class_prob = float(neg_count)/(pos_count+neg_count)

    # transform number of occurence to probability with Laplace Smoothing
    positive_sum = 0.0
    negative_sum = 0.0
    for x in positive:
        positive_sum += positive[x]
    for x in negative:
        negative_sum += negative[x]  

    unique_counter = Counter(positive) + Counter(negative)
    unique_words = len(unique_counter)

    diff_pos = set(negative.keys()) - set(positive.keys())
    diff_neg = set(positive.keys()) - set(negative.keys())
    for x in diff_pos:
        positive[x] = 0.0
    for x in diff_neg:
        negative[x] = 0.0

    for x in positive:
        positive[x] = (positive[x]+1.0) / (positive_sum+unique_words)
    for x in negative:
        negative[x] = (negative[x]+1.0) / (negative_sum+unique_words)

    # print positive
    # print negative

    test_count = 0
    correct_count = 0
    test = []
    pred = []

    # read training data
    with open(data_folder + '_test.txt', 'r') as f:
        for i, line in enumerate(f):
            test_count+=1
            # positive review
            if line[:1] == '1':
                test.append(1)
                test_dict = dict(x.split(':') for x in line[2:].split(' '))
                # format string to int type
                for x in test_dict:
                    test_dict[x] = int(test_dict[x])

                # guess test_dict is pos or neg
                pos_prob = np.log(pos_class_prob)
                neg_prob = np.log(neg_class_prob)

                for x in positive:
                    if x in test_dict:
                        pos_prob += np.log(positive[x])
                    else:
                        pos_prob += np.log(1-positive[x])

                for x in negative:
                    if x in test_dict:
                        neg_prob += np.log(negative[x])
                    else:
                        neg_prob += np.log(1-negative[x])

                # evaluate correct or not
                if pos_prob > neg_prob:
                    correct_count += 1
                    pred.append(1)
                else:
                    pred.append(-1)


            # negative review
            else:
                test.append(-1)
                test_dict = dict(x.split(':') for x in line[3:].split(' '))
                for x in test_dict:
                    test_dict[x] = int(test_dict[x])

                # guess test_dict is pos or neg
                pos_prob = np.log(pos_class_prob)
                neg_prob = np.log(neg_class_prob)

                for x in positive:
                    if x in test_dict:
                        pos_prob += np.log(positive[x])
                    else:
                        pos_prob += np.log(1-positive[x])

                for x in negative:
                    if x in test_dict:
                        neg_prob += np.log(negative[x])
                    else:
                        neg_prob += np.log(1-negative[x])

                # evaluate correct or not
                if neg_prob > pos_prob:
                    correct_count += 1
                    pred.append(-1)
                else:
                    pred.append(1)
        


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
            odds_ratio[x] = positive[x]/negative[x]#max(positive[x]/negative[x], negative[x]/positive[x])

    print ('top 10 words regards odds_ratio class1/class2:')
    for l in Counter(odds_ratio).most_common(10):
        print(l[0], end=' ')
    print()

  

    # print and draw confusion matrix
    class_names = ['class1','class2']

    cnf_matrix = confusion_matrix(test, pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix '+data_folder)


    plt.savefig('confusion_matrix_'+data_folder+'_bernoulli.png', bbox_inches='tight')



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))