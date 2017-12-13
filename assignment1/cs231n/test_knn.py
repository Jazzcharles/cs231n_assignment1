# Run some setup code for this notebook.

from __future__ import print_function
import random
import numpy as np

from cs231n.data_utils import load_CIFAR10

import matplotlib.pyplot as plt




'''
# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
'''

from cs231n.classifiers import KNearestNeighbor
from cs231n import get_data


def cal_standard_knn():
    # Create a kNN classifier instance.
    # Remember that training a kNN classifier is a noop:
    # the Classifier simply remembers the data and does no further processing
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    print('KNN Classifier Train Done\n')

    #------------------------------------------------------------

    # Open cs231n/classifiers/k_nearest_neighbor.py and implement
    # compute_distances_two_loops.

    # Test your implementation:
    print('Ready to test with 2 loops')
    #dists = classifier.compute_distances_two_loops(X_test)
    #print(dists.shape)

    print('Ready to test with 1 loop')
    #dists = classifier.compute_distances_one_loop(X_test)
    #print(dists.shape)

    print('Ready to test with 0 loop\n')
    dists = classifier.compute_distances_no_loops(X_test)
    print(dists.shape)

    #------------------------------------------------------------
    print('Ready to predict')
    y_pred = classifier.predict_labels(dists, 3)

    print('Accurarcy = %s' % np.mean(y_pred == y_test))

    #------------------------------------------------------------


def test_cross_validation(X_train, y_train):

    print('Ready to test with cross_validation')

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10]

    X_train_folds = []
    y_train_folds = []

    print('Train data shape = ' , X_train.shape)
    y_train = y_train.reshape(-1, 1)
    print('Train label shape = ' , y_train.shape)

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    k_to_accuracies = {}

    for each_k in k_choices:
        k_to_accuracies.setdefault(each_k, [])
        for i in range(num_folds):
            classfer = KNearestNeighbor()
            X_train_slice = np.vstack(X_train_folds[0: i] + X_train_folds[i+1: num_folds])
            y_train_slice = np.vstack(y_train_folds[0: i] + y_train_folds[i+1: num_folds])
            y_train_slice = y_train_slice.reshape(-1)
            #print('debug')
            #print(y_train_slice.shape)

            X_test_slice = X_train_folds[i]
            y_test_slice = y_train_folds[i]
            y_test_slice = y_test_slice.reshape(-1)
            #print(X_train_slice.shape)

            classfer.train(X_train_slice, y_train_slice)
            dis = classfer.compute_distances_no_loops(X_test_slice)
            y_predict = classfer.predict_labels(dis, each_k)


            acc = np.mean(y_predict == y_test_slice)
            k_to_accuracies[each_k].append(acc)

            #break
        #break


    for each_k in k_choices:
        for item in k_to_accuracies[each_k]:
            print('k = %d, acc = %f' %(each_k, item))



if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data.get_data()
    test_cross_validation(X_train, y_train)
