import numpy as np
from cs231n.data_utils import load_CIFAR10




def get_data(num_training = 28000, num_val = 1000, num_test = 1000):

    # Load the raw CIFAR-10 data.
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print()
    #--------------------------------------------------------#
    mask = range(num_training, num_training + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    #------------------------------------------------------------#
    # Reshape the image data into rows
    # X_train = np.reshape(X_train, (X_train.shape[0], -1))
    # X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # print('Train and test shape')
    # print(X_train.shape, X_test.shape, '\n')

    return X_train, y_train, X_val, y_val, X_test, y_test
#------------------------------------------------------------
