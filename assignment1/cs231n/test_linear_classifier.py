import numpy as np
from cs231n.classifiers.linear_classifier import *
from cs231n.gradient_check import grad_check_sparse
from cs231n import get_data

if __name__ == '__main__':

    X_train, y_train, X_test, y_test = get_data.get_data()
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    # W = np.random.randn(3073, 10) * 0.0001
    #
    # loss, grad = svm_loss_vectorized(W, X_train, y_train, 0.000005)
    # print('loss: %f' % (loss,))
    #
    # f = lambda w: svm_loss_vectorized(w, X_train, y_train, 0.0)[0]
    # grad_numerical = grad_check_sparse(f, W, grad)


    svm = LinearSVM()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4, batch_size=200,
                          num_iters=1000, verbose=True)

    y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
    y_val_pred = svm.predict(X_test)
    print('validation accuracy: %f' % (np.mean(y_test == y_val_pred), ))

