import numpy as np
from random import shuffle
import math
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass

  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)
  # new_scores as exp, make them negative to avoid overflow
  new_scores = scores - np.reshape(np.max(scores, axis=1), (-1, 1))
  sum_expr = np.sum(np.exp(new_scores), axis=1, keepdims=True)
  prob = np.exp(new_scores) / sum_expr     # N * C
  y_mat = np.zeros_like(prob)   # N * C
  y_mat[np.arange(num_train), y] = 1.0

  for i in range(num_train):
    for j in range(num_classes):
      loss += -y_mat[i, j] * np.log(prob[i, j])
      dW_each = X[i] * (prob[i, j] - y_mat[i, j])
      '''
      if j == y[i]:
        dW[i, j] = -X[i] + prob[i, j] * X[i]
      else:
        dW[i, j] = prob[i, j] * X[i]
      '''
    dW += dW_each

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * np.sum(W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)
  # new_scores as exp, make them negative to avoid overflow
  new_scores = scores - np.reshape(np.max(scores, axis=1), (-1, 1))
  sum_expr = np.sum(np.exp(new_scores), axis=1, keepdims=True)
  prob = np.exp(new_scores) / sum_expr  # N * C --- prob[i, j] = e**[i, j] / sigma(e**i)
  y_mat = np.zeros_like(prob)
  y_mat[np.arange(num_train), y] = 1.0

  loss += -y_mat * (np.log(prob))
  dW += np.dot(X.T, prob - y_mat)


  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * np.sum(W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

