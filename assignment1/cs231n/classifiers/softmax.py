from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    # for i in range(num_train):
    #     loss[i] = -X[i].dot(W)[y[i]] + np.log(np.sum(np.exp(X[i].dot(W)),axis=1))

    for i in range(num_train):
        scores = X[i].dot(W)

        # Take care of Numeric stability: shift
        shift_scores = scores - np.max(scores)
        loss_i = -shift_scores[y[i]] +np.log(np.sum(np.exp(shift_scores)))
        loss += loss_i


        for j in range(num_classes):
            prob = np.exp(shift_scores[j])/ np.sum(np.exp(shift_scores))

            # https://zhuanlan.zhihu.com/p/21485970
            # chain rule derivation / Probablitstic interpretation


            if j == y[i]:
                dW[:,j] += (-1 + prob)*X[i]
            else:
                dW[:,j] += prob*X[i]


    loss /= num_train
    loss += reg * np.sum(W*W)
    dW = dW/num_train + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    shift_scores = scores - np.max(scores, axis=1).reshape(-1,1)

    softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)

    loss = -np.log(softmax_output[range(num_train),y])
    loss = np.mean(loss)
    loss += reg*np.sum(W*W)

    softmax_output_copy = softmax_output.copy()
    # softmax_output is N,C. minus 1 in i,y[i]
    adjust = np.zeros_like(softmax_output)
    adjust[range(num_train),y] = -1
    softmax_output_copy += adjust

    # (D,N) * (N,C); 
    dW = (X.T).dot(softmax_output_copy)
    dW = dW/num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
