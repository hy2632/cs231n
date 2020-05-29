from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        # scores is f(xi,W) = Wxi
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            # for other categories, accumulate the loss
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                # Look at the expression for the gradient.
                # Transpose to columns cuz j/yi in range(C)
                dW[:,y[i]] -= X[i].T
                dW[:,j] += X[i].T
        # loss L is the function of f(xi;W)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss =np.zeros(y.shape)
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    delta = 1.0 
    scores = X.dot(W)
    # X.dot(W) has a shape of (N,C) and we need to map the scores of real lable using y (N,)
    # how to map the elements in a matrix using another vector?
    # See numpy tutorial : a[np.arange(a.shape[0]),b]


    correct_class_scores  = scores[np.arange(scores.shape[0]),y].reshape(-1,1) #(N,1)

    # Another way is, the score of real label must be the maximum in each row.
    margins = np.maximum(0,scores-correct_class_scores +delta)
    margins[np.arange(margins.shape[0]),y] = 0 # the real labels have no loss
    loss = np.sum(margins,axis=1).mean() + reg*np.sum(W*W)





    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dW is a (D,C) matrix. When label is not correct label, dWi =  sum(I(margini>0))*-Xi
    # when correct label, dWi = I(margini>0)*Xi
    # So design a coeff_mat (N,C) which marks margin >0 and whether correct label


    num_train = X.shape[0]
    num_classes = W.shape[1]
    coeff_mat = np.zeros((num_train,num_classes))
    coeff_mat[margins>0] = 1 
    coeff_mat[np.arange(num_train),y] = -np.sum(coeff_mat,axis=1)

    dW = (X.T).dot(coeff_mat)
    # X.T (D,N) , coeff_mat (N,C), dW (D,C)

    dW = dW/num_train + 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
