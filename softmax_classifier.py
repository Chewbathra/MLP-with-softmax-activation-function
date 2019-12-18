"""
Softmax Classifier with levelwise derivatives.

Author: Joao A. Candido Ramos

/!\IMPORTANT/!\:
This code is based on Assignment 1 given during the course "CS231n: 
Convolutional Neural Networks for Visual Recognition" at Stanford University.
"""
import numpy as np


class SoftmaxClassifier:
    def __init__(self, dims, num_classes, std=1e-3, random_seed=0):
        """
            
        """
        self.num_classes = num_classes
        if random_seed:
            np.random.seed(random_seed)
        self.W = std * np.random.randn(dims, num_classes)

    def forward(self, X):
        """
        Computes the forward pass, takes data as input and output probabilities.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.

        Returns:
        - The probabilities of each sample to become to a given class.
        """
        #######################################################################
        # TODO: Compute the scores and apply the softmax, scores should be
        # stored in self.scores and the result of the softmax in 
        # self.softmax_output
        #######################################################################
        self.scores = X.dot(self.W)
        
        # softmax
        exp = np.exp(self.scores)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return self.probs

    def loss(self, X, y, reg=0.0):
        """
        Computes the loss.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - reg: (float) regularization strength.

        Returns:
        - the loss
        """
        # get the number of samples
        N = X.shape[0]
        #######################################################################
        # TODO: Compute the loss. You should get the probabilities calling 
        # self.probs and not implementing the forward there again
        #######################################################################
        loss = np.sum(-np.log(self.probs[range(N), y])) / N + reg * np.sum(np.square(self.W)) 
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return loss

    def backward(self, X, y, reg, dP_dS=False):
        """
        Computes the gradients for each level of the graph.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - reg: (float) regularization strength.

        Returns:
        - DLoss / dWeights
        """
        scores = None
        N = X.shape[0]

        #######################################################################
        # TODO: Find the local derivatives and the gradients at each level.
        # Replace the np.zeros(1) values by your derivations.
        #######################################################################
        # y_onehot
        y_onehot = np.zeros((N, self.num_classes))
        y_onehot[np.arange(N), y] = 1 
        
        # LEVEL 1
        # dLoss / dProbs
        self.dLoss_dProbs = -np.divide(y_onehot, self.probs)
        
        # LEVEL 2
        # dProbs / dScores
        # IMPORTANT: This is not trivial to do in a vectorized manner you are
        # allowed to use as many loops as you want inside the if
        # IF YOU IMPLEMENT THIS DERIVATIVE WITHOUT LOOPS YOU HAVE A BONUS !!! =)
        #######################################################################
        # LOOP ZONE
        self.dProbs_dScores = np.zeros(1)
        if dP_dS:
            self.dProbs_dScores = np.empty((N, self.num_classes, self.num_classes))
            for k in range(N):
                q = self.probs[k]
                for i in range(len(q)):
                    for j in range(len(q)):
                        if i == j:
                            self.dProbs_dScores[k,i,j] = q[i] * (1-q[i])
                        else:
                            self.dProbs_dScores[k,i,j] = -q[i]*q[j]
#            pass
        # END OF LOOP ZONE
        #######################################################################

        # dLoss / dScores
        # Since it's not easy to vectorize dProbs / dScores, do not use the 
        # chain rule there !!! Prefer the analytical solution !
        self.dLoss_dScores = self.probs - y_onehot

        # LEVEL 3
        # dScores / dWeights
        self.dScores_dW = X

        # dLoss / dWeights (by chain rule)
        self.dLoss_dW = self.dScores_dW.T.dot(self.dLoss_dScores)
        
        # dScores / dX
        self.dScores_dX = self.W

        # dLoss / dX (by chain rule)
        self.dLoss_dX = self.dScores_dX.dot(self.dLoss_dScores.T).T
        
        # REGULARIZER
        # add regularization to the gradients of the weights
        self.dLoss_dW = self.dLoss_dW + reg * 2 * self.W
        
        # choose the good value for grads
        grads = self.dLoss_dW
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return grads

    def train(self, X, y, num_iters=100, learning_rate=0.0,
              reg=0.0, batch_size=10, verbose=True):
        """
        Train the softmax classifier using stochastic gradient descent (SGD).

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - X_val: A numpy array of shape (M, D) containing validation data (cf. X)
        - y_val: A numpy array of shape (M,) containing validation labels (cf. y)
        - num_iters: (integer) number of steps to take when optimizing
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Returns:
        - A dictionary with the history of losses and accuracies for training 
        and validation data
        """
        history = {}
        history["train_loss"] = []
        history["train_acc"] = []
        
        N, C = X.shape
        max_iterations = N // batch_size
        for it in range(1, num_iters+1):
            batch_indices = np.random.choice(N, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            probs = self.forward(X_batch)
            loss = self.loss(X_batch, y_batch, reg)
            grad = self.backward(X_batch, y_batch, reg)

            ###############################################################
            # TODO: optimize with stochastic gradient descent (SGD)
            ###############################################################
            self.W = self.W - learning_rate * grad
            ###############################################################
            #----------------------- END OF YOUR CODE ---------------------
            ###############################################################

            # Decay learning rate
            #learning_rate *= learning_rate_decay

            # keep a trace of training
            history["train_loss"].append(loss)
            history["train_acc"].append((self.predict(X) == y).mean())

            if verbose:
                print("[TRAIN] iteration {:03d}/{:03d}, loss: {:.5f}, accuracy: {:.5f}".format(it, num_iters, history["train_loss"][-1], history["train_acc"][-1]), end="\r")
        if verbose:
            print(''.ljust(150), end='\r')
        return history

    def predict(self, X):
        """
        Computes the predictions given the probs, basically find the index of 
        highest prob.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.

        Returns:
        - y_pred: A numpy array of shape (N, 1) containing the classes predicted
        """
        #######################################################################
        # TODO: use the forward pass to find the predictions.
        #######################################################################
        y_pred = np.argmax(self.forward(X), axis=1)
        #######################################################################
        #--------------------------- END OF YOUR CODE -------------------------
        #######################################################################
        return y_pred
