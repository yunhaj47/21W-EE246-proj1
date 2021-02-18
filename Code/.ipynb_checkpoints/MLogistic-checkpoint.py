import numpy as np

class MLogistic(object):
    def __init__(self, dim=[10,784], reg_param=0):
        """"
        Inputs:
          - dim: dimensions of the weights [number_classes X number_features]
          - reg : Regularization type [L2,L1,L]
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        dim[1] += 1
        self.dim = dim
        self.w = np.zeros(self.dim)
        
    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        
        add_all_one_feature = np.ones(N)
            
        X_out[:,0] = add_all_one_feature
            
        X_out[:,1:] = X

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out  
        
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        
        X_aug = self.gen_features(X)
        
        N,d = X_aug.shape
        
        
        # some auxiliary results that may need
        lin = self.w @ X_aug.T        
        exp = np.exp(lin)
        denorm = np.sum(exp, axis=0)
        denorm = denorm.reshape((1, -1))
        
        # softmax is the weights matrix for the combination of x_i's
        # it contains the "probability"
        
        softmax = exp / denorm 
        
        # compute the loss
        
        loss_component = softmax[y, range(softmax.shape[1])]
        loss += np.sum(np.log(loss_component)) / (-N)
        
        # compute the gradient
        
        # find the position that we want to subtract 1.0
        
        softmax[y, range(softmax.shape[1])] -= 1.0
        
        grad += (softmax @ X_aug)
        
        grad /= N
    
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
                
                batch_idx = np.random.choice(N, batch_size, replace = False)
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss = 0.0
                grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
                
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w = self.w - eta * grad
             
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        
        X_aug = self.gen_features(X)
        
        output = X_aug @ self.w.T
        
        y_pred = np.argmax(output, 1)
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred