import numpy as np

class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        self.dim = [d+1, 1]
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
        N,d = X.shape 
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #

        X_aug = self.gen_features(X)
#         print((-y * (X_aug @ self.w))) # (5000 * 1)
        
        loss = (1 / N) * np.sum(np.log( 1 + np.exp(-y * (X_aug @ self.w))))
        
#         temp = y / (np.exp(y * (X_aug @ self.w)) + 1)
#         print('The shape of temp is', temp.shape)
        
        grad = (-1 / N) * (X_aug.T @ (y / (np.exp(y * (X_aug @ self.w)) + 1)))
        
        
        
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
        
#         # this is unnecessary
#         y_pred_soft = 1 / (1 + (X.T @ self.w))
        
#         # hard decision
#         y_pred[y_soft > 0.5] = 1
#         y_pred[y_soft < 0.5] = -1
        
#         # break the tie randomly
#         s = np.random.binomial(1, .5, np.sum(y_soft == 0.5)) * 2 - 1
#         y_pred[y_soft == 0.5] = s
        
        y_lin = X.T @ self.w
        
        # hard decision
        y_pred[y_lin > 0] = 1
        y_pred[y_lin < 0] = -1
        
        # break the tie randomly
        s = np.random.binomial(1, .5, np.sum(y_lin == 0)) * 2 - 1
        y_pred[y_lin == 0] = s
    
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred