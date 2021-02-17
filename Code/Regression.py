import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            
            # X = [x1, x2, x3, ..., xN]'
            # X = [[1, 1, 1, ..., 1],
            #      [x1, x2, x3, ..., xN]]'
            
            add_all_one_feature = np.ones(N)
            
            X_out[:,0] = add_all_one_feature
            
            X_out[:,1:] = X
            
#             print('X is', X)
           
#             print('X_out is', X_out)
            
#             print('X_out shape is', X_out.shape)
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            
            X = X.reshape((-1,)) # convert (N, 1) to (N,)
#             print('X shape is', X.shape)
            for i in range(m + 1):
                X_out[:,i] = np.power(X, i)
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #
            
            X_aug = self.gen_poly_features(X)

            y = y.reshape(-1, 1)

            y_pred = self.predict(X)

            loss = ((y_pred - y).T @ (y_pred - y) / N)[0]  + (self.reg / 2) * (self.w.T @ self.w)[0]

            grad = (-2 / N) * X_aug.T @ (y - X_aug @ self.w) + self.reg * self.w
                
#             X_aug = self.gen_poly_features(X) # X is subsampled
#             y = y.reshape(-1, 1)              # y is subsampled 
#             # get the prediction
#             y_pred = self.predict(X)         
#             # calculate the loss on the subsampled dataset
#             loss = ((np.transpose(y_pred - y) @ (y_pred - y)) / N)[0]  
#             # calculate the grad
#             grad = -(2 / N) * np.transpose(X_aug) @ (y - X_aug @ self.w)
       
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
            X_aug = self.gen_poly_features(X)
            
            y = y.reshape(-1, 1)
            
            y_pred = self.predict(X)
            
            loss = ((y_pred - y).T @ (y_pred - y) / N)[0] + (self.reg / 2) * (self.w.T @ self.w)[0]
            
            # calculate the grad
            grad = (-2 / N) * X_aug.T @ (y - X_aug @ self.w) + self.reg * self.w
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
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
                
                # sample indices without replacement
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
                
                # compute the loss and gradient
                # loss_and_grad will take responsible for these
                
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                
                # NOTE: the loss above is the loss on the subsampled dataset
                # we need to find the loss on the whole dataset
                # use the emp_loss will make the loss history less oscillated
                y_pred = self.predict(X)
                y = y.reshape(-1,1)
#                 loss = ((np.transpose(y_pred - y) @ (y_pred - y)) / N)[0] 
                loss = ((y_pred - y).T @ (y_pred - y) / N)[0] + (self.reg / 2) * (self.w.T @ self.w)[0]
                
                # update the weights 
                # NOTE: the grad here is based on minibatch-SGD
                
                self.w = self.w - eta * grad
                
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #

            y = y.reshape(-1, 1)
            
            # w = (X'X)^{-1}X'y = X^{+}y if (X'X) is invertible, i.e., X has linearly independent columns
            
            X_aug = self.gen_poly_features(X)
            
            reg_term = np.identity(X_aug.shape[1]) * (self.reg * N / 2)
            
            self.w = np.linalg.inv((X_aug.T @ X_aug) + reg_term) @ X_aug.T @ y
            
            y_pred = self.predict(X)
            
            loss = ((y_pred - y).T @ (y_pred - y) / N)[0] + (self.reg / 2) * (self.w.T @ self.w)[0]
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            
            y = y.reshape(-1, 1)
            
            X_aug = self.gen_poly_features(X)
            
            reg_term = np.identity(X_aug.shape[1]) * (self.reg * N / 2)
            
            self.w = np.linalg.inv(X_aug.T @ X_aug + reg_term) @ X_aug.T @ y
            
            y_pred = self.predict(X)
            
            loss = ((y_pred - y).T @ (y_pred - y) / N)[0] + (self.reg / 2) * (self.w.T @ self.w)[0]
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            
            # first we should augment the data matrix X 
            # add an additional feature to each instance and set it to one
            
            X_aug = self.gen_poly_features(X)
            
            # prediction
            
            y_pred = X_aug @ self.w
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #

            X_aug = self.gen_poly_features(X)
            
            y_pred = X_aug @ self.w
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred