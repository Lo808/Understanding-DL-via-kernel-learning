import torch

class KernelRegressor:

    def __init__(self,kernel_func,sigma=1.0):

        self.kernel_func=kernel_func
        self.sigma=sigma
        self.mse_loss= lambda x,y: torch.cdist(x,y,p=2)

    def invers_fit(self,X_train,y_train):
        """
        Solve for f* using the closed form solution of kernel regression.
        This method require to inverse the kernel matrix and might be computationally expensive.
        In the paper, it is stated that direct methods always provide an highly accurate interpolation.
        """
        self.X_train=X_train
        self.y_train=y_train
        y_onehot = torch.nn.functional.one_hot(y_train.long(),10).float()

        # Compute Kernel matrix used to find f*
        self.K=self.kernel_func(self.X_train,self.X_train,self.sigma)

        # Add small regularization for numerical stability
        self.K+=1e-8*torch.eye(self.K.size(0)) 

        # Inverse to find weights
        self.alpha=torch.linalg.solve(self.K,y_onehot)

    def optim_fit(self,X_train,y_train,lr=1e-3,n_iters=1000):
        """
        Solve for f* using gradient descent optimization.
        Use EigenPro to compute gradient more efficiently for kernels like in paper
        https://github.com/EigenPro
        """
        pass

    def predict(self,X_test):
        """
        Use the calculated weights to predict on new data
        """
        if self.alpha is None:
            raise ValueError("Model not fitted yet")
        
        K_test=self.kernel_func(X_test,self.X_train,self.sigma)
        return K_test@self.alpha
    
    def compute_rkhs_norm(self):
        """
        Compute the RKHS norm of the learned function f*
        
        """
        if self.alpha is None:
            raise ValueError("Model not fitted yet")
        
        # Use the fomula of the paper to compute the RHKS norm
        interaction_matrix=self.alpha.T @ self.K @ self.alpha

        return torch.sqrt(torch.trace(interaction_matrix))
    
    
    




    