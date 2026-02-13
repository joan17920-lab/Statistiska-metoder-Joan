import numpy as np
import scipy.stats as st

class LinearRegress:
    def __init__(self,X,y):
        self.X = X
        self.Y = y
        self.beta = None

# a least squared approximation of the mean 
    def lsm(self):
        self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
        return self
 

# A property  d that contains the number of features
    @property 
    def d (self):
        return None if self.X is None else int(self.X.shape[1]-1)

# Aproperty  n that contains the size of the sample    
    @property
    def n (self):
        return None if self.X is None else int(self.X.shape[0])
    
# A metod to calculat y hat:
    def predict (self,X = None):
        if self.beta is None:
            raise ValueError("Run lsm() first.")
        if X is None:
            X = self.X
        return X @ self.beta
    
# A method to calculate SSE
    def sse(self):
        y_hat = self.predict()
        return np.sum(np.square(self.Y - y_hat))

# A method to calculate MSE
    def mse(self):
        return self.sse()/self.n

# A method to calculate the Syy.
    def Syy(self):
        return (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n
    
# A method to calculate an unbiased estimator of the variance
    def var(self):
        return self.sse()/(self.n-self.d-1)
    
# A method to calculat the standard deviation 
    def std(self):
        return np.sqrt(self.var())

# A method to calculate the RMSE  
    def rmse(self):
        
        return np.sqrt(self.mse())

# A method for the significance of the regression.
    def f_test(self):         
        ssr = self.Syy() - self.sse()
        f_value = (ssr/self.d)/self.var()
        p_ftest = st.f.sf(f_value, self.d, self.n-self.d-1)
        return f_value, p_ftest

#  A method for the relevance of the regression (R2).
    def R2(self):
      return 1-self.sse()/self.Syy()
    
# Significance tests on individual variables
    def t_test(self):
        if self.beta is None:
            raise ValueError("Run lsm() first.")
        c = np.linalg.pinv(self.X.T @ self.X)*self.var()
        t_value = self.beta / np.sqrt(np.diag(c))
        p_ttest = 2 * st.t.sf(np.abs(t_value), self.n-self.d-1)
        return t_value, p_ttest
    
    
# The Pearson number between all pairs of parameters
    @staticmethod
    def corr(X):
        return np.corrcoef(X[:,1:], rowvar=False)

# Confidence intervals on individual parameters.
    def interval(self,alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        c = np.linalg.pinv(self.X.T @ self.X)*self.var()
        margin = st.t.isf(alpha/2, self.n-self.d-1)*np.sqrt(np.diag(c))
        lower = self.beta - margin
        upper = self.beta + margin
        return lower,upper
