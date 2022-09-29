from sklearn.svm import LinearSVC
import numpy as np
import docplex.mp.model as cpx

class PairwiseIntegerCut(object):
    '''
    Module for finding interpretable separating hyperplane. Inputs are:
    
    - X: Data (rows are samples, columns are features) in numpy format
    - cardinality: (beta in formulation on overleaf) maximum l0 norm of hyperplane (non-zero coeffs)
    - M: M in formulation on overleaf, maximum integer coefficient for hyperplane   
    '''
    
    
    def __init__(self, X, cardinality = 3, M = 10):
        #initialize problem params
        self.X = X
        self.d = self.X.shape[1]
        self.n = self.X.shape[0]
        self.cardinality = cardinality
        self.M = M
        
        #Compute minimum separations
        self.eps = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            unique_vals = np.unique(np.sort(X[:,i]))
            if len(unique_vals) > 1:
                self.eps[i] = np.min(unique_vals[1:] - unique_vals[:-1] )
            else:
                self.eps[i]= 10
            
    def solve(self, z, centers):
        
        #Pull out num. of clusters
        k = z.shape[1]
        
        #Init dictionaries
        w_dict = {}
        b_dict = {}
        xi_dict = {}
        
        for i in range(k-1):
            for j in range(i+1, k):
                #Check that clusters are non-empty
                if (z[:,i].sum() == 0) or (z[:,j].sum() == 0):
                    w_dict[i,j] = np.zeros(self.d)
                    b_dict[i,j] = 0
                    xi_dict[i,j] =  np.zeros((self.n,2))

                else:
                    #Get training samples for svm ij
                    X_ij = np.concatenate([self.X[z[:,i].astype(np.bool),:], self.X[z[:,j].astype(np.bool),:]])
                    y_ij = np.concatenate([np.ones(int(z[:,i].astype(np.bool).sum())),
                                           -1*np.ones(int(z[:,j].astype(np.bool).sum()))])
                    
                    #Fit cuts 
                    #Can probs save computation by just pulling xis from ip model
                    w, b = self.findCut(X_ij, y_ij)
                    w_dict[i, j] = w
                    b_dict[i, j] = b
                    
                    #Compute distance from hp
                    xis = np.zeros((self.n,2))
                    
                    score = (self.X @ w + b)
                    xis[:,0] = np.clip(-score, 0, None)
                    
                    #For second class make hyperplane non-inclusvie
                    score = (self.X @ w + b + np.abs(self.eps @ w))
                    xis[:,1] = np.clip(score, 0, None)

                    xi_dict[i,j] = xis
                    
        return xi_dict, w_dict, b_dict
    
    def findCut(self, X_ij, y_ij):
        
        #Set up cplex
        model = cpx.Model(name = "parallel_cut")
        
        n, d = X_ij.shape
        
        # Define variables
        
        #W captures hyperplane coefficients
        w_var = {}
        w_var_neg = {}

        w_var_abs = {}
        w_abs_const = {}
        
        #Z capture what variables are non-zero
        self.z_var = {}
        self.z_var_neg = {}
        
        #Need to add eps
        for i in range(d):
            #Initialize two w/z (one positive one negative) per dim
            w_var[i] = model.integer_var(lb = 0, ub = self.M,name = "w_{0}".format(i))
            w_var_neg[i] = model.integer_var(lb = 0, ub = self.M,name = "w_neg_{0}".format(i))

            self.z_var[i] = model.binary_var(name = "z_{0}".format(i))
            self.z_var_neg[i] = model.binary_var(name = "z_neg_{0}".format(i))
        
        #B captures intercept of hyperplane
        b_var = model.continuous_var(lb = -10000,ub = 10000, name = "b")
        
        
        eps_var = {}
        
        #init dvars
        for i in range(n): 
            eps_var[i] = model.continuous_var(lb = 0, name = "eps_{0}".format(i))

        # Cardinality constraint (l0 norm)
        card_const_ub = model.add_constraint(ct = model.sum(self.z_var[i]+self.z_var_neg[i] for i in range(d)) <= self.cardinality,
                                          ctname = "card_const")
        
        #Remove trivial solution
        card_const_lb = model.add_constraint(ct = model.sum(self.z_var[i]+self.z_var_neg[i] for i in range(d)) >= 1,
                                          ctname = "card_const")
        card_const_w = model.add_constraint(ct = model.sum(w_var[i]+w_var_neg[i]for i in range(d)) >= 1,
                                          ctname = "card_const_w")

        
        
        #Constraints on maximum values of w coefficients
        z_consts_ub = {}
        z_consts_neg_ub = {}
        z_joint_const = {}
        
        w_abs_consts_ub = {}
        w_abs_consts_lb = {}

        for i in range(d):
            z_consts_ub[i] = model.add_constraint(ct = w_var[i] <= self.M*self.z_var[i], ctname = "z_const_upper_{0}".format(i))
            z_consts_ub[i] = model.add_constraint(ct = w_var_neg[i] <= self.M*self.z_var_neg[i], ctname = "z_neg_const_upper_{0}".format(i))
            z_joint_const[i] = model.add_constraint(ct = self.z_var_neg[i]+self.z_var[i] <= 1, ctname = "z_joint_{0}".format(i))


        #Constraints to track misclassification
        eps_consts = {}
        
        for i in range(n):
            if y_ij[i] > 0.5:
                eps_consts[i] =  model.add_constraint(ct = model.sum(w_var[j]*X_ij[i,j]-w_var_neg[j]*X_ij[i,j] for j in range(d)) + b_var >= -eps_var[i],
                                                      ctname = 'eps_const_{0}'.format(i))
            else:
                eps_consts[i] =  model.add_constraint(ct = model.sum(w_var[j]*(X_ij[i,j] + self.eps[j])-w_var_neg[j]*(X_ij[i,j]+ self.eps[j]) for j in range(d)) + b_var <= eps_var[i],
                                                      ctname = 'eps_const_{0}_1'.format(i))
                eps_consts[i] =  model.add_constraint(ct = model.sum(w_var[j]*(X_ij[i,j] - self.eps[j])-w_var_neg[j]*(X_ij[i,j]- self.eps[j]) for j in range(d)) + b_var <= eps_var[i],
                                                      ctname = 'eps_const_{0}_2'.format(i))
        
        
        #Objective is to minimize misclassifications
        objective = model.sum(eps_var[i] for i in range(n))
       
        #Solve model
        model.minimize(objective)
        model.solve()
        
        #Extract solution
        w = np.zeros(d)
        for i in range(d):
            w[i] = w_var[i].solution_value - w_var_neg[i].solution_value
        b = b_var.solution_value
        
        return w, b