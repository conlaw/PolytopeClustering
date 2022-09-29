from cluster_assignment import ClusterAssignment
from pairwise_integer_cut import PairwiseIntegerCut
import numpy as np


class MPCInit(object):
    '''
    Module for performing initialization scheme of MPC Algorithm
    '''
    
    
    def __init__(self, X, k, max_iter = 20, hp_gen = 'SVM', cardinality = 3, M=10):
        '''
        Inputs:
        - X: data (rows samples, columns features) in numpy format
        - k: number of clusters for initialization scheme
        - max_iter: Maximum number of iterations of alternating minimization

        '''
        
        #Init variables
        self.X = X
        self.n = X.shape[0]
        self.k = k
        self.max_iter = max_iter
        self.hp_gen = hp_gen
        
        #Initialize clustering and hyperplane modules
        self.cluster = ClusterAssignment(self.X, self.k)
        self.svm = PairwiseIntegerCut(self.X,cardinality=cardinality,M=M)

        
    
    def fit(self):
        
        #Init variables 
        self.xi = None #set xi as none so first clustering is just kmeans
        self.z = np.zeros((self.n, self.k))
        self.i = 0

        while self.i < self.max_iter:
            #iteratively assign clusters and define them with hps
            self.centers, z = self.cluster.cluster(self.xi)
            self.xi, self.w, self.b = self.svm.solve(z, self.centers)
            
            #Convergence criteria is cluster assignments dont change
            if np.array_equal(z, self.z):
                break
            else:
                self.z = z
                
            self.i = self.i+1
        
        #Computes cluster and svm loss (use it for reporting)
        self.cluster.computeLoss(self.z, self.centers, self.xi)
        
        return self.z, self.centers, self.w, self.b
    
    