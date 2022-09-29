from sklearn.cluster import KMeans
import numpy as np
import docplex.mp.model as cpx
import pandas as pd

class ClusterAssignment(object):
    '''
    Module for the clustering sub-problem of MPC initialization scheme
    '''
    
    def __init__(self, X, k, max_iter = 1000, convergence_tol = 1e-10, random_state = 42, 
                 learning_rate = 0.0001, lb_perc = 0.05, lam = 1e20):
        
        '''
        Inputs
        - X: data (rows samples, columns features) in numpy format
        - k: Number of clusters for init. scheme
        - 
        '''
        
        #Init variables
        self.X = X
        self.n = self.X.shape[0]
        self.k = k
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.learning_rate = learning_rate
        self.lam = lam
        self.seed = random_state
        self.lower_bound = max(int(0.05*self.n),1)
        self.centers= None
        
        #Set up constraints of clustering problem
        self.initAssignmentModel()

    
    def initAssignmentModel(self):
        #Set up cplex
        self.assignModel = cpx.Model(name = "cluster_assignment")
        
        # Define variables
        self.z_var = {}
        self.u_var = {}

        #init dvars
        for k in range(self.k): 
            self.u_var[k] = self.assignModel.binary_var(name = "u_{0}".format(k))

            for t in range(self.n): 
                self.z_var[t, k] = self.assignModel.binary_var(name = "z_{0}_{1}".format(t, k))

        # Constraints
        self.constraint_z = {}
        self.constraint_bounds = {}

        # sum_{k} z_neg_{t,k} = 1
        for t in range(self.n): 
            self.constraint_z[t] = self.assignModel.add_constraint(ct = \
                                                            self.assignModel.sum(self.z_var[t, k] for k in range(self.k) ) == 1,
                                                            ctname = "constraint_z_{0}".format(t))

        # u_neg_{k} * lb <= sum_{t} z_neg_{t,k}
        for k in range(self.k): 
            self.constraint_bounds[k, 1] = self.assignModel.add_constraint(ct = \
                                        self.assignModel.sum(self.z_var[t, k] for t in range(self.n) ) <= self.u_var[k] * self.n,
                                    ctname = "constraint_bound_{0}_{1}".format(k, 1))

            self.constraint_bounds[k, 0] = self.assignModel.add_constraint(ct = \
                                        self.assignModel.sum(self.z_var[t, k] for t in range(self.n) ) >= self.u_var[k] * self.lower_bound,
                                    ctname = "constraint_bound_{0}_{1}".format(k, 1))
            
        
        self.assignModel.add_constraint(ct = self.assignModel.sum(self.u_var[k] for k in range(self.k)) >= self.k, ctname = "nontrivial_cluster")
        
    def initCluster(self, method = 'KMeans'):
        #Initialize cluster assignments
        if method == 'KMeans':
            clf = KMeans(n_clusters=self.k, random_state=self.seed, n_init = 100).fit(self.X)
            centers = clf.cluster_centers_
            labels = pd.get_dummies(clf.predict(self.X)).to_numpy()
        elif method == 'random':
            centers = self.X[np.random.choice(self.n, self.k, replace=False), :]
        else:
            raise ValueError('Specified initial clustering method not implemented.')
        
        return centers, labels
    
    def cluster(self, xi):
        
        #If we haven't done cluster assignments, generate using initialization schem
        if (self.centers is None) or (xi is None):
            self.centers, self.z = self.initCluster()
            self.computeLoss(self.z, self.centers, None)
            return self.centers, self.z
        
        iterNum = 1
        
        while iterNum < self.max_iter:
            #Iterate assigning clusters + updating centers
            z, u  = self.assignClusters(self.centers, xi)
            new_centers = self.updateCenters(self.centers,z)
            
            if np.sum(z != self.z) < 1:
                break
            else:
                self.centers = new_centers
                self.z = z
            iterNum += 1
        

        #print('Total iterations ', iterNum)
        self.computeLoss(z, self.centers, xi)
        
        return self.centers, z
    
    def updateCenters(self, centers, assignments):
        
        #Update cluster centers
        new_centers = np.zeros(centers.shape)
        
        for k in range(self.k):
            X_k = self.X[assignments[:,k].astype(np.bool),:]
            c = centers[k]
            
            #Only update if there are points assigned to it
            if X_k.shape[0] > 0:
                #new_centers[k,:] = self.updateCenter(c, X_k)
                new_centers[k,:] = self.updateCenterMean(c, X_k)
            else:
                new_centers[k,:] = centers[k,:]
                    
        return new_centers
    
    def updateCenter(self,c, X, tol = 1e-6, max_iteration = 1000): 

        # Initialization
        current_c = c

        # learning rate
        eta = self.learning_rate
        grad_value = self._gradient_psi(current_c, X)
        norm_square_grad = np.sum(np.square(grad_value))
        iteration = 0
        
        #perform gradient updates using gradient descent
        while (norm_square_grad > tol) and (iteration <= max_iteration): 
            grad_value = self._gradient_psi(current_c, X)
            norm_square_grad = np.sum(np.square(grad_value))
            current_c = current_c - eta * grad_value
            iteration = iteration + 1

        return current_c
    
    def updateCenterMean(self,c, X, tol = 1e-6, max_iteration = 1000): 
        #update centers using mean 
        return X.mean(axis=0)

    def _gradient_psi(self,current_c, X): 
        # sum_{t in set_1} 2 * (c - x^t)
        return 2 * np.sum(current_c - X, 0)

    def assignClusters(self, centers, xi):
        
        #print(xi[0,1])
        
        #print(xi.keys())
            
        distances = self._compute_cluster_distances(self.X, centers)
        
        objective = self.assignModel.sum(self.assignModel.sum( ( self.z_var[t, k] * distances[t,k] ) for t in range(self.n) ) for k in range(self.k) ) + \
        self.assignModel.sum( self.assignModel.sum(self.assignModel.sum( (self.lam*self.z_var[t,i]*xi[i,j][t,0] + self.lam*self.z_var[t,j]*xi[i,j][t,1]) for t in range(self.n)) for j in range(i+1, self.k)) for i in range(self.k-1))
        
        self.assignModel.minimize(objective)

        # Solve
        self.assignModel.solve()
        
        # Pull out solution
        z_sol = np.zeros((self.n, self.k))
        u_sol = np.zeros(self.k)

        for k in range(self.k): 
            u_sol[k] = self.u_var[k].solution_value

            for t in range(self.n):  
                z_sol[t][k] = self.z_var[t, k].solution_value
        
        #print(u_sol)
        return z_sol, u_sol

        
    def _compute_cluster_distances(self,data, cluster_centers):
        distances = np.zeros((self.X.shape[0], cluster_centers.shape[0]))

        for i in range(self.X.shape[0]):
            for j in range(cluster_centers.shape[0]):
                distances[i,j] = np.linalg.norm(data[i,:]-cluster_centers[j,:], ord=2)

        return distances
    
    def computeLoss(self, z, centers, xi):
        #clustering loss
        distances = self._compute_cluster_distances(self.X, centers)
        self.cl_loss = sum([ sum([z[t,k]*distances[t,k] for t in range(self.n)]) for k in range(self.k)] )
        
        #Representation loss
        if xi is not None:
            self.svm_loss = sum([sum([ sum([z[t,i]*xi[i,j][t,0]+z[t,j]*xi[i,j][t,1] for t in range(self.n)]) for j in range(i+1, self.k)] ) for i in range(self.k-1)])
        else:
            self.svm_loss = -1
        

