import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd

'''
Helper classes for computing clustering metrics
'''

class SilHouetteMetric(object):
    
    def __init__(self,X):
        self.distance = distance_matrix(X, X, p = 2)
    
    @staticmethod
    def compute_metric_once(X, labels):
        distance = distance_matrix(X, X, p = 2)
        
        if len(np.unique(labels)) < 2:
            return -1

        assignment = pd.get_dummies(labels).to_numpy()
        
        cluster_dists =np.matmul(distance,assignment/(assignment.sum(axis=0)+0.00001))
        adjustment_multipliers=assignment*(assignment.sum(axis=0)/(assignment.sum(axis=0)-1+0.000001))
        adjustment_multipliers = np.where(adjustment_multipliers == 0, 1, adjustment_multipliers)
        cluster_dists_corrected = cluster_dists*adjustment_multipliers
        cluster_counts = assignment.sum(axis=0)

        silhouette_scores = np.zeros(assignment.shape)
        for cl in range(assignment.shape[1]):
            if (cluster_counts[cl]  == 0):
                continue
            intra_clust = cluster_dists_corrected[:,cl]
            inter_clust = cluster_dists_corrected[:,[i for i in range(assignment.shape[1]) if (i != cl) and (cluster_counts[i] > 0)]].min(axis=1)
            silhouette_scores[:,cl] = (inter_clust - intra_clust)/np.max(np.array([inter_clust,intra_clust]),axis=0)
        
        return np.sum(silhouette_scores*assignment)/assignment.shape[0]

    def compute_metric(self, labels):
        if len(np.unique(labels)) < 2:
            return -1

        assignment = pd.get_dummies(labels).to_numpy()
        
        cluster_dists =np.matmul(self.distance,assignment/(assignment.sum(axis=0)+0.00001))
        adjustment_multipliers=assignment*(assignment.sum(axis=0)/(assignment.sum(axis=0)-1+0.000001))
        adjustment_multipliers = np.where(adjustment_multipliers == 0, 1, adjustment_multipliers)
        cluster_dists_corrected = cluster_dists*adjustment_multipliers
        cluster_counts = assignment.sum(axis=0)

        silhouette_scores = np.zeros(assignment.shape)
        for cl in range(assignment.shape[1]):
            if (cluster_counts[cl]  == 0):
                continue
            intra_clust = cluster_dists_corrected[:,cl]
            inter_clust = cluster_dists_corrected[:,[i for i in range(assignment.shape[1]) if (i != cl) and (cluster_counts[i] > 0)]].min(axis=1)
            silhouette_scores[:,cl] = (inter_clust - intra_clust)/np.max(np.array([inter_clust,intra_clust]),axis=0)
        
        return np.sum(silhouette_scores*assignment)/assignment.shape[0]
    
    

class DunnIndex(object):
    
    def __init__(self,X):
        self.distance = distance_matrix(X, X, p = 2)
    
    @staticmethod
    def compute_metric_once(X, labels):
        distance = distance_matrix(X, X, p = 2)
        assignment = pd.get_dummies(labels).to_numpy()
        cl_similarity_matrix = np.matmul(assignment,assignment.T)
        return distances[cl_similarity_matrix < 0.5].min()/distances[cl_similarity_matrix > 0].max()
    
    def compute_metric(self, labels):
        if len(np.unique(labels)) < 2:
            return -1
            
        assignment = pd.get_dummies(labels).to_numpy()

        distances = self.distance #[index_ordering,:][:,index_ordering]
        cl_similarity_matrix = np.matmul(assignment,assignment.T)
        
        return distances[cl_similarity_matrix < 0.5].min()/distances[cl_similarity_matrix > 0].max()