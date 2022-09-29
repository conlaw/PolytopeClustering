import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpc_init import MPCInit
from cluster_metrics import *

'''
This file contains a bunch of helper functions for running MPC algorithm
'''

def z_to_labels(z):

    '''
    Helper function to go from 1-hot assignment matrix (1 col/cluster) to array of labels (1 entry/data)
    '''
    return (pd.melt(pd.DataFrame(z).reset_index(), id_vars = ['index'])
            .query('value == 1')
            .sort_values('index')['variable']
           )


def init_hps(X, k, card = 1, M = 1, metric = 'silhouette'):
    '''
    Helper function to create initial cluster assignment and separation
    '''
    
    #Fit initial clustering 
    clf = MPCInit(X, k, max_iter = 20, 
                                  cardinality = card,
                                  M = M,
                                  hp_gen = 'integer_linear')
    z, centers, w, b = clf.fit()
    labels = z_to_labels(z)
    
    if metric == 'silhouette':
        print('Starting sil (unfiltered): ', SilHouetteMetric.compute_metric_once(X, labels))
    elif metric == 'dunn':
        print('Starting dunn (unfiltered): ', DunnIndex.compute_metric_once(X, labels))
        
    eps = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        unique_vals = np.unique(np.sort(X[:,i]))
        if len(unique_vals) > 1:
            eps[i] = np.min(unique_vals[1:] - unique_vals[:-1] )
        else:
            eps[i]= 10

    #Pull out polytopes for all clusters
    polytope_A = {}
    polytope_b = {}
    polytope_ineq_A = {}
    polytope_ineq_b = {}

    for cl in np.unique(labels):
        polytope_A[cl] = []
        polytope_b[cl] = []
        polytope_ineq_A[cl] = []
        polytope_ineq_b[cl] = []

        for edge_2 in w.keys():
            if cl in edge_2:
                if edge_2[0] == cl:
                    polytope_A[cl].append(w[edge_2])
                    polytope_b[cl].append(round(b[edge_2],8))
                else:
                    polytope_ineq_A[cl].append(w[edge_2])
                    polytope_ineq_b[cl].append(round(b[edge_2],8))

        polytope_A[cl] = np.array(polytope_A[cl])
        polytope_b[cl] = np.array(polytope_b[cl])
        polytope_ineq_A[cl] = np.array(polytope_ineq_A[cl])
        polytope_ineq_b[cl] = np.array(polytope_ineq_b[cl])

    # Check which datapoints meet each polytope
    cl_membership = np.zeros(X.shape[0])-1
    total_membership = np.zeros(X.shape[0])


    for cl in np.unique(labels):
        
        if polytope_A[cl].shape[0] > 0:
            membership_eq = np.all((polytope_A[cl] @ X.T ).T + polytope_b[cl] >= 0,axis=1)
        else:
            membership_eq = np.ones(X.shape[0]).astype(np.bool)

        if polytope_ineq_A[cl].shape[0] > 0:
            membership_ineq = np.all((polytope_ineq_A[cl] @ X.T ).T + polytope_ineq_b[cl] < 0,axis=1)
        else:
            membership_ineq = np.ones(X.shape[0]).astype(np.bool)
            
        membership = membership_eq & membership_ineq
        
        total_membership = total_membership + membership
        cl_membership = np.where(membership.astype(np.int)*(cl+1) > 0, cl, cl_membership) 

    #Filter out unclassified points (will add them back in later)
    labels = pd.Series(cl_membership[cl_membership >= 0])
    X_filtered =  X[cl_membership < 0,:]
    X = X[cl_membership >= 0,:]
    print('Filtered points: ', X_filtered.shape[0])
    
    if len(np.unique(labels)) < 2:
        print('mono-cluster')
        return X, labels, {}, {}, X_filtered

    # Impute labels for data points falling into two clusters (points on edge)
    X, labels = imputeLabels(X[total_membership[cl_membership >= 0] < 2,:], 
                             cl_membership[cl_membership >= 0][total_membership[cl_membership >= 0] < 2],
                             X[total_membership[cl_membership >= 0] >= 2,:])
    
    
    if metric == 'silhouette':
        print('Starting sil (filtered): ', SilHouetteMetric.compute_metric_once(X, labels))
    elif metric == 'dunn':
        print('Starting dunn (filtered): ', DunnIndex.compute_metric_once(X, labels))

    return X, labels, w, b, X_filtered


def imputeLabels(X_fixed, labels_fixed, X_impute, metric='silhouette'):
    '''
    Assigns labels for points that fall outside of the polytopes
    '''
    
    #Go through each missing point and add it to the cluster that maximizes silhouette
    for i in range(X_impute.shape[0]):
        sil_score = -2
        for cl in np.unique(labels_fixed):
            if metric == 'silhouette':
                sil_score_new = SilHouetteMetric.compute_metric_once(np.concatenate([X_fixed, X_impute[i:(i+1),:]]), 
                                                 np.concatenate([labels_fixed, [cl]]))
            elif metric == 'dunn':
                sil_score_new = DunnIndex.compute_metric_once(np.concatenate([X_fixed, X_impute[i:(i+1),:]]), 
                                               np.concatenate([labels_fixed, [cl]]))
            if sil_score_new > sil_score:
                cl_final = cl
                sil_score = sil_score_new
        
        X_fixed = np.concatenate([X_fixed, X_impute[i:(i+1),:]])
        labels_fixed = np.concatenate([labels_fixed, [cl_final]])
    
    return X_fixed, pd.Series(labels_fixed)
