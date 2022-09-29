import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from cluster_metrics import *
from mpc_helpers import *
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import os, time

'''
Main function for running MPC algorithm
'''

def MPCPolytopeOpt(data, k, metric = 'silhouette', card = 1, M = 1, max_k = 10, verbose = False,
                  ):
    '''
    Main function to perform MPC and then do coordinate descent.
    Inputs:
    - data: data (row samples, column features) in numpy format
    - k: number of clusters for initialization schem
    - metric: Clustering metric to optimize
    - card: Number of non-zero coefficients in separating hyperplanes
    - M: Maximum integer value for separating hyperplanes
    - max_k: Maximum number of clusters that we can generate during local search
    - verbose: whether to print intermediary updates
    '''
    
    #Generate initial clustering + separating hyperplanes
    X, labels,w ,b, X_filtered = init_hps(data, k, card, M, metric=metric)
        
    #Set up module to compute metric
    if metric == 'silhouette':
        metricComp = SilHouetteMetric(X)
    elif metric == 'dunn':
        metricComp = DunnIndex(X)
    else:
        raise Exception('No associated metric module')

    changed = True
    edge_set = allowable_edges4(X.shape[1],card,M)
    
    #Get minimum separation for each feature
    eps = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        unique_vals = np.unique(np.sort(X[:,i]))
        if len(unique_vals) > 1:
            eps[i] = np.min(unique_vals[1:] - unique_vals[:-1] )
        else:
            eps[i]= 10
    
    #Keep trying local improvements until nothing leads to a performance bump
    # Could change this to be time limit
    while changed:
        changed = False
        max_cluster_label = np.max(labels)
        current_labels = np.unique(labels)
        last_silhouette = metricComp.compute_metric(labels)
        
        #Loop over existing edges
        for edge in w.keys():
            #Pull data points relevant to this edge
            labs = pd.Series(labels).copy()
            focus_points = labels.isin(list(edge))
            X_fixed = X[~focus_points,:]
            X_focus = X[focus_points,:]

            
            #Pull out polytopes for both clusters (minus edge we're moving)
            polytope_A = {}
            polytope_b = {}
            polytope_ineq_A = {}
            polytope_ineq_b = {}
            
            for cl in edge:
                polytope_A[cl] = []
                polytope_b[cl] = []
                polytope_ineq_A[cl] = []
                polytope_ineq_b[cl] = []

                
                for edge_2 in w.keys():
                    if edge_2 == edge:
                        continue
                    elif cl in edge_2:
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

            base_membership = {}

            for cl in edge:
                if (polytope_A[cl].shape[0] > 0) or (polytope_ineq_A[cl].shape[0] > 0):
                    if polytope_A[cl].shape[0] > 0:
                        membership_eq = np.all((polytope_A[cl] @ X_focus.T ).T + polytope_b[cl] >= 0,axis=1)
                    else:
                        membership_eq = np.ones(X_focus.shape[0]).astype(np.bool)

                    if polytope_ineq_A[cl].shape[0] > 0:
                        membership_ineq = np.all((polytope_ineq_A[cl] @ X_focus.T ).T + polytope_ineq_b[cl] < 0,axis=1)
                    else:
                        membership_ineq = np.ones(X_focus.shape[0]).astype(np.bool)

                    base_membership[cl] = membership_eq & membership_ineq
                else:
                    base_membership[cl] = np.ones(X_focus.shape[0]).astype(np.bool)

            #Pull current edge and labels for focus data points       
            w_edge = w[edge]
            b_edge = b[edge]
            labels_focus = labels[focus_points].copy().to_numpy()
            
            #Set up loop constants
            curr_silhouette = last_silhouette
            best_b = b_edge
            best_w = w_edge

            new_labels = labels_focus
            
            for new_w in edge_set:
                # Pull out potential list of thresholds b to look at
                hp_score = X_focus @ new_w
                potential_labels = -np.unique(hp_score)

                for b_new in potential_labels[np.argsort(np.abs(potential_labels - b_edge))]:
                    edge_incl = hp_score + b_new >= 0
                    focus_labels = np.where(base_membership[edge[0]] & edge_incl, edge[0], 
                                        np.where( base_membership[edge[1]] & ~edge_incl, edge[1],np.max(labels)+1))
                    labs[focus_points] = focus_labels
                    
                    if len(np.unique(focus_labels)) < 2:
                        continue 
                        
                    if len(np.unique(labs)) > max_k:
                        continue 

                    new_sil = metricComp.compute_metric(labs)
                    if new_sil > curr_silhouette:
                        best_w = new_w
                        best_b = b_new
                        curr_silhouette = new_sil
                        new_labels = labs.copy()
                        focus_labs = focus_labels.copy()
                        changed = True
            
            #If moving edge improved, update edge set
            if changed:
                if verbose:
                    print('New perf: ', curr_silhouette)
                
                #Update edges
                best_silhouette = curr_silhouette
                labels = new_labels.copy()
                b[edge] = best_b
                w[edge] = new_w

                dropped_clusters = [x for x in edge if x not in np.unique(new_labels)]
                new_clusters = [x for x in np.unique(new_labels) if x not in current_labels]
                
                if len(np.unique(new_clusters)) > 0:
                    w, b = create_new_cluster(w,b,edge,focus_labs,labels_focus, w_edge, b_edge)
    

                if len(dropped_clusters) > 0:
                    for cl in dropped_clusters:
                        keys = list(w.keys())
                        for key in keys:
                            if cl in key:
                                w.pop(key,None)
                                b.pop(key,None)

                break
            
        if changed:
            continue
        
        #Try splitting clusters
        num_cl = len(np.unique(labels))
        for cl in np.unique(labels):
            if len(np.unique(labels)) >= max_k:
                break 

            labels, w, b = splitCluster(cl, X, labels, metricComp, w, b, card, M)
            
        if len(np.unique(labels)) > num_cl:
            changed = True
            
        current_silhouette = metricComp.compute_metric(labels)      
        if current_silhouette - last_silhouette < 1e-7:
            print('minor improvement')
            print('Last sil', last_silhouette)
            print('Current sil', curr_silhouette)

            break
    
    #Impute filtered points
    X_final, label_final = imputeLabels(X, labels, X_filtered, metric = metric)

    return X_final, label_final, w, b


def splitCluster(cl, X, labels, metricComp, w, b, card = 1, M = 1):
    '''
    Function for evaluating whether there's an interpretable split of a cluster that leads to better perf.
    '''
    
    
    cl = int(cl)
    baseline_metric = metricComp.compute_metric(labels)
    
    labs = labels.copy()
    new_cluster_id = int(np.max(labels)+1)
    focus_points = labels == cl
    X_cl = X[labels == cl,:]
    best_edge = None
    best_metric = -1
    
    #Loop over possible edges
    for new_edge in allowable_edges4(X_cl.shape[1],card, M):
        hp_score = X_cl @ new_edge
        potential_labels = -np.unique(hp_score)
        
        # loop over possible intercepts
        for b_new in potential_labels:
            edge_incl = hp_score + b_new >= 0
            focus_labels = np.where(edge_incl, cl, new_cluster_id)
            labs[focus_points] = focus_labels

            if len(np.unique(labs)) < 2:
                continue 

            new_metric = metricComp.compute_metric(labs)

            if new_metric > best_metric:
                best_b = b_new
                best_edge = new_edge
                best_metric = new_metric
                new_labels = labs.copy()
                focus_labs = focus_labels
                changed = True
    
    #If best split is better than baseline, add new edges + clsuter
    if best_metric > baseline_metric:
        print('Splitting cluster ',cl)

        labels = new_labels
        w[(cl, new_cluster_id)] = best_edge
        b[(cl, new_cluster_id)] = best_b
        
        edges = list(w.keys())
        for edge_n in edges:
            if cl in edge_n:
                new_edge = (edge_n[0] if edge_n[0] != cl else new_cluster_id, 
                       edge_n[1] if edge_n[1] != cl else new_cluster_id)
                w[new_edge] = w[edge_n]
                b[new_edge] = b[edge_n]
    

    return new_labels, w, b

def create_new_cluster(w,b,edge,new_labels,labels_focus, w_edge, b_edge):
    '''
    Helper function to set-up new cluster if created by moving hyperplane
    '''
    #Pull out old label for new cluster
    old_lab = int(labels_focus[new_labels == np.max(new_labels)][0])
    new_lab = int(np.max(new_labels))
    
    w_new = {}
    b_new = {}
    
    #Create new edges for new pairs of cluster
    edges = list(w.keys())
    for edge_n in edges:
        if old_lab in edge_n:
            if edge_n == edge:
                new_edge = (edge_n[0] if edge_n[0] != old_lab else new_lab, 
                       edge_n[1] if edge_n[1] != old_lab else new_lab)

                w[new_edge] = w_edge
                b[new_edge] = b_edge

                new_edge = (edge_n[0] if edge_n[0] == old_lab else new_lab, 
                       edge_n[1] if edge_n[1] == old_lab else new_lab)

                w[new_edge] = w[edge_n]
                b[new_edge] = b[edge_n]

                continue

            new_edge = (edge_n[0] if edge_n[0] != old_lab else new_lab, 
                       edge_n[1] if edge_n[1] != old_lab else new_lab)
            w[new_edge] = w[edge_n]
            b[new_edge] = b[edge_n]
            
    return w, b


'''
Functions that compute edge sets to search over

'''
def allowable_edges(d, card):
    possible_edges = []
    
    for combo in list(combinations(range(d),card)):
        edge = np.zeros(d)
        edge[combo]  = 1
        possible_edges.append(edge)
        
    return possible_edges

def allowable_edges2(d, card):
    possible_edges = []
    
    for combo in list(combinations(range(d),card)):
        edge = np.zeros(d)
        edge[list(combo)]  = 1
        possible_edges.append(edge)
        edge = np.zeros(d)
        edge[list(combo)]  = -1
        possible_edges.append(edge)

        
    return possible_edges

def allowable_edges3(d, card, M= 1):        
    possible_edges = []
    
    if card == 1:
        for combo in list(combinations(range(d),card)):
            edge = np.zeros(d)
            edge[list(combo)]  = 1
            possible_edges.append(edge)
            
    elif card == 2:
        for combo in list(combinations(range(d),1)):
            edge = np.zeros(d)
            edge[list(combo)]  = 1
            possible_edges.append(edge)

        for combo in list(combinations(range(d),card)):
            for i in range(0,M+1):
                if i == 0:
                    continue
                for j in range(0,M+1):
                    if j == 0:
                        continue
                        
                    if (abs(i) == abs(j)) and abs(i) > 1:
                        continue
                    edge = np.zeros(d)
                    edge[list(combo)[0]]  = i
                    edge[list(combo)[1]]  = j
                    possible_edges.append(edge)

                    
        
    return possible_edges

def allowable_edges4(d, card, M= 1):        
    possible_edges = []
    
    if card == 1:
        for combo in list(combinations(range(d),card)):
            edge = np.zeros(d)
            edge[list(combo)]  = 1
            possible_edges.append(edge)
            edge = np.zeros(d)
            edge[list(combo)]  = -1
            possible_edges.append(edge)
            
    elif card == 2:
        for combo in list(combinations(range(d),1)):
            edge = np.zeros(d)
            edge[list(combo)]  = 1
            possible_edges.append(edge)
            edge = np.zeros(d)
            edge[list(combo)]  = -1
            possible_edges.append(edge)

        for combo in list(combinations(range(d),card)):
            for i in range(-M,M+1):
                if i == 0:
                    continue
                for j in range(-M,M+1):
                    if j == 0:
                        continue
                        
                    if (abs(i) == abs(j)) and abs(i) > 1:
                        continue
                    edge = np.zeros(d)
                    edge[list(combo)[0]]  = i
                    edge[list(combo)[1]]  = j
                    possible_edges.append(edge)

                    
        
    return possible_edges