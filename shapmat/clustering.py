import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import shap

class shap_clustering:
    """
    Cluster PC1 and PC2 of CRC patients using k-mean clustering
    Args:
        - PC_scores: a dataframe of PC1 and PC2 
        - y: a series of labels with Patient_id as its indices
    """
    def __init__(self,PC_scores,y):
        self.PC_scores = PC_scores
        self.y = y 
        
    def get_CRC_only(self):
        """
        Get PC1 and PC2 of CRC patients only.
        """
        y = self.y
        PC_scores = self.PC_scores
        
        # get the Patient_ID of CRC patients
        CRC_idx = []
        for i in y.index:
            if y.loc[i] == 1:
                CRC_idx.append(i)

        PC_CRC = PC_scores.loc[CRC_idx]
        return PC_CRC
    
    def kmeans(self,n_clusters):
        PC_CRC = self.get_CRC_only()

        kmeans = KMeans(n_clusters=n_clusters
        , init='k-means++', max_iter=1000, n_init=10, random_state=0)
        
        y_pred_kmeans = kmeans.fit_predict(PC_CRC)
        cluster_labels = [y+1 for y in y_pred_kmeans]

        PC_CRC['cluster'] = [y+1 for y in kmeans.labels_]
        return PC_CRC