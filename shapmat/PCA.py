from sklearn import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class customPCA:
    def __init__(self,X,crc_proba,n_components=2,scale=False):
        self.X = X
        self.crc_proba = crc_proba
        self.scale = scale
        
    def PCA(self,n_components=2):
        pca = decomposition.PCA(n_components=n_components,random_state=0)
        
        if self.scale:
            scalled = StandardScaler().fit_transform(self.X)
            pca.fit(scalled)
        else:
            pca.fit(self.X)
        return pca
    
    def PCA_scores(self):
        pca = self.PCA()
        if self.scale:
            scores = pca.transform(StandardScaler().fit_transform(self.X))
        else:
            scores = pca.transform(self.X)
        
        scores_df = pd.DataFrame(scores,columns=['PC1','PC2'],index=self.X.index)

        return scores_df
        
    def cumulative_explained_variance(self):
        """
        Return cumulative explained variance of PC1 and PC2
        """
        pca = self.PCA()
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

        pc_df = pd.DataFrame(['PC1', 'PC2'], columns=['PC'])
        explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
        cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])

        explained_cumulative_var = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
        
        return explained_cumulative_var