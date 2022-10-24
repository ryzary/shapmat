import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PCAplot():
    def __init__(self,PC_df,CEV_df,y,y_pred_crc_proba=None,correct_pred_only=True):
        self.PC_df = PC_df
        self.cev = CEV_df # cumulative explained variance
        
        if correct_pred_only:
            y_correct = y.loc[PC_df.index]
            self.y = y_correct
        
        else:
            self.y = y.values
        
        self.crc_proba = y_pred_crc_proba
        
    def plot_pca(self,figsize=(8,8),dpi=60,disease_label='CRC',savefig=False,output_path=None,title=None):
        fig = plt.figure(figsize=figsize,dpi=dpi) 
        
        scatter = plt.scatter(self.PC_df['PC1'],self.PC_df['PC2'],
                         s=60,c=self.y,cmap='jet',alpha=0.5)           

        handles,labels = scatter.legend_elements()
        legend = plt.legend(handles,labels,fontsize=25,loc='best') # ,bbox_to_anchor = (1.42, 1.03))
        legend.get_texts()[0].set_text('Control')
        legend.get_texts()[1].set_text(disease_label)
        
        # explained variance
        EV_PC1 = self.cev['Explained Variance'][0].round(2)
        EV_PC2 = self.cev['Explained Variance'][1].round(2)

        plt.xlabel(f'PC1 ({EV_PC1})',fontsize=25)
        plt.ylabel(f'PC2 ({EV_PC2})',fontsize=25)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(title,fontsize=40)

        if savefig:
            plt.savefig(output_path)

        plt.show()
    
    def plot_pca_with_proba(self,figsize=(8,8),dpi=60,savefig=False,output_path=None,title=None):
        """
        Plot PCA with Predicted CRC Probability as the colors
        """
        fig = plt.figure(figsize=(figsize[0]+4,figsize[1]),dpi=80)
        
        plt.scatter(self.PC_df['PC1'],self.PC_df['PC2'],
                         s=60,c=self.crc_proba,cmap='jet',alpha=0.5) 
        
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20) 
        
        # explained variance
        EV_PC1 = self.cev['Explained Variance'][0].round(2)
        EV_PC2 = self.cev['Explained Variance'][1].round(2)

        plt.xlabel(f'PC1 ({EV_PC1})',fontsize=25)
        plt.ylabel(f'PC2 ({EV_PC2})',fontsize=25)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(title,fontsize=40)

        if savefig:
            plt.savefig(output_path)
        plt.show()