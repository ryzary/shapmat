import shap
import pandas as pd
import numpy as np

class Explainer:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
    
    def explainer(self):
        model = self.model
        explainer = shap.TreeExplainer(model, data=self.X)
        return explainer
        
    def shap(self):
        """
        Return an array of shap values  
        """
        explainer = self.explainer()
        shap_values = explainer.shap_values(self.X)
        
        if len(shap_values) == 2: # for RF model
            shap_values = shap_values[1] # choose class 1 (CRC)
        return shap_values
    
    def nonzero_mean_shap(self,df_shap):
        """
        Filter out features that have zero mean(|shap|).
        Return a dataframe of SHAP values
        """
        df_shap_abs = abs(df_shap)
    
        mean_per_col = df_shap_abs.mean(0)
        nonzero_mean = mean_per_col[mean_per_col!=0]
        
        nonzero_shap_features = list(nonzero_mean.index)
        df_shap_filtered = df_shap[nonzero_shap_features]
        
        print(f"Number of features with nonzero mean(|SHAP|): {len(nonzero_shap_features)}/{len(self.X.columns)}")  
        return df_shap_filtered

    def shap_df(self,correct_pred_only=True,filter_zero_column=True):
        """
        Return a dataframe SHAP values with bacteria names as the column
        """
        shap_values = self.shap()
        column_names = self.X.columns
        patient_ids = self.X.index
        
        df_shap = pd.DataFrame(shap_values, columns=column_names, index=patient_ids)
        
        if correct_pred_only:
            y_pred = self.model.predict(self.X)
            
            is_prediction_correct = []
            for pred, label in zip(y_pred, self.y):
                if pred == label:
                    is_prediction_correct.append(True)
                else:
                    is_prediction_correct.append(False)

            df_shap['correct_pred'] = is_prediction_correct
            
            df_shap = df_shap[df_shap['correct_pred']==True]
            df_shap = df_shap.drop('correct_pred',axis=1)
        
        if filter_zero_column:
            df_shap = self.nonzero_mean_shap(df_shap)
        
        return df_shap

