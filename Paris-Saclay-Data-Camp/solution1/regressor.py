from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np
import xgboost as xg
from sklearn import linear_model
from sklearn import ensemble


class Regressor(BaseEstimator):
    def __init__(self):
        self.n_components = 100                                                                                             
        self.learning_rate = 0.2                                                 
        self.list_molecule = ['A', 'B', 'Q', 'R']                                
        self.dict_reg = {}
        self.dict_reg1 = {}
        self.dict_reg2 = {}
        self.dict_reg3 = {}
        for mol in self.list_molecule:                                           
            self.dict_reg[mol] = Pipeline([                                      
                ('pca', PCA(n_components=self.n_components-90)),                    
                ('reg',xg.XGBRegressor(nthread=1,n_estimators=300,max_depth=5,subsample=0.9,learning_rate=0.2))    
            ])
            
            self.dict_reg1[mol] = Pipeline([                                      
                ('pca', PCA(n_components=self.n_components-90)),                    
                ('reg', ensemble.ExtraTreesRegressor(n_jobs=4, n_estimators=300,
                                    bootstrap=False, verbose=0, random_state=23,max_depth=8)),
            ]) 
            
            self.dict_reg3[mol] = Pipeline([                                      
                ('pca', PCA(n_components=self.n_components-90)),                    
                ('reg', ensemble.BaggingRegressor(xg.XGBRegressor(nthread=1,n_estimators=300,max_depth=5,subsample=0.9),
                                                  n_estimators=20,random_state=23))
            ]) 

    def fit(self, X, y):
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            XX_mol = X[ind_mol]                                                  
            y_mol = y[ind_mol].astype(float)                                     
            self.dict_reg[mol].fit(XX_mol, np.log(y_mol))
            self.dict_reg1[mol].fit(XX_mol, np.log(y_mol))
            X_new = np.exp(self.dict_reg[mol].predict(XX_mol))
            X1_new = np.exp(self.dict_reg1[mol].predict(XX_mol))
            X_new = np.reshape(X_new, (len(X_new), 1))
            X1_new = np.reshape(X1_new, (len(X1_new), 1))
            X_f = np.concatenate((XX_mol,X_new,X1_new),axis=1)
            self.dict_reg3[mol].fit(X_f, np.log(y_mol))
            
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]             
            XX_mol = X[ind_mol].astype(float)
            X_new = np.exp(self.dict_reg[mol].predict(XX_mol))
            X1_new = np.exp(self.dict_reg1[mol].predict(XX_mol))
            X_new = np.reshape(X_new, (len(X_new), 1))
            X1_new = np.reshape(X1_new, (len(X1_new), 1))
            X_f = np.concatenate((XX_mol,X_new,X1_new),axis=1)
            y_pred[ind_mol] = np.exp(self.dict_reg3[mol].predict(X_f))       
        return y_pred
