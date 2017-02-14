from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,VotingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import xgboost as xg
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 100
        self.n_neighbors = 6
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components-50)), 
            #('clf1', RandomForestClassifier(n_estimators=1000, random_state=42,max_features=self.max_features)),
            #('clf2', xg.XGBClassifier(nthread=1,n_estimators=500,max_depth=5,learning_rate=0.2,subsample=0.8,min_child_weight=2))
            ('clf', svm.SVC(C=1e5, probability=True))
        ])
        
        self.clf1 = Pipeline([
            ('pca', PCA(n_components=self.n_components-50)), 
            ('clf1', RandomForestClassifier(n_estimators=1000, random_state=42)),
            ('clf2', xg.XGBClassifier(nthread=1,n_estimators=300,max_depth=5,learning_rate=0.06))
            
        ])
        
        
        self.clf2 = Pipeline([
            ('pca', PCA(n_components=self.n_components-50)), 
            ('clf', ensemble.ExtraTreesClassifier(n_jobs=4, n_estimators=2000, max_features=20, min_samples_split=3,
                                    bootstrap=False, verbose=0, random_state=23))
            ])
        
        self.clf3 = Pipeline([
            ('pca', PCA(n_components=self.n_components-50)), 
            ('clf', KNeighborsClassifier(n_neighbors=self.n_neighbors)),
            ])
        
        self.clf4 = Pipeline([
            ('pca', PCA(n_components=self.n_components-50)), 
            ('clf', ensemble.BaggingClassifier(base_estimator=self.clf2,n_estimators=5))
            ])

    def fit(self, X, y):
        self.clf.fit(X,y)
        self.clf1.fit(X,y)
        self.clf2.fit(X,y)
        self.clf3.fit(X,y)
        X_new = self.clf.predict_proba(X)
        X1_new = self.clf1.predict_proba(X)
        X2_new = self.clf2.predict_proba(X)
        X3_new = self.clf3.predict_proba(X)
        X_f = np.concatenate((X,X_new,X1_new,X2_new,X3_new),axis=1)
        self.clf4.fit(X_f,y)

    def predict(self, X):
        X_new = self.clf.predict(X)
        X1_new = self.clf1.predict(X)
        X2_new = self.clf2.predict(X)
        X3_new = self.clf3.predict(X)
        X_f = np.concatenate((X,X_new,X1_new,X2_new,X3_new),axis=1)
        return self.clf4.predict(X_f)

    def predict_proba(self, X):
        X_new = self.clf.predict_proba(X)
        X1_new = self.clf1.predict_proba(X)
        X2_new = self.clf2.predict_proba(X)
        X3_new = self.clf3.predict_proba(X)
        X_f = np.concatenate((X,X_new,X1_new,X2_new,X3_new),axis=1)
        return self.clf4.predict_proba(X_f)
    
