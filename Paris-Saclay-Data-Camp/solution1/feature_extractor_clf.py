import numpy as np
import pandas as pd
from sklearn import ensemble, feature_extraction, preprocessing

# import pandas as pd


class FeatureExtractorClf():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        # transform counts to TFIDF features
        tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
        XX = tfidf.fit_transform([np.array(dd) for dd in X_df['spectra']]).toarray()
        XX = preprocessing.normalize(XX)
        XX = np.log(XX+1)
        return XX 
