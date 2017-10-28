from dataclass import dataclass

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
#import xgboost
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



import pandas as pd
import numpy as np




class ai:
    def __init__(self, data, algo, par1, par2):
        self.algo=algo
        self.par1=par1
        self.par2=par2
        self.ai = None
        self.dataset = data
        if( self.algo == "SVC"):
            self.ai = aiclass_supportvector_classifier( )
        if( self.algo == "GNBC"):
            self.ai = aiclass_gaussiannb_classifier( )
        if( self.algo == "NNR"):
            self.ai = aiclass_nn_regressor( par1 )
        if (self.algo == "NNC"):
            self.ai = aiclass_nn_classifier( par1 )
        if( self.algo == "NNRR"):
            self.ai = aiclass_nn_radius_regressor( par1 )
        if (self.algo == "NNRC"):
            self.ai = aiclass_nn_radius_classifier( par1 )
        if (self.algo == "XGBR"):
            self.ai = aiclass_xgboost_regressor()
        if (self.algo == "XGBC"):
            self.ai = aiclass_xgboost_classifier()
        if (self.algo == "XGBRC"):
            self.ai = aiclass_xgboost_regclassifier( par1, par2 )

    def seterrfunc(self, x):
        self.ai.errorfunc = x

    def plotprediction(self):
        title = self.algo +  " " + str(self.par1) + " " + str(self.par2) + " " + " ".join( self.ai.dataset.getnames() )
        plotequity( self.res, self.ai.dataset.outofsample_res , title)

    def plottestandprediction(self):
        title = self.algo + " " + str(self.par1) + " " + str(self.par2) + " " + " ".join(self.dataset.getnames())
        plotequity(self.test_res.append( self.res ) , self.dataset.learn_res.append( self.dataset.outofsample_res), title)

    def fit(self):
        self.ai.fit(self.dataset.insample_data, self.dataset.insample_res)
    def predict(self, x=None):
        self.res = self.ai.predict(self.dataset.outofsample_data)
        return self.res
    def predict_data(self, x):
        return self.ai.predict(x)
    def test(self):
        self.testres = self.ai.test(self.dataset.learn_data)
        return self.testres
    def score(self):
        self.scoring = roc_auc_score(self.res, self.dataset.outofsample_res)
        return self.scoring
    def plot(self):
        self.ai.plot()


class aiclass:
    errorfunc = "auc"
    def __init__(self, data):
        self.ai = None
    def fit(self, indata, inres):
        self.ai.fit(indata, inres)
    def test(self, data):
        test_res = pd.DataFrame(self.ai.predict(data), columns=["Res"],index=data)
        test_cor = np.corrcoef(np.asarray(test_res["Res"]), np.asarray(self.dataset.learn_res["Res"]) )[0,1]
        return test_cor
    def predict(self, data):
        pres = self.ai.predict(data)
#        res = pd.DataFrame( pres, columns=["Res"],  index = data.index )
        res = pd.DataFrame( pres,  index = data.index )
        return res


class aiclass_nn_regressor(aiclass):
    def __init__(self, k=5):
        self.ai = KNeighborsRegressor(k)

class aiclass_nn_classifier(aiclass):
    def __init__(self, k=5):
        self.ai = KNeighborsClassifier(k)

class aiclass_nn_radius_regressor(aiclass):
    def __init__(self, r=0.2):
        self.ai = RadiusNeighborsRegressor(radius=r)

class aiclass_nn_radius_classifier(aiclass):
    def __init__(self, r=0.2):
        self.ai = RadiusNeighborsClassifier(radius=r)
class aiclass_gaussiannb_classifier(aiclass):
    def __init__(self):
        self.ai = GaussianNB()
class aiclass_supportvector_classifier(aiclass):
    def __init__(self):
        self.ai = SVC(C=0.01, kernel='linear', degree=5, tol=0.001, probability=True)

class aiclass_xgboost_regressor(aiclass):
    def __init__(self):
        self.ai = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, scale_pos_weight= 0.7)
    def fit(self, indata, inres):
        print("XGBoost Fit")
        eval_set = [(indata, inres), (indata, inres)]
#        self.ai.fit(indata, inres,  early_stopping_rounds=30, eval_set=eval_set)
#        self.ai.fit(indata, inres, eval_metric="map", early_stopping_rounds=10, eval_set=eval_set, verbose=False)
        self.ai.fit(indata, inres, eval_metric=self.errorfunc, early_stopping_rounds=10, eval_set=eval_set, verbose=True)
    def plot(self):
        xgboost.plot_importance(self.ai)
        plt.show()


class aiclass_xgboost_classifier(aiclass):
    def __init__(self):
        self.ai = xgboost.XGBClassifier(learning_rate=0.5, n_estimators=10000, max_depth=5)
    def fit(self, indata, inres):
        def errorfunc(pred, features):
#            print("ef")
            res = 0
#            print(pred.shape)
#            print(features.get_label().shape)
            pred2 = np.where( pred > 0.5,1, 0)
            for n in range(0, pred.shape[0]):
                print((n, int(features.get_label()[n]), pred2[n]))
                if ( int(features.get_label()[n]), int(pred2[n]) ) == ( 0, 0 ) :
                    print("Good #1")
                    res = res - 10
                if ( int(features.get_label()[n]), int(pred2[n]) ) == ( 0, 1 ) :
                    print("Fail #2")
                    res = res + 10
            return (1, res )

        print("XGBoost Fit")
        eval_set = [(indata, inres), (indata, inres)]
#        self.ai.fit(indata, inres, eval_metric="auc", early_stopping_rounds=3, eval_set=eval_set)
        self.ai.fit(indata, inres, eval_metric=errorfunc, early_stopping_rounds=3, eval_set=eval_set, verbose=False)


class aiclass_xgboost_regclassifier(aiclass):
    seplevel = 0.5
    predlevel = 0.5

    def __init__(self, level = 0.5, predlevel = 0.5):
        self.ai = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=1000, max_depth=3, scale_pos_weight= 0.7 )
        self.seplevel = level
        self.predlevel = predlevel

    def fit(self, indata, inres):
        def errorfunc(pred, features):
            res = 0
            pred2 = np.where( pred > self.seplevel,1, 0)
            for n in range(0, pred.shape[0]):
                if ( int(features.get_label()[n]), int(pred2[n]) ) == ( 1, 0 ) :
                    res = res + 5
                if ( int(features.get_label()[n]), int(pred2[n]) ) == ( 0, 1 ) :
                    res = res + 10
            return ("ef_binary", res )

        print("XGBoost Fit")
        eval_set = [(indata, inres), (indata, inres)]
        self.ai.fit(indata, inres, eval_metric="auc", early_stopping_rounds=30, eval_set=eval_set, verbose = False)
#       self.ai.fit(indata, inres, eval_metric=errorfunc, early_stopping_rounds=10, eval_set=eval_set, verbose=False)

    def predict(self, data):
        pres = self.ai.predict(data)
#        res = pd.DataFrame( pres, columns=["Res"],  index = data.index )
        res = pd.DataFrame( np.where(pres > self.predlevel, 1, 0),  index = data.index )
        return res

    def plot(self):
        xgboost.plot_importance(self.ai)
        plt.show()