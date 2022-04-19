


import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


class DTSClassifier(object):
    

    def __init__(self, x_train, y_train, x_val, y_val, scorers):
        # classe qui implemente l arbre de decision
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.estimator = DecisionTreeClassifier()
        self.scorers = scorers
        self.best_accuracy = 0

    def train_sans_grid(self):
         
        #entrainement sans hyper param
        DTS = self.estimator
        DTS.fit(self.x_train, self.y_train)
       

        pred_val = DTS.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)
        
            
        print('Accuracy validation: {:.3%}'.format(accu_val))

        

    def train(self, grid_search_params={}, random_search=True):
        
        #entrainement avec recherche d hyperparamettre
        # Grid search 
        searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": 5,
            "return_train_score": True,
            "n_jobs": 4,
            "verbose": 1}

        if random_search:
            print("Using randomized search:")
            search_g = RandomizedSearchCV(self.estimator, grid_search_params).set_params(**searching_params)
        else:
            print("Using complet search:")
            search_g = GridSearchCV(self.estimator, grid_search_params).set_params(**searching_params)

        # Model 
        search_g.fit(self.x_train, self.y_train)
        
        self.estimator = search_g.best_estimator_
        self.best_accuracy = search_g.best_score_
        self.hyper_search = search_g

        
        faire une prediction sur les donnees de val
        pred_val = self.estimator.predict(self.x_val)

        
        accu_val = accuracy_score(self.y_val, pred_val)

        return accu_val, self.estimator, self.best_accuracy

    def predict(self, X):
        """
        faire la prediction sur le model entraine
        """
        return self.estimator.predict(X)
        
