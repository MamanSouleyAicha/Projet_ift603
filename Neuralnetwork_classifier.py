# -*- coding: utf-8 -*-
"""
Maman Souley Aicha mama3101
Mahamadou Sangare sanm03301
"""








import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

class NNClassifier(object):
    
   
    def __init__(self, x_train, y_train, x_val, y_val, scorers):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.estimator = MLPClassifier(max_iter=800)
        self.scorers = scorers
        self.hyper_search = None
        self.best_accuracy = 0
        
        

    def train_sansGrid(self):
        #entraine le model sans la recherche d hyperparam
        
        KNN = self.estimator
        KNN.fit(self.x_train, self.y_train)
        pred_val = self.estimator.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)

        
        print('Accuracy validation: {:.3%}'.format(accu_val))

        
    
    def train_hyperparameter(self, estimator_params, random_search=True, verbose=False):
        #entrainement des models avec la recherche d hyperparametre en utilisant grid search
        searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": 5,
            "return_train_score": True,
            "verbose": int(verbose),
            "n_jobs": 4}

        if random_search:
            if verbose:
                print("Using randomized search:")
            self.hyper_search = RandomizedSearchCV(self.estimator, estimator_params).set_params(**searching_params)
        else:
            if verbose:
                print("Using complet search:")
            self.hyper_search = GridSearchCV(self.estimator, estimator_params).set_params(**searching_params)

        # Chercher les meilleurs param
        self.hyper_search.fit(self.x_train, self.y_train)


        self.estimator = self.hyper_search.best_estimator_
        
        pred_val = self.hyper_search.predict(self.x_val)

        accu_val = accuracy_score(self.y_val, pred_val)

        if verbose:
            print()
            print('Best cross val accuracy : {}'.format(self.hyper_search.best_score_))
            print('Best estimator:\n{}'.format(self.hyper_search.best_estimator_))
            print()
            print('Accuracy validation: {:.3%}'.format(accu_val))

        return accu_val, self.estimator
    
    def prediction(self,x):
        # faire la prediction sur le model entraine
        return self.estimator.predict(x)
        
    
        
    def erreur(self, t, prediction):
        """
        Retourne la diff????rence au carr???? entre
        la cible ``t`` et la pr????diction ``prediction``.
        """
        return (t - prediction) ** 2
    

        
        
    
