# -*- coding: utf-8 -*-
"""
Ãƒâ€°diteur de Spyder
Maman Souley Aicha mama3101
Mahamadou Sangare sanm0301

"""
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


class RandForestClassifier(object):
    """
    Class qui implemente la foret aleatoir de sklearn

    """
    
    def __init__(self, x_train, y_train, x_val, y_val,scorers):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.num_features = x_train.shape[1]
        self.estimator = RandomForestClassifier(n_jobs=4)  
        self.scorers = scorers
        self.hyper_search = None
    def train_default(self, verbose=False):
        """
        EnTraine le model avec les parameters par defauts de sklearn .

        affiche l accuracy pour la data de validation .
        et fit self.estimator avec les meilleurs scores
        """
        self.estimator.fit(self.x_train, self.y_train)

        pred_val = self.estimator.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)

        if verbose:

            print('Accuracy des donne de validation: {:.3%}'.format(accu_val))
    def train_hyperparameter(self, estimator_params, random_search=True, verbose=False):
        """
        entraine le modele sur en utilisant le grid searc

        Inputs:
        - grid_search_params (dict) -- 
        - random_search (bool), default=True -- si True utilise le Randimized Search, 
                si False cherche toute les combinaison possible (trop long a excuter).

        
        - fit le self.estimator avec le meilleur score
        """

        # Grid search
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

        # RECHERCHE hyper parameters
        self.hyper_search.fit(self.x_train, self.y_train)

        # garder les meilleurs param
        self.estimator = self.hyper_search.best_estimator_

        # Predictions de validation data
      
        pred_val = self.hyper_search.predict(self.x_val)

        # Train et validation accuracy
        
        accu_val = accuracy_score(self.y_val, pred_val)

        if verbose:
            print()
            print('Best cross val accuracy : {}'.format(self.hyper_search.best_score_))
            print('Best estimator:\n{}'.format(self.hyper_search.best_estimator_))
            print()
            
            print('Accuracy validation: {:.3%}'.format(accu_val))

        

    def predict(self, X):
        """
        Utilise le model entraine pour predire la sortie de nos donnees de test
        """
        return self.estimator.predict(X)

    
        
    