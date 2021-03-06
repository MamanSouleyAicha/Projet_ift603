# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:28:59 2022

@author: toshiba
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

class KNNClassifier(object):

    def __init__(self, x_train, y_train, x_val, y_val, scorers):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
        self.estimator = KNeighborsClassifier(n_neighbors=3, n_jobs=4)
        self.scorers = scorers
        self.best_accuracy = 0

    def train_sans_grid(self):
        # entraine le model sans le grid search
        KNN = self.estimator
        KNN.fit(self.x_train, self.y_train)
        pred_val = KNN.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)
        print('Accuracy des donne de validation: {:.3%}'.format(accu_val))
        
    def train(self, grid_search_params={}, random_search=True):
        # entraine le model en utilisant le grid search

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

        # Predictions
        
        pred_val = self.estimator.predict(self.x_val)

        #  accuracy VALIDATION
        
        accu_val = accuracy_score(self.y_val, pred_val)
        
        print('Best cross val accuracy : {}'.format(self.hyper_search.best_score_))
        print('Best estimator:\n{}'.format(self.hyper_search.best_estimator_))
        print()
        
        print('Accuracy validation: {:.3%}'.format(accu_val))

        return accu_val, self.estimator, self.best_accuracy

    def predict(self,x):
        
        return self.estimator.predict(x)
        
    
    def erreur(self, t, prediction):
        """
        Retourne la diff????rence au carr???? entre
        la cible ``t`` et la pr????diction ``prediction``.
        """
        return (t - prediction) ** 2

