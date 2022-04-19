# -*- coding: utf-8 -*-
"""
Ã‰diteur de Spyder

Maman Souley Aicha mama3101
Mahamadou Sangare sanm0301

"""


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(object):
    
    
    
    Class qui implemente la regression logistic de sklearn
    def __init__(self, x_train, y_train, x_val, y_val, scorers):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.estimator = LogisticRegression(n_jobs=4)
        self.scorers = scorers

    def train_sans_grid(self):
        LogisticRegression = self.estimator
        LogisticRegression.fit(self.x_train, self.y_train)
        
        pred_val = LogisticRegression.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)
        print('Accuracy des donne de validation: {:.3%}'.format(accu_val))

        

    def train(self, grid_search_params={}, random_search=True):
        """
        enTraine le model en utilisant le grid_search
        Inputs:
        - grid_search_params (dict) -- 

        
        """
        # Grid search init with kfold
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

        # entrainement de Model
        search_g.fit(self.x_train, self.y_train)

        # garder les meilleurs param et les affiche en sortie
        self.estimator = search_g.best_estimator_
        self.best_accuracy = search_g.best_score_
        self.hyper_search = search_g
        # Predictions de validation data
        
        pred_val = search_g.predict(self.x_val)
        
        
        acc_val = accuracy_score(self.y_val, pred_val)

        print('Best cross val accuracy : {}'.format(self.hyper_search.best_score_))
        print('Best estimator:\n{}'.format(self.hyper_search.best_estimator_))
        print()
            
        print('Accuracy validation: {:.3%}'.format(acc_val))

    def predict(self, X):
        """
        utilise le model entraine pour predire la sortie des donnees de test
        """
        return self.estimator.predict(X)
       
