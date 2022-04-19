

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


class svmClassifier(object):
    """
    classe qui implemnte le classifier svm
    """
    def __init__(self, x_train, y_train, x_val, y_val,scorers):
        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val
        self.scorers = scorers
  

      

        self.estimator = SVC(probability=True)
        

    def train_sans_grid(self):
        """
        entraine le model sans la recherche d hyperparametre
        """
        svm = self.estimator
        svm.fit(self.x_train, self.y_train)
        
        pred_val = svm.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)

        
        print('Accuracy des donne de validation: {:.3%}'.format(accu_val))

    def train(self, grid_search_params={}, random_search=True):
        """
        entraine le modele avec un perte de cross entropie

        Inputs:
        - grid_search_params (dict) 

       
        """
        # Grid search
        searching_params = searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv":KFold(n_splits=5, shuffle=True),
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
        
        pred_val = search_g.predict(self.x_val)

       
        acc_val = accuracy_score(self.y_val, pred_val)

        
        print('Best cross val accuracy : {}'.format(self.hyper_search.best_score_))
        print('Best estimator:\n{}'.format(self.hyper_search.best_estimator_))
        print()
            
        print('Accuracy validation: {:.3%}'.format(acc_val))
        return acc_val, self.estimator, self.best_accuracy

    def predict(self, X):
        
        return self.estimator.predict(X)
    def erreur(self, t, prediction):
      """
      Retourne la diffÃ©rence au carrÃ© entre
      la cible ``t`` et la prÃ©diction ``prediction``.
      """
      return (t - prediction) ** 2
        
