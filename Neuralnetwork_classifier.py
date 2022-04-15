# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:23:54 2022

@author: user
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

class NeuralNetworkClassifier:
    
    """

     
        
        
    """
    def __init__(self ):
        self.x_train=None
        self.y_train=None
        self.classifier_mlp = MLPClassifier()
        self.alpha =None
        self.hidden_layer=None
        self.max_iter=800
        
        

    def train_sansGrid(self, x_train,y_train):
        
        """  dans cette classe on entraine notre modèle 
        """
        self.classifier_mlp.fit(self.x_train, self.y_train)
        
    
    def train(self, x_train,y_train):
        
        get_params= {'solver': ['lbfgs'], 
              'max_iter': [800],
              'alpha': 10.0 ** -np.arange(1, 10), 
              'hidden_layer_sizes':np.arange(10, 15), 
              'random_state':[0]}
          
        grid = GridSearchCV(MLPClassifier(),get_params,cv=10)
        grid.fit(x_train,y_train)
        
        self.alpha =grid.best_params_["alpha"]
        self.hidden_layer=grid.best_params_["hidden_layer_sizes"]
        self.max_iter=800
        
        
        self.classifier_mlp=MLPClassifier(hidden_layer_sizes=self.hidden_layer,alpha=self.alpha,max_iter=800)
        self.classifier_knn.fit(x_train,y_train)
        return "ccc"
    
    def prediction(self,x):
        
        return self.classifier_knn.predict(x) 
        
    
        
    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (t - prediction) ** 2
    

        
        
    
