# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 22:51:03 2022

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



class PerceptronClassifier:
    
    """
    La CLass qui implemente l'arbre de decision qui se base sur le model de Sklearn'

    Parameters:
    - x_train (array) -- Tableau des données du  Train
    - y_train (array) -- les labels des données d'entrainement'
    - x_val (array) --  les valeurs de validations .
    - y_val (array) -- label des valeurs pour validate the model.
    - class_names (array) -- Array of names to link with labels
    
    
    self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None
    """
    def __init__(self ):
        self.x_train=None
        self.y_train=None
        self.classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        self.n_neighbors =None
        self.metric=None
        

    def train_sansGrid(self, x_train,y_train):
        
        """  dans cette classe on entraine notre modèle 
        """
        self.classifier_knn.fit(self.x_train, self.y_train)
        return "dd"
    
    def train(self, x_train,y_train):
        
        param_grid={'n_neighbors':np.arange(1,20),
                    'metric':['euclidean','manhattan']}

        grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=10)
        grid.fit(x_train,y_train)
        self.n_neighbors=grid.best_params_["n_neighbors"]
        self.metric=grid.best_params_["metric"]
        print(self.metric)
        print("xxxxxxxxxxxxxxxxx")
        
        self.classifier_knn=KNeighborsClassifier(self.n_neighbors ,self.metric)
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
    
    def aicha():
        return "Aicha............."
        
        
    