# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:56:46 2022

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
from sklearn.tree import DecisionTreeClassifier


class DTClassifier:
    
    
    def __init__(self ):
        self.x_train=None
        self.y_train=None
        self.classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        self.n_neighbors =None
        self.metric=None
        
       


        

    def train_sansGrid(self, x_train,y_train):
        
        """  dans cette classe on entraine notre modèle 
        """
        self.classifier_dt.fit(x_train, y_train)
    
    
    def train(self, x_train,y_train):
        
        param_grid={'n_neighbors':np.arange(1,20),
                    'metric':['euclidean','manhattan']}

        grid=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=10)
        grid.fit(x_train,y_train)
        self.n_neighbors=grid.best_params_["n_neighbors"]
        self.metric=grid.best_params_["metric"]
        print(self.metric)
        
        self.classifier_knn=KNeighborsClassifier(self.n_neighbors ,self.metric)
        self.classifier_knn.fit(x_train,y_train)
    
    def prediction(self,x):
        
        return self.classifier_dt.predict(x) 
        
    
        
  
    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (t - prediction) ** 2
    