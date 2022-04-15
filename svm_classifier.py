# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 22:02:13 2022

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


from sklearn.svm import SVC

class SVMClassifier:

  
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
      self.classifier_svm = SVC()
      self.c =None
      self.gamma=None
      self.kernel=None
      

      

  def train_sansGrid(self, x_train,y_train):
      
      """  dans cette classe on entraine notre modèle 
      """
      self.classifier_svm.fit(self.x_train, self.y_train)
      
  
  def train(self, x_train,y_train):
      

      
      param_grid = {'C': [0.1,1, 10, 100], 
                    'gamma': [1,0.1,0.01,0.001],
                    'kernel': ['rbf', 'poly']}

      grid=GridSearchCV(SVC(),param_grid,cv=10)
      grid.fit(x_train,y_train)
      self.c=grid.best_params_["C"]
      self.gamma=grid.best_params_["gamma"]
      self.kernel=grid.best_params_["kernel"]
    
    
      self.classifier_svm=SVC(C=self.c,kernel=self.kernel,gamma=self.gamma())
      self.classifier_svm.fit(x_train,y_train)
      

  
  def prediction(self,x):
    return self.classifier_svm.predict(x) 
      
    

  def erreur(self, t, prediction):
      """
      Retourne la différence au carré entre
      la cible ``t`` et la prédiction ``prediction``.
      """
      return (t - prediction) ** 2
  

      
        
        
    