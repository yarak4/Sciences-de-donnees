# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Import de packages externes
import pandas as pd
from collections import Counter
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)
    max_index = np.argmax(nb_fois)  # Index du maximum dans nb_fois
    return valeurs[max_index]

def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    P = np.array(P)  # Conversion en tableau numpy
    P = P[P > 0]  # Filtrer les probabilités nulles pour éviter log(0)
    k = len(P)  # Nombre de classes non nulles

    if k <= 1:  # Si une seule classe ou aucune, H(P) = 0
        return 0.0

    return -np.sum(P * np.log(P)) / np.log(k)

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)  # Trouver les classes et leurs effectifs
    P = nb_fois / len(Y)  # Calcul des probabilités
    return shannon(P)  # Calcul de l'entropie de Shannon

def analyser_classifieur_matrice(clf, X, y, classe_positive=None, n_folds=5):
    """
    Évalue un classifieur sur données vectorisées (X, y) avec affichage de résultats et matrice de confusion.

    - clf : classifieur (hérite de Classifier)
    - X : matrice numpy (features déjà vectorisées)
    - y : labels (array ou liste)
    - classe_positive : pour classification binaire, sinon None pour multiclasse
    - n_folds : nombre de folds
    """
    # Binarisation si demandé
    if classe_positive is not None:
        y = np.array([1 if label == classe_positive else -1 for label in y])
        mode = f"Binaire (classe {classe_positive})"
        labels_display = [-1, 1]
    else:
        y = np.array(y)
        mode = "Multiclasse"
        labels_display = sorted(np.unique(y))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    total_conf_matrix = np.zeros((len(labels_display), len(labels_display)), dtype=int)
    total_time = 0

    print(f"▶ Évaluation {mode} en {n_folds} folds...\n")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()
        clf.train(X_train, y_train)
        elapsed = time.time() - start
        total_time += elapsed

        y_pred = np.array([clf.predict(x) for x in X_test])
        acc = clf.accuracy(X_test, y_test)
        scores.append(acc)

        cm = confusion_matrix(y_test, y_pred, labels=labels_display)
        total_conf_matrix += cm

        print(f"Fold {fold+1} — Accuracy : {acc:.4f} — Temps : {elapsed:.2f} s")

    print("\n Résumé général")
    print(f"→ Type : {mode}")
    print(f"→ Moyenne accuracy : {np.mean(scores):.4f}")
    print(f"→ Écart-type : {np.std(scores):.4f}")
    print(f"→ Temps total : {total_time:.2f} sec (≈ {total_time/n_folds:.2f} sec/fold)")

    # 🎨 Affichage graphique de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_display, yticklabels=labels_display)
    plt.title(f'Matrice de confusion — {mode}')
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()





class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        correct_predictions = 0
        for i in range(len(desc_set)):
            if self.predict(desc_set[i]) == label_set[i]:
                correct_predictions += 1
        return correct_predictions / len(desc_set)
    

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.k = k
        self.desc_set = None
        self.label_set = None

        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.linalg.norm(x - self.desc_set, axis = 1)
        indices_proches = np.argsort(distances)
        indices_k_nearest = indices_proches[:self.k]
        labels_k_nearest = self.label_set[indices_k_nearest]
        proportion = len(labels_k_nearest[labels_k_nearest == 1])/self.k
        return 2*(proportion-0.5)
        
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x)>=0:
            return 1
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set

class ClassifierKNNMulti(Classifier):
    """ KNN multiclasse avec distance euclidienne. Hérite de Classifier. """

    def __init__(self, input_dimension, k):
        super().__init__(input_dimension)
        self.k = k
        self.desc_set = None
        self.label_set = None

    def train(self, desc_set, label_set):
        self.desc_set = desc_set
        self.label_set = label_set

    def score(self, x):
        """ Retourne -distance pour compatibilité (pas utilisé ici) """
        distances = np.linalg.norm(x - self.desc_set, axis=1)
        return -np.min(distances)

    def predict(self, x):
        distances = np.linalg.norm(x - self.desc_set, axis=1)
        indices_k = np.argsort(distances)[:self.k]
        labels_k = self.label_set[indices_k]
        return Counter(labels_k).most_common(1)[0][0]
    
class ClassifierKNNCosine(Classifier):
    """ KNN multiclasse avec distance cosinus. Hérite de Classifier. """

    def __init__(self, input_dimension, k):
        super().__init__(input_dimension)
        self.k = k
        self.desc_set = None
        self.label_set = None

    def train(self, desc_set, label_set):
        self.desc_set = desc_set
        self.label_set = label_set

    def cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def score(self, x):
        distances = [self.cosine_distance(x, d) for d in self.desc_set]
        return -min(distances)

    def predict(self, x):
        distances = [self.cosine_distance(x, d) for d in self.desc_set]
        indices_k = np.argsort(distances)[:self.k]
        labels_k = self.label_set[indices_k]
        return Counter(labels_k).most_common(1)[0][0]


class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.w = np.random.randn(input_dimension)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if score>0:
            return 1
        return -1
    
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self,input_dimension)
        self.eps = learning_rate
        if not init:
            self.w = np.random.randn(input_dimension) * 0.01
        else:
            self.w = np.zeros(input_dimension)
        self.allw =[self.w.copy()]
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        indices = np.random.permutation(len(desc_set))
        for i in indices:
            if label_set[i] * np.dot(self.w, desc_set[i]) <= 0:
                self.w += self.eps * label_set[i] * desc_set[i]
                self.allw.append(self.w.copy())
     
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        norm_diff_list = []
        for _ in range(nb_max):
            old_w = self.w.copy()
            self.train_step(desc_set, label_set)
            norm_diff = np.linalg.norm(self.w - old_w)
            norm_diff_list.append(norm_diff)
            if norm_diff < seuil:
                break
        return norm_diff_list
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
    
    def get_allw(self):
        return self.allw

