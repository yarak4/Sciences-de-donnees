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
from sklearn.feature_extraction.text import CountVectorizer
import graphviz as gv

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

def entropie_conditionnelle(Y, Xj):
    """Calcule l'entropie conditionnelle H(Y|Xj)."""
    valeurs_Xj, counts = np.unique(Xj, return_counts=True)
    H_Y_given_Xj = 0.0
    
    for v, count in zip(valeurs_Xj, counts):
        
        Y_subset = Y[Xj == v]  # Filtrer Y selon la valeur v de Xj
        H_Y_given_Xj += (count / len(Y)) * entropie(Y_subset)
    
    return H_Y_given_Xj



def construit_AD(X, Y, epsilon, LNoms=[]):
    """Construit un arbre de décision en utilisant seulement les fonctions définies avant."""

    entropie_ens = entropie(Y)

    # Condition d'arrêt : entropie faible ou données trop petites
    if entropie_ens <= epsilon or len(Y) <= 5 or len(np.unique(Y)) == 1:
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        return noeud

    # Initialisation des variables pour trouver l'attribut optimal
    min_entropie = float('inf')
    i_best = -1
    Xbest_valeurs = None

    for i in range(X.shape[1]):  # Parcourir tous les attributs
        H_Y_given_Xi = entropie_conditionnelle(Y, X[:, i])
        
        if H_Y_given_Xi < min_entropie:
            min_entropie = H_Y_given_Xi
            i_best = i
            Xbest_valeurs = np.unique(X[:, i])

    if i_best == -1:  # Aucun gain d'information -> on crée une feuille
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
        return noeud

    # Création du nœud pour l'attribut sélectionné
    noeud = NoeudCategoriel(i_best, LNoms[i_best] if LNoms else f"att_{i_best}")

    for v in Xbest_valeurs:
        indices = (X[:, i_best] == v)  # Filtrer X et Y selon la valeur v
        X_subset, Y_subset = X[indices], Y[indices]

        if len(Y_subset) == 0:  # Cas où aucun exemple ne correspond
            feuille = NoeudCategoriel(-1, "Label")
            feuille.ajoute_feuille(classe_majoritaire(Y))
            noeud.ajoute_fils(v, feuille)
        else:
            noeud.ajoute_fils(v, construit_AD(X_subset, Y_subset, epsilon, LNoms))

    return noeud




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
        return 1

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
    

class BernoulliNaiveBayes(Classifier):
    """ Classifieur Bernoulli Naive Bayes multiclasse """

    def __init__(self, input_dimension, alpha=1.0):
        super().__init__(input_dimension)
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = {}       # log(P(c))
        self.feature_probs = {}         # P(w|c)

    def train(self, desc_set, label_set):
        self.classes = np.unique(label_set)
        n_features = self.dimension

        for c in self.classes:
            X_c = desc_set[label_set == c]
            self.class_log_prior[c] = np.log(len(X_c) / len(desc_set))

            # Laplace smoothing
            count_wc = np.sum(X_c > 0, axis=0)
            total_wc = X_c.shape[0]
            self.feature_probs[c] = (count_wc + self.alpha) / (total_wc + 2 * self.alpha)

    def score(self, x):
        """ Renvoie un dictionnaire {classe: score log-proba} """
        eps = 1e-12
        scores = {}
        for c in self.classes:
            P = np.clip(self.feature_probs[c], eps, 1 - eps)
            log_likelihood = np.sum(x * np.log(P) + (1 - x) * np.log(1 - P))
            scores[c] = self.class_log_prior[c] + log_likelihood
        return scores

    def predict(self, x):
        """ Prédit la classe avec la plus grande proba log-score """
        scores = self.score(x)
        return max(scores, key=scores.get)

    
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)  # Appel du constructeur de la classe mère
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)


    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def draw(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple 
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return None
    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        if self.est_feuille():
            return 1
        total = 0
        for noeud in self.Les_fils:
            total += self.Les_fils[noeud].compte_feuilles()
        return total
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g