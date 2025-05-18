# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import pandas as pd
from collections import Counter
import numpy as np
import copy
import time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def evaluer_classifieur(clf, X_train, y_train, X_test, y_test, afficher_cm=True):
    """
    Entraîne, évalue, affiche les temps, l'accuracy, et la matrice de confusion.
    
    clf : classifieur (instance)
    X_train, y_train : données d'entraînement
    X_test, y_test : données de test
    afficher_cm : bool, affiche ou non la matrice de confusion

    Retourne : accuracy, temps_train, temps_test
    """
    # Entraînement
    print("Entraînement...")
    start_train = time.time()
    clf.train(X_train, y_train)
    elapsed_train = time.time() - start_train
    print(f"Temps d'entraînement : {elapsed_train:.4f} secondes")

    # Prédiction + évaluation
    print("Évaluation...")
    start_test = time.time()
    y_pred = np.array([clf.predict(x) for x in X_test])
    elapsed_test = time.time() - start_test

    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Temps d'évaluation : {elapsed_test:.4f} secondes")

    # Matrice de confusion
    if afficher_cm:
        classes_uniques = np.unique(np.concatenate((y_test, y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=classes_uniques)
        afficher_matrice_confusion(cm, classes_uniques)

    return accuracy, elapsed_train, elapsed_test

def accuracy_avec_confusion(clf, X_test, y_test):
    """
    clf : classifieur entraîné (doit avoir une méthode predict(x))
    X_test : ndarray des descriptions de test
    y_test : ndarray des labels réels

    Affiche l'accuracy et la matrice de confusion, retourne l'accuracy
    """
    # Prédictions
    y_pred = np.array([clf.predict(x) for x in X_test])

    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy : {accuracy:.4f}")

    # Matrice de confusion
    classes_uniques = np.unique(np.concatenate((y_test, y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=classes_uniques)

    # Affichage
    afficher_matrice_confusion(cm, classes_uniques)

    return accuracy


def afficher_resultats_validation_croisee(perfs, moyenne, ecart_type, matrice_confusion, classes):
    print("Performances par fold :")
    for i, perf in enumerate(perfs, 1):
        print(f"  Fold {i}: {perf:.4f}")
    print(f"\nPerformance moyenne : {moyenne:.4f}")
    print(f"Écart-type : {ecart_type:.4f}\n")

def crossval(C, X, y, nb_iter):
    performances, moyenne, ecart_type, matrice_confusion = validation_croisee(C, (X,y), nb_iter)
    # Affichage des résultats
    print("\nValidation croisée :")
    for i, perf in enumerate(performances):
        print(f"  Fold {i+1}: {perf:.4f}")
    print(f"\nMoyenne des performances : {moyenne:.4f}")
    print(f"Écart-type : {ecart_type:.4f}")
    
    # Affichage matrice de confusion cumulée
    classes = np.unique(y)
    print("\nMatrice de confusion cumulée sur tous les folds :")
    afficher_matrice_confusion(matrice_confusion, classes)
    
    return performances, moyenne, ecart_type

    

def afficher_matrice_confusion(matrice_confusion, classes):
    """
    matrice_confusion: np.array carrée (nb_classes x nb_classes)
    classes: liste ou array des noms de classes dans l'ordre
    
    Affiche la matrice de confusion avec heatmap et annotations.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité terrain')
    plt.title('Matrice de Confusion')
    plt.show()

def validation_croisee(C, DS, nb_iter):
    """
    C: instance de classifieur (ex: ClassifierKNNMulti déjà instancié)
    
     X est un ndarray de descriptions (ex: vecteurs)
    y est un ndarray des labels correspondants
    nb_iter: int, nombre de folds

    Rend: (liste_perf, moyenne_perf, ecart_type_perf, matrice_confusion_cumulee)
    """
    X, y = DS
    n = len(y)
    indices = np.random.permutation(n)
    X = X[indices]
    y = y[indices]

    tailles_fold = n // nb_iter
    performances = []
    
    # Trouver nombre de classes unique
    classes_uniques = np.unique(y)
    nb_classes = len(classes_uniques)
    # Pour indexer matrice confusion
    class_to_idx = {c: i for i, c in enumerate(classes_uniques)}
    
    matrice_confusion = np.zeros((nb_classes, nb_classes), dtype=int)
    
    for i in range(nb_iter):
        debut = i * tailles_fold
        fin = (i + 1) * tailles_fold if i < nb_iter - 1 else n
        
        X_test = X[debut:fin]
        y_test = y[debut:fin]
        
        X_train = np.concatenate((X[:debut], X[fin:]), axis=0)
        y_train = np.concatenate((y[:debut], y[fin:]), axis=0)
        
        clf = copy.deepcopy(C)  # copie propre du classifieur (réinitialisé)
        clf.train(X_train, y_train)
        
        perf = clf.accuracy(X_test, y_test)
        performances.append(perf)
        
        # Prédictions pour matrice confusion
        for true_label, x_desc in zip(y_test, X_test):
            pred_label = clf.predict(x_desc)
            i_true = class_to_idx[true_label]
            i_pred = class_to_idx[pred_label]
            matrice_confusion[i_true, i_pred] += 1
    
    moyenne = np.mean(performances)
    ecart_type = np.std(performances)
    
    return performances, moyenne, ecart_type, matrice_confusion

def train_test_evaluation(X_train, y_train, X_test, y_test, classifier_class, classifier_params):
    clf = classifier_class(**classifier_params)

    start_train = time.time()
    clf.train(X_train, y_train)
    elapsed_train = time.time() - start_train

    start_test = time.time()
    correct = 0
    for i in range(len(X_test)):
        pred = clf.predict(X_test[i])
        if pred == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test)
    elapsed_test = time.time() - start_test

    print(f"Temps entraînement : {elapsed_train:.4f}s")
    print(f"Accuracy sur test : {accuracy:.4f}")
    print(f"Temps évaluation : {elapsed_test:.4f}s")

    return accuracy


