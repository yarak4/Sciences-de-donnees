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
    DS: tuple (X, y) où
        - X est un ndarray de descriptions (ex: vecteurs)
        - y est un ndarray des labels correspondants
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


def prepare_vectorizer(texts_train, texts_full=None, binary=True, max_features=50_000):
    """
    Crée et fit un CountVectorizer binaire (ou pas) sur les textes d'entraînement
    Optionnellement fit sur le dataset full pour un vocab plus large.
    Retourne le vectorizer entraîné.
    """
    vect = CountVectorizer(binary=binary, max_features=max_features)
    if texts_full is not None:
        vect.fit(texts_full)
    else:
        vect.fit(texts_train)
    return vect

def vectorize_texts(vect, texts):
    """ Transforme une liste de textes en matrice numpy vectorisée via un vectorizer déjà fit """
    return vect.transform(texts).toarray()

def get_labels(dataframe, target_col='target'):
    """ Récupère les labels au format numpy array """
    return dataframe[target_col].to_numpy()

def evaluate_fold(clf, X_train, y_train, X_test, y_test, labels_display):
    """
    Entraîne clf sur (X_train, y_train), prédit sur X_test, calcule accuracy, matrice confusion, temps.

    Retourne (accuracy, matrice_confusion, elapsed_time)
    """
    start = time.time()
    clf.train(X_train, y_train)
    elapsed = time.time() - start

    y_pred = np.array([clf.predict(x) for x in X_test])
    acc = clf.accuracy(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels_display)

    return acc, cm, elapsed


def plot_confusion_matrix(conf_matrix, labels_display, title="Matrice de confusion"):
    """
    Affiche une matrice de confusion avec seaborn heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_display, yticklabels=labels_display)
    plt.title(title)
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()


def crossval_evaluation_with_time(clf_class, X, y, n_folds=5, **clf_kwargs):
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    total_conf_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)
    total_train_time = 0
    total_test_time = 0

    print(f"▶ Validation croisée {n_folds}-fold")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clf_class(input_dimension=X.shape[1], **clf_kwargs)

        start_train = time.time()
        clf.train(X_train, y_train)
        elapsed_train = time.time() - start_train

        start_test = time.time()
        y_pred = np.array([clf.predict(x) for x in X_test])
        elapsed_test = time.time() - start_test

        acc = np.mean(y_pred == y_test)
        scores.append(acc)
        total_train_time += elapsed_train
        total_test_time += elapsed_test

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        total_conf_matrix += cm

        print(f"Fold {fold+1} — Accuracy : {acc:.4f} — Temps entraînement : {elapsed_train:.2f}s — Temps test : {elapsed_test:.2f}s")

    print("\nRésumé général")
    print(f"→ Moyenne accuracy : {np.mean(scores):.4f}")
    print(f"→ Écart-type : {np.std(scores):.4f}")
    print(f"→ Temps total entraînement : {total_train_time:.2f} s (≈ {total_train_time/n_folds:.2f} s/fold)")
    print(f"→ Temps total évaluation : {total_test_time:.2f} s (≈ {total_test_time/n_folds:.2f} s/fold)")

    plt.figure(figsize=(8, 6))
    sns.heatmap(total_conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Matrice de confusion cumulée sur {n_folds} folds')
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()

    return np.mean(scores), np.std(scores), total_train_time, total_test_time

def analyser_classifieur_matrice(clf, X_train, y_train, X_test, y_test,
                                X_full=None, y_full=None,
                                classe_positive=None, n_folds=5):
    """
    Évalue un classifieur avec :
      - train/test fixes sur X_train/y_train et X_test/y_test
      - validation croisée (KFold) sur X_full/y_full si fournis

    Supporte classification binaire (classe_positive) ou multiclasse.

    Affiche accuracy, temps, matrice de confusion pour les deux évaluations.
    """

    def binarize_labels(y, pos_class):
        return np.array([1 if label == pos_class else -1 for label in y])

    # Prépare labels selon mode binaire ou multiclasse
    if classe_positive is not None:
        y_train_proc = binarize_labels(y_train, classe_positive)
        y_test_proc = binarize_labels(y_test, classe_positive)
        mode = f"Binaire (classe {classe_positive})"
        labels_display = [-1, 1]
        if X_full is not None and y_full is not None:
            y_full_proc = binarize_labels(y_full, classe_positive)
        else:
            y_full_proc = None
    else:
        y_train_proc = np.array(y_train)
        y_test_proc = np.array(y_test)
        mode = "Multiclasse"
        labels_display = sorted(np.unique(y_train_proc))
        y_full_proc = np.array(y_full) if y_full is not None else None

    print(f"=== Évaluation ({mode}) sur jeu train/test fixes ===")
    start = time.time()
    clf.train(X_train, y_train_proc)
    y_pred = np.array([clf.predict(x) for x in X_test])
    elapsed = time.time() - start
    acc = clf.accuracy(X_test, y_test_proc)
    print(f"Accuracy sur test : {acc:.4f} - Temps entraînement+prédiction : {elapsed:.2f}s")
    cm = confusion_matrix(y_test_proc, y_pred, labels=labels_display)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_display, yticklabels=labels_display)
    plt.title(f"Matrice de confusion - test fixe ({mode})")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()

    # Validation croisée
    if X_full is not None and y_full_proc is not None:
        print(f"\n=== Validation croisée KFold ({mode}) sur dataset complet ===")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        total_conf_matrix = np.zeros((len(labels_display), len(labels_display)), dtype=int)
        total_time = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
            X_tr, X_te = X_full[train_idx], X_full[test_idx]
            y_tr, y_te = y_full_proc[train_idx], y_full_proc[test_idx]
            start = time.time()
            clf.train(X_tr, y_tr)
            y_pred_fold = np.array([clf.predict(x) for x in X_te])
            elapsed = time.time() - start
            total_time += elapsed
            acc_fold = clf.accuracy(X_te, y_te)
            scores.append(acc_fold)
            cm_fold = confusion_matrix(y_te, y_pred_fold, labels=labels_display)
            total_conf_matrix += cm_fold
            print(f"Fold {fold+1}: Accuracy={acc_fold:.4f}, Temps={elapsed:.2f}s")

        print(f"\nMoyenne accuracy CV : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"Temps total CV : {total_time:.2f}s ({total_time/n_folds:.2f}s par fold)")

        plt.figure(figsize=(7,6))
        sns.heatmap(total_conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_display, yticklabels=labels_display)
        plt.title(f"Matrice de confusion - Validation croisée ({mode})")
        plt.xlabel("Prédiction")
        plt.ylabel("Réel")
        plt.show()

        return acc, np.mean(scores)
    else:
        return acc, None
