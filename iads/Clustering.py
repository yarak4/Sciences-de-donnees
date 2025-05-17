# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def centroide(data):
    """ Calcule le centroïde des exemples dans un DataFrame ou np.array. """
    return np.mean(data, axis=0)

def normalisation(df):
    """Normalise les colonnes d'un DataFrame afin que leurs valeurs soient dans l'intervalle [0, 1]."""
    # Calcul des min et max pour chaque colonne
    min_values = df.min()
    max_values = df.max()
    
    # Normalisation de chaque colonne
    normalized_df = (df - min_values) / (max_values - min_values)
    
    return normalized_df

def initialise_CHA(df):
    partition = {i: [i] for i in range(len(df))}
    return partition

def dist_centroides(group1, group2):
    """
    Calcule la distance euclidienne entre les centroïdes de deux groupes d'exemples.

    group1, group2 : (DataFrame ou np.array) - Deux groupes d'exemples.
    """
    # Calculer le centroïde des deux groupes
    centroid1 = np.mean(group1, axis=0)
    centroid2 = np.mean(group2, axis=0)

    # Calculer la distance euclidienne entre les deux centroïdes
    dist = np.linalg.norm(centroid1 - centroid2)
    return dist

def fusionne(DF, P0, verbose=False):
    
    min_distance = float('inf')
    key1, key2 = None, None

    # Recherche de la paire de clusters les plus proches
    for cluster1 in P0:
        for cluster2 in P0:
            if cluster1 != cluster2:
                dist = dist_centroides(DF.iloc[P0[cluster1]], DF.iloc[P0[cluster2]])
                if dist < min_distance:
                    min_distance = dist
                    key1, key2 = cluster1, cluster2

    # Fusion des deux clusters les plus proches
    if verbose:
        print(f"fusionne: distance minimale trouvée entre {P0[key1]} et {P0[key2]} = {min_distance:.4f}")
        print(f"fusionne: les 2 clusters dont les clés sont [{key1}, {key2}] sont fusionnés")

    # Mise à jour de la partition après fusion
    P1 = P0.copy()
    new_key = max(P0.keys()) + 1  # Nouvelle clé unique pour le cluster fusionné
    P1[new_key] = P0[key1] + P0[key2]  # Fusion des indices des deux clusters
    del P1[key1], P1[key2]  # Suppression des anciens clusters

    if verbose:
        print(f"fusionne: on crée la nouvelle clé {new_key} dans le dictionnaire.")
        print(f"fusionne: les clés de [{key1}, {key2}] sont supprimées car leurs clusters ont été fusionnés.")

    return P1, key1, key2, min_distance

import scipy.cluster.hierarchy as sch
def CHA_centroid(data, verbose=False, dendrogramme=False):
    # Initialisation de la partition avec CHA
    partition = initialise_CHA(data)
    fusions = []
    
    # Affichage du démarrage de l'algorithme si verbose=True
    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")

    # Tant qu'il y a plus d'un cluster, on continue les fusions
    while len(partition) > 1:
        # Fusion des 2 clusters les plus proches
        P1, clust1, clust2, dist = fusionne(data, partition, verbose)
        
        # Enregistrement de la fusion dans le tableau des résultats
        fusions.append([clust1, clust2, dist, len(partition[clust1]) + len(partition[clust2])])
        
        # Mise à jour de la partition (retirer les anciens clusters et ajouter le nouveau)
        partition = P1
    
    # Si on veut afficher le dendrogramme
    if dendrogramme:
        # Construction de la matrice de liens (linkage matrix) pour le dendrogramme
        linkage_matrix = []
        for fusion in fusions:
            clust1, clust2, dist, size = fusion
            linkage_matrix.append([clust1, clust2, dist, size])
        
        # Convertir la liste en np.array pour la fonction de scipy
        linkage_matrix = np.array(linkage_matrix)
        
        # Construction du dendrogramme
        plt.title('Dendrogramme')
        plt.xlabel("Indice d'exemple")
        plt.ylabel('Distance')
        sch.dendrogram(linkage_matrix, leaf_font_size=24.)
        plt.show()
    
    # Retourner la liste des fusions effectuées
    return fusions
