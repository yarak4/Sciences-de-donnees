import string
import numpy as np
import pandas as pd

def nettoyage(chaine):
    chaine_mod = ""  # On crée une chaîne vide pour stocker le texte nettoyé

    # On parcourt chaque caractère de la chaîne
    for c in chaine:
        # Si le caractère est une ponctuation (mais PAS l'apostrophe), on le remplace par un espace
        if c in string.punctuation and c != "'":
            chaine_mod += " "
        else:
            # Sinon, on ajoute le caractère en minuscule à la chaîne finale
            chaine_mod += c.lower()

    return chaine_mod  # On retourne la chaîne nettoyée

def text2vect(chaine, mots_inutiles):
    chaine_nettoye = nettoyage(chaine)  # On nettoie la chaîne avec la fonction précédente

    mots = chaine_nettoye.split()  # On découpe la chaîne en mots (liste de tokens)

    # On filtre les mots : on garde seulement ceux qui NE SONT PAS dans la liste des mots inutiles
    mots_filtres = [mot for mot in mots if mot not in mots_inutiles]

    return mots_filtres  # On retourne la liste des mots utiles

def train_test_split(df, label_col, taux=0.05, random_seed=42):
    """
    découpe un DataFrame en deux DataFrames (train et test) en respectant la proportion de chaque classe.
    
    Args:
        df (pd.DataFrame): Le DataFrame complet à découper.
        label_col (str): Le nom de la colonne des labels/classes.
        taux (float): La proportion (entre 0 et 1) d'exemples à prendre pour le jeu d'entraînement.
        random_seed (int): Graine pour la reproductibilité du tirage aléatoire.
    
    Returns:
        df_train (pd.DataFrame): Sous-ensemble d'entraînement avec environ 'taux' exemples par classe.
        df_test (pd.DataFrame): Sous-ensemble test avec le reste des exemples.
    """
    
    # Fixer la graine aléatoire pour reproductibilité
    np.random.seed(random_seed)
    
    # Initialiser les DataFrames vides pour stocker train et test
    df_train = None
    df_test = None
    
    # Récupérer toutes les classes uniques dans la colonne label
    les_labels = df[label_col].unique()
    
    # Boucle sur chaque classe pour découper stratifié
    for l in les_labels:
        
        # Nombre total d'exemples pour cette classe
        nb_total = df[label_col].value_counts()[l]
        
        # Calcul du nombre d'exemples à prendre pour l'entraînement
        nb_pris = int(nb_total * taux)
        
        # Liste des indices des exemples appartenant à la classe l
        les_ids = df[df[label_col] == l].index.to_list()
        
        # Mélanger la liste des indices pour un tirage aléatoire
        np.random.shuffle(les_ids)
        
        # Sélectionner les indices pour train et test
        train_ids = les_ids[:nb_pris]
        test_ids = les_ids[nb_pris:]
        
        # Extraire les sous-DataFrames correspondant
        df_train_part = df.loc[train_ids]
        df_test_part = df.loc[test_ids]
        
        # Concaténer ces sous-DataFrames dans les DataFrames finaux
        if df_train is None:
            df_train = df_train_part
        else:
            df_train = pd.concat([df_train, df_train_part])
        
        if df_test is None:
            df_test = df_test_part
        else:
            df_test = pd.concat([df_test, df_test_part])
    
    # Mélanger les exemples dans les DataFrames finaux pour éviter un ordre trié par classe
    df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Retourner les deux DataFrames
    return df_train, df_test