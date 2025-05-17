import string

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