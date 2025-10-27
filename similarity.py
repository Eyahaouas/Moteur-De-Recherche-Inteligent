# Importer les modules nécessaires
import torch.nn.functional as F  # Pour les fonctions de similarité
from clip_encoder import encode_text, encode_image_url  # Pour encoder texte/images

# Fonction pour calculer les similarités entre un vecteur d'entrée et des résultats
def compute_similarities(input_vector, results, mode="text"):
    # Si le vecteur d'entrée est vide, retourner une liste vide
    if input_vector is None:
        return []
    
    # Liste pour stocker les résultats avec leur score
    scored_results = []
    
    # Parcourir chaque résultat
    for item in results:
        try:
            # Si mode image, encoder l'image du résultat
            if mode == "image":
                result_vector = encode_image_url(item.get("image", ""))
            else:
                # Mode texte : concaténer titre et snippet (ou URL)
                text = item.get("title", "") + " " + item.get("snippet", item.get("url", ""))
                # Encoder le texte (limitré à 77 caractères)
                result_vector = encode_text(text[:77])
            
            # Si l'encodage a réussi
            if result_vector is not None:
                # Calculer la similarité cosinus
                score = F.cosine_similarity(input_vector, result_vector).item()
                # Ajouter le résultat et son score à la liste
                scored_results.append((item, score))
        
        # En cas d'erreur, afficher et continuer
        except Exception as e:
            print(f"Erreur de similarité: {str(e)}")
            continue
    
    # Trier les résultats par score décroissant
    return sorted(scored_results, key=lambda x: x[1], reverse=True)