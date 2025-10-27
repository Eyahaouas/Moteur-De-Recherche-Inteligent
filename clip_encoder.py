import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from functools import lru_cache

@lru_cache(maxsize=None)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

model, preprocess = load_model()
device = next(model.parameters()).device

def encode_text(text):
    try:
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            return model.encode_text(tokens)
    except Exception as e:
        print(f"Erreur d'encodage texte: {str(e)}")
        return None

def encode_image_url(url):
    try:
        # Télécharge l'image depuis l'URL avec un timeout de 10 secondes
        response = requests.get(url, timeout=10)
        
        # Vérifie que la requête a réussi (status code 200)
        response.raise_for_status()
        
        # Ouvre l'image téléchargée et la convertit en RGB
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Prétraite l'image et l'ajoute dans un batch (dimension supplémentaire)
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Désactive le calcul du gradient pour l'inférence
        with torch.no_grad():
            # Encode l'image avec le modèle et retourne le résultat
            return model.encode_image(image_input)
            
    except Exception as e:
        # En cas d'erreur, affiche le message et retourne None
        print(f"Erreur d'encodage image: {str(e)}")
        return None