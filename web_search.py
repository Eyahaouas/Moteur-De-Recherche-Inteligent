# web_search.py
import requests       # Pour faire des requêtes HTTP
import os             # Pour accéder aux variables d'environnement
from dotenv import load_dotenv  # Pour charger les variables d'environnement

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Fonction pour effectuer une recherche web via l'API Google Custom Search
def search_web(query):
    try:
        # Récupérer la clé API et l'ID du moteur depuis l'environnement
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            raise ValueError("Google API Key ou CSE ID manquant dans le fichier .env")

        # Paramètres de la requête API
        params = {
            "key": api_key,  
            "cx": cse_id,  
            "q": query,      
            "num": 10,       
            "fields": "items(title,link,snippet,pagemap(cse_image,cse_thumbnail))"
        }
        
        # Envoyer la requête GET à l'API Google
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",  
            params=params,     
            timeout=15         
        )
        
        # Vérifier si la requête a échoué
        response.raise_for_status()
        
        # Convertir la réponse en JSON et retourner les items
        data = response.json()
        return data.get("items", [])

    except Exception as e:
        # Afficher l'erreur dans la console
        print(f"Erreur de recherche: {str(e)}")
        return []
