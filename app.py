from flask import Flask, request, jsonify, render_template
from clip_encoder import encode_text, encode_image_url
from web_search import search_web
from similarity import compute_similarities
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import tempfile
import clip

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = None, None

def load_clip_model():
    global model, preprocess
    if model is None:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_first_image(item):
    pagemap = item.get("pagemap", {})
    if "cse_image" in pagemap and pagemap["cse_image"]:
        return pagemap["cse_image"][0].get("src")
    if "cse_thumbnail" in pagemap and pagemap["cse_thumbnail"]:
        return pagemap["cse_thumbnail"][0].get("src")
    return None

def generate_query_from_vector(vector, top_k=5):
    """
    Génère une requête textuelle simple à partir du vecteur image CLIP.
    Ici c'est un exemple simplifié qui renvoie une liste de mots-clés simulés.
    Tu peux remplacer par un modèle OCR ou une vraie extraction.
    """
    # Exemple factice : utiliser un texte générique
    return "image content"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    load_clip_model()
    try:
        data = request.get_json()
        mode = data.get("mode", "text")

        if mode == "image":
            image_url = data.get("image_url")
            if not image_url:
                return jsonify({"error": "URL d'image manquante"}), 400
            vector = encode_image_url(image_url)
            if vector is None:
                return jsonify({"error": "Échec de l'encodage"}), 500

            # Générer une requête textuelle à partir de l'image
            query = generate_query_from_vector(vector)

        else:
            query = data.get("query", "").strip()
            if not query:
                return jsonify({"error": "Requête vide"}), 400
            vector = encode_text(query)
            if vector is None:
                return jsonify({"error": "Échec de l'encodage"}), 500

        results = search_web(query)
        scored = compute_similarities(vector, results, mode)

        return jsonify([{
            "url": item["link"],
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "image": get_first_image(item),
            "score": score
        } for item, score in scored])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload_search", methods=["POST"])
def upload_search():
    load_clip_model()
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Type de fichier non supporté"}), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        try:
            image = Image.open(temp_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                vector = model.encode_image(image_input)
            if vector is None:
                return jsonify({"error": "Échec de l'encodage"}), 500

            # Générer une requête textuelle à partir de l'image uploadée
            query = generate_query_from_vector(vector)

            results = search_web(query)
            scored = compute_similarities(vector, results, "image")

            return jsonify([{
                "url": item["link"],
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "image": get_first_image(item),
                "score": score
            } for item, score in scored])

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
