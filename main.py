import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import tempfile
import clip
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

# Initialisation app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = None, None

def load_clip_model():
    global model, preprocess
    if model is None:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

# Utils
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def search_web_pages(query):
    api_key = os.getenv("API_KEY")
    cx = os.getenv("SEARCH_ENGINE_ID")
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
        "num": 10
    }

    response = requests.get(url, params=params)
    return response.json().get("items", [])

def get_first_image(item):
    pagemap = item.get("pagemap", {})
    if "cse_image" in pagemap and pagemap["cse_image"]:
        return pagemap["cse_image"][0].get("src")
    if "cse_thumbnail" in pagemap and pagemap["cse_thumbnail"]:
        return pagemap["cse_thumbnail"][0].get("src")
    return None

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    load_clip_model()
    data = request.get_json()
    mode = data.get("mode", "text")

    if mode == "image":
        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "URL d'image manquante"}), 400

        img_data = requests.get(image_url).content
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.jpg")
        with open(temp_path, 'wb') as handler:
            handler.write(img_data)

        image = Image.open(temp_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            query_vector = model.encode_image(image_input)
        os.remove(temp_path)
        query = "Recherche par image"

    else:
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Requête vide"}), 400

        text_input = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_vector = model.encode_text(text_input)

    results = search_web_pages(query)
    response = []
    for item in results:
        response.append({
            "url": item.get("link"),
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "image": get_first_image(item),
            "score": None
        })

    return jsonify(response)

@app.route("/upload_search", methods=["POST"])
def upload_search():
    load_clip_model()
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

    image = Image.open(temp_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_vector = model.encode_image(image_input)

    results = search_web_pages("image similaire")
    scored = []

    for item in results:
        image_url = get_first_image(item)
        if image_url:
            try:
                img_data = requests.get(image_url, timeout=5).content
                temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_web.jpg")
                with open(temp_img_path, 'wb') as handler:
                    handler.write(img_data)

                web_image = Image.open(temp_img_path).convert("RGB")
                web_input = preprocess(web_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    web_vector = model.encode_image(web_input)

                score = torch.cosine_similarity(query_vector, web_vector).item()

                scored.append({
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "image": image_url,
                    "score": round(score, 3)
                })

                os.remove(temp_img_path)

            except Exception as e:
                print("Erreur image page web :", e)
                continue

    os.remove(temp_path)

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    return jsonify(scored)

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
