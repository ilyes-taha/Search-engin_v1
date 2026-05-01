from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import logging
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import time
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('stopwords')

def read_pdf(path):
    try:
        with open(path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            content = ""
            for pg in pdf.pages:
                page_text = pg.extract_text()
                if page_text:
                    content += page_text
        return content
    except Exception as e:
        logger.warning("Could not parse PDF '%s': %s", path, e)
        return ""

nexora = Flask(__name__)

CORPUS_DIR = "corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

corpus = {}

for fname in os.listdir(CORPUS_DIR):
    fpath = os.path.join(CORPUS_DIR, fname)
    if fname.endswith(".txt"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                corpus[fname] = f.read()
        except Exception as e:
            logger.warning("Could not read file '%s': %s", fpath, e)
    elif fname.endswith(".pdf"):
        content = read_pdf(fpath)
        if content:
            corpus[fname] = content

french_stopwords = set(stopwords.words('french'))

def normalize(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in french_stopwords]
    return " ".join(tokens)

normalized_corpus = {fname: normalize(body) for fname, body in corpus.items()}

engine = None
doc_matrix = None
doc_names = []

if normalized_corpus:
    non_empty = {k: v for k, v in normalized_corpus.items() if v.strip()}
    if non_empty:
        try:
            engine = TfidfVectorizer()
            doc_matrix = engine.fit_transform(non_empty.values())
            doc_names = list(non_empty.keys())
        except ValueError as e:
            logger.warning("Could not build search index: %s", e)
    else:
        logger.info("Corpus contains only stopwords or empty documents. Search index not built.")
else:
    logger.info("No documents found in corpus directory '%s'. Add .txt or .pdf files to enable search.", CORPUS_DIR)

@nexora.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if engine is None:
            return render_template("results.html", hits=[], elapsed=0, query=request.form["q"], empty_corpus=True)

        t0 = time.time()
        query = request.form["q"]
        norm_query = normalize(query)
        query_vec = engine.transform([norm_query])

        similarity = np.dot(doc_matrix, query_vec.T).toarray().flatten()
        ranked = sorted(zip(doc_names, similarity), key=lambda x: x[1], reverse=True)

        hits = [
            (name, score, get_preview(corpus[name], query))
            for name, score in ranked if score > 0
        ]

        elapsed = round(time.time() - t0, 4)
        return render_template("results.html", hits=hits, elapsed=elapsed, query=query, empty_corpus=False)

    return render_template("search.html", empty_corpus=(engine is None))

@nexora.route("/suggest", methods=["GET"])
def suggest():
    if engine is None:
        return jsonify([])
    q = request.args.get("q", "").lower()
    matches = [w for w in engine.get_feature_names_out() if q in w][:8]
    return jsonify(matches)

def get_preview(text, query):
    parts = text.split('.')
    keywords = query.lower().split()
    for part in parts:
        if any(kw in part.lower() for kw in keywords):
            return f"…{part.strip()}…"
    return text[:200] + "…"

@nexora.route('/view/<doc_name>')
def view_doc(doc_name):
    fpath = os.path.join(CORPUS_DIR, doc_name)
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            body = f.read()
        return render_template("viewer.html", name=doc_name, content=body)
    except FileNotFoundError:
        return render_template("404.html"), 404

@nexora.route('/fetch/<doc_name>')
def fetch_doc(doc_name):
    return send_from_directory(CORPUS_DIR, doc_name, as_attachment=True)

if __name__ == "__main__":
    nexora.run(debug=True)
