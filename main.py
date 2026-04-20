import pickle
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

FAISS_INDEX_PATH = r"D:\arxiv_faiss_index.bin"
METADATA_PATH    = r"D:\arxiv_metadata.pkl"
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "llama3.2"

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    df_subset = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Building TF-IDF index...")
texts = df_subset["text"].tolist()
vectorizer   = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)

print("Backend ready. Records:", len(df_subset))


def retrieve_and_rerank(query, top_k=20, alpha=0.5):
    q_vec = model.encode([query])
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec.astype("float32"), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row             = df_subset.iloc[idx]
        authority_boost = int(row["authority"])
        final_score     = float(score) * (1 + alpha * authority_boost)
        results.append({
            "chunk_id":         int(row["chunk_id"]),
            "title":            str(row["title"]),
            "abstract":         str(row["abstract"])[:300],
            "categories":       str(row["categories"]),
            "doc_type":         str(row["doc_type"]),
            "authority":        authority_boost,
            "cosine_similarity": round(float(score), 4),
            "final_score":      round(float(final_score), 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:3]


def keyword_search(query):
    q_vec  = vectorizer.transform([query])
    scores = sklearn_cosine(q_vec, tfidf_matrix).flatten()
    return round(float(scores.max()), 4)


def answer_question(query):
    top_chunks = retrieve_and_rerank(query)

    context = ""
    for i, chunk in enumerate(top_chunks):
        full_abstract = df_subset.iloc[chunk["chunk_id"]]["abstract"]
        context += "Document {} ({}):\n".format(i + 1, chunk["doc_type"])
        context += "Title: {}\n".format(chunk["title"])
        context += "Abstract: {}\n\n".format(full_abstract)

    prompt = (
        "You are a research assistant helping researchers find information from academic papers.\n\n"
        "Below are 3 relevant research papers. Read them carefully and answer the question "
        "based on what these papers describe. Always cite which document number you are referring to.\n\n"
        "{}\n"
        "Question: {}\n\n"
        "Provide a detailed answer based on the papers above, citing document numbers:"
    ).format(context, query)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 600},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        answer = resp.json().get("response", "No response received from the model.")
    except requests.exceptions.ConnectionError:
        answer = "Could not connect to Ollama. Make sure it is running at http://localhost:11434."
    except requests.exceptions.Timeout:
        answer = "Ollama request timed out. The model may still be loading."
    except Exception as e:
        answer = "Error calling Ollama: {}".format(e)

    return answer, top_chunks
