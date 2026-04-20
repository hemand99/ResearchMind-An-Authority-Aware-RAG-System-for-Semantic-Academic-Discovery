import os

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


os.chdir(r"D:\\")

FAISS_INDEX_PATH = "arxiv_faiss_index.bin"
METADATA_PATH = "arxiv_metadata.pkl"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
ALPHA = 0.5


st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index():
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_data(show_spinner="Loading paper metadata...")
def load_metadata():
    return pd.read_pickle(METADATA_PATH)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")


@st.cache_resource(show_spinner="Building TF-IDF matrix...")
def load_keyword_index(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


index = load_index()
df_subset = load_metadata().reset_index(drop=True)
model = load_model()
vectorizer, tfidf_matrix = load_keyword_index(tuple(df_subset["text"].fillna("").astype(str).tolist()))


def get_doc_color(doc_type):
    if str(doc_type).strip().lower() == "official research paper":
        return "#22c55e"
    return "#f59e0b"


def retrieve_and_rerank(query):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, 12)

    results = []
    for cosine_sim, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(df_subset):
            continue

        row = df_subset.iloc[idx]
        authority = float(row.get("authority", 0) or 0)
        final_score = float(cosine_sim) * (1 + ALPHA * authority)
        results.append(
            {
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "Untitled")),
                "categories": str(row.get("categories", "N/A")),
                "abstract": str(row.get("abstract", "")),
                "text": str(row.get("text", "")),
                "authority": authority,
                "doc_type": str(row.get("doc_type", "Informal Research Paper")),
                "cosine_similarity": float(cosine_sim),
                "final_score": float(final_score),
            }
        )

    results.sort(key=lambda item: item["final_score"], reverse=True)
    return results[:3]


def keyword_search(query):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return float(scores.max()) if scores.size else 0.0


def answer_question(query):
    top_chunks = retrieve_and_rerank(query)

    if not top_chunks:
        return "No relevant source papers were found for this question.", []

    context_blocks = []
    for idx, chunk in enumerate(top_chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Document {idx}",
                    f"Title: {chunk['title']}",
                    f"Categories: {chunk['categories']}",
                    f"Document Type: {chunk['doc_type']}",
                    f"Cosine Similarity: {chunk['cosine_similarity']:.4f}",
                    f"Final Authority Score: {chunk['final_score']:.4f}",
                    f"Abstract: {chunk['abstract']}",
                    f"Full Text Snippet: {chunk['text'][:2200]}",
                ]
            )
        )

    prompt = (
        "You are an enterprise research assistant answering questions from retrieved ArXiv papers.\n"
        "Use all three documents when relevant and synthesize them into one clear, detailed response.\n"
        "Do not give a shallow summary. Explain the main ideas, methods, findings, and differences across the papers.\n"
        "If the papers only partially answer the question, say what is supported by the sources and what remains uncertain.\n"
        "Cite supporting documents inline using [Doc 1], [Doc 2], [Doc 3].\n"
        "Structure the answer in this order:\n"
        "1. Direct answer\n"
        "2. Key evidence from the papers\n"
        "3. Comparison or synthesis across the papers\n"
        "4. Short conclusion\n\n"
        f"{chr(10).join(context_blocks)}\n\n"
        f"Question: {query}\n\n"
        "Write an elaborated answer grounded only in the documents above:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.15, "num_predict": 900},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        answer_text = response.json().get("response", "").strip() or "No answer returned by Ollama."
    except requests.exceptions.ConnectionError:
        answer_text = "Could not connect to Ollama at http://localhost:11434. Please make sure the Ollama server is running."
    except requests.exceptions.Timeout:
        answer_text = "The Ollama request timed out. The model may still be loading."
    except requests.RequestException as exc:
        answer_text = f"Ollama request failed: {exc}"

    return answer_text, top_chunks


def render_source_cards(sources):
    for idx, source in enumerate(sources, start=1):
        doc_color = get_doc_color(source["doc_type"])
        st.markdown(
            f"""
            <div class="source-card" style="border-left: 4px solid {doc_color};">
                <div class="source-topline">
                    <span class="source-rank">#{idx}</span>
                    <span class="source-badge" style="background: {doc_color}22; color: {doc_color}; border: 1px solid {doc_color}55;">
                        {source["doc_type"]}
                    </span>
                </div>
                <div class="source-title">{source["title"]}</div>
                <div class="source-meta">Categories: {source["categories"]}</div>
                <div class="score-grid">
                    <div class="score-chip">Cosine Similarity: {source["cosine_similarity"]:.4f}</div>
                    <div class="score-chip">Final Authority Score: {source["final_score"]:.4f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_score_chart(semantic_score, keyword_score):
    chart_df = pd.DataFrame(
        {
            "Search Type": ["Semantic Search", "Keyword Search"],
            "Score": [semantic_score, keyword_score],
            "Color": ["#38bdf8", "#f59e0b"],
        }
    )
    st.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "bar", "cornerRadiusTopLeft": 8, "cornerRadiusTopRight": 8},
            "encoding": {
                "x": {"field": "Search Type", "type": "nominal", "sort": None, "title": None},
                "y": {"field": "Score", "type": "quantitative", "title": "Score"},
                "color": {"field": "Color", "type": "nominal", "scale": None, "legend": None},
                "tooltip": [
                    {"field": "Search Type", "type": "nominal"},
                    {"field": "Score", "type": "quantitative", "format": ".4f"},
                ],
            },
            "width": "container",
        },
        use_container_width=True,
    )


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Sans+3:wght@400;600;700&display=swap');

        :root {
            --bg: #08111f;
            --panel: rgba(12, 24, 43, 0.92);
            --panel-strong: rgba(17, 32, 56, 0.98);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e6edf7;
            --muted: #90a3bf;
            --accent: #38bdf8;
            --accent-2: #14b8a6;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 30%),
                linear-gradient(180deg, #06101d 0%, #091524 48%, #07111e 100%);
            color: var(--text);
            font-family: "Source Sans 3", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 13, 24, 0.98), rgba(10, 22, 38, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.02em;
        }

        .shell {
            background: linear-gradient(180deg, rgba(10, 20, 36, 0.92), rgba(10, 20, 36, 0.8));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.4rem 1.4rem 1rem 1.4rem;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(10px);
        }

        .hero {
            padding: 0.4rem 0 1rem 0;
        }

        .eyebrow {
            display: inline-block;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #7dd3fc;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            margin: 0;
        }

        .hero-copy {
            margin-top: 0.75rem;
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 760px;
        }

        .sidebar-card {
            background: rgba(16, 28, 49, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.9rem;
        }

        .sidebar-label {
            color: #7dd3fc;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            margin-bottom: 0.35rem;
        }

        .sidebar-value {
            font-size: 0.98rem;
            line-height: 1.45;
            color: var(--text);
        }

        .formula {
            font-family: Consolas, monospace;
            background: rgba(8, 17, 31, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 14px;
            padding: 0.85rem;
            color: #bae6fd;
            margin-top: 0.5rem;
            font-size: 0.84rem;
        }

        .legend-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }

        .legend-pill {
            border-radius: 999px;
            padding: 0.42rem 0.8rem;
            font-size: 0.82rem;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.7);
        }

        .source-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(10, 19, 34, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
        }

        .source-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.7rem;
        }

        .source-rank {
            color: #7dd3fc;
            font-weight: 700;
            font-size: 0.84rem;
            letter-spacing: 0.08em;
        }

        .source-badge {
            border-radius: 999px;
            padding: 0.22rem 0.68rem;
            font-size: 0.76rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .source-title {
            font-size: 1.03rem;
            font-weight: 700;
            line-height: 1.4;
            margin-bottom: 0.4rem;
        }

        .source-meta {
            color: var(--muted);
            margin-bottom: 0.75rem;
        }

        .score-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
        }

        .score-chip {
            background: rgba(8, 17, 31, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            padding: 0.4rem 0.7rem;
            font-family: Consolas, monospace;
            font-size: 0.8rem;
            color: #c7d7ec;
        }

        .message-block {
            padding-top: 0.3rem;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
        }

        [data-testid="stChatInput"] {
            background: rgba(9, 18, 32, 0.88);
            border-top: 1px solid rgba(148, 163, 184, 0.08);
        }

        [data-testid="stChatInput"] textarea {
            background: rgba(13, 24, 43, 0.96) !important;
            color: var(--text) !important;
            border: 1px solid rgba(148, 163, 184, 0.12) !important;
            border-radius: 18px !important;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(56, 189, 248, 0.28);
            background: linear-gradient(135deg, rgba(20, 184, 166, 0.18), rgba(56, 189, 248, 0.22));
            color: #e0f2fe;
            font-weight: 700;
            padding: 0.7rem 1rem;
        }

        .section-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.05rem;
            margin: 0.4rem 0 0.9rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("## Enterprise RAG System")
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-label">Dataset</div>
            <div class="sidebar-value">10,000 ArXiv Papers</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-label">Embedding Model</div>
            <div class="sidebar-value">Sentence-Transformers all-mpnet-base-v2</div>
        </div>
        <div class="sidebar-card">
            <div class="sidebar-label">Authority Formula</div>
            <div class="formula">final_score = cosine_similarity x (1 + 0.5 x authority_boost)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="legend-row">
            <div class="legend-pill" style="color:#22c55e;">Official Research Papers</div>
            <div class="legend-pill" style="color:#f59e0b;">Informal Research Papers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    """
    <div class="shell">
        <div class="hero">
            <div class="eyebrow">Enterprise Documentation RAG</div>
            <h1 class="hero-title">Research chat over your ArXiv knowledge base</h1>
            <div class="hero-copy">
                Ask a research question and the system retrieves semantically relevant papers,
                applies authority-aware re-ranking, and generates an answer with Llama 3.2 via Ollama.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="message-block">{message["content"]}</div>', unsafe_allow_html=True)
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown('<div class="section-title">Top 3 source papers</div>', unsafe_allow_html=True)
            render_source_cards(message["sources"])
            with st.expander("Search score comparison", expanded=False):
                render_score_chart(message.get("semantic_score", 0.0), message.get("keyword_score", 0.0))


user_query = st.chat_input("Ask a research question about the indexed ArXiv papers...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f'<div class="message-block">{user_query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Searching the index and generating an answer..."):
            answer_text, top_chunks = answer_question(user_query)
            semantic_score = top_chunks[0]["final_score"] if top_chunks else 0.0
            keyword_score = keyword_search(user_query)

        st.markdown(f'<div class="message-block">{answer_text}</div>', unsafe_allow_html=True)

        if top_chunks:
            st.markdown('<div class="section-title">Top 3 source papers</div>', unsafe_allow_html=True)
            render_source_cards(top_chunks)
            with st.expander("Search score comparison", expanded=False):
                render_score_chart(semantic_score, keyword_score)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": top_chunks,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
        }
    )
