# ==========================================
# 📘 OFFLINE DOCUMENT Q&A CHATBOT (No FAISS, Streamlit)
# ==========================================

import streamlit as st
import PyPDF2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------------------------
# ✅ Step 1: Document Loader
# ------------------------------------------
def load_document(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type! Use PDF or TXT.")
    return text.strip()


# ------------------------------------------
# ✅ Step 2: Split into Chunks
# ------------------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# ------------------------------------------
# ✅ Step 3: Build Embeddings (No FAISS)
# ------------------------------------------
def build_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embedder, embeddings


# ------------------------------------------
# ✅ Step 4: Load QA Model
# ------------------------------------------
@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")


# ------------------------------------------
# ✅ Step 5: Retrieve Most Relevant Chunks (Cosine Similarity)
# ------------------------------------------
def retrieve_context(query, embedder, embeddings, chunks, top_k=2):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return " ".join([chunks[i] for i in top_indices])


# ------------------------------------------
# ✅ Step 6: Generate Answer
# ------------------------------------------
def generate_answer(query, embedder, embeddings, chunks, qa_model):
    context = retrieve_context(query, embedder, embeddings, chunks)
    prompt = f"Answer the question using only the context below:\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    response = qa_model(prompt, max_new_tokens=150)[0]["generated_text"]
    return response.strip()


# ------------------------------------------
# 🚀 Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="Offline Document Q&A Chatbot", layout="wide")
st.title("📘 Offline Document Q&A Chatbot (No FAISS, Works on Windows)")
st.markdown("Upload a **PDF** or **TXT** document and ask questions about it — runs fully offline!")

uploaded_file = st.file_uploader("📂 Upload your document", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("📖 Reading and processing document..."):
        text = load_document(uploaded_file)
        chunks = chunk_text(text)
        embedder, embeddings = build_embeddings(chunks)
        qa_model = load_qa_model()
        st.success("✅ Document processed and model loaded!")

    st.markdown("### 💬 Ask a question about your document")
    query = st.text_input("Enter your question here:")

    if query:
        with st.spinner("🤔 Thinking..."):
            answer = generate_answer(query, embedder, embeddings, chunks, qa_model)
        st.markdown("### 🤖 Answer:")
        st.write(answer)

else:
    st.info("Please upload a PDF or TXT file to begin.")
