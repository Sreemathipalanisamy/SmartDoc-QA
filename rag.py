import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import tempfile
import re
import chromadb
from chromadb.config import Settings

# Force Chroma to reinitialize
os.environ["CHROMA_ALREADY_INITIALIZED"] = "false"

# === Caching OCR text extraction ===
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    doc = DocumentFile.from_pdf(tmp_path)
    model = ocr_predictor(pretrained=True)
    result = model(doc)

    pages_text = []
    for page_idx, page in enumerate(result.pages):
        page_text = ""
        for block in page.blocks:
            for line in block.lines:
                page_text += " ".join(word.value for word in line.words) + "\n"
        pages_text.append((page_text, page_idx + 1))

    os.unlink(tmp_path)
    return pages_text

# === Chunking text into manageable pieces ===
def improved_split_text(pages_text, chunk_size=500, overlap=100):
    all_chunks = []

    for page_text, page_num in pages_text:
        text = re.sub(r'\s+', ' ', page_text)
        text = re.sub(r'\n+', '\n', text)

        paragraphs = [p for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            words = para.split()
            if current_size + len(words) <= chunk_size:
                current_chunk.extend(words)
                current_size += len(words)
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    all_chunks.append({"text": chunk_text, "page": page_num})

                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                    current_size = len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

                if len(words) <= chunk_size:
                    current_chunk.extend(words)
                    current_size += len(words)
                else:
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = words[i:i + chunk_size]
                        chunk_text = " ".join(chunk)
                        all_chunks.append({"text": chunk_text, "page": page_num})
                    current_chunk = words[-overlap:] if len(words) > overlap else words
                    current_size = len(current_chunk)

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            all_chunks.append({"text": chunk_text, "page": page_num})

    return all_chunks

# === Hybrid search: embeddings + keywords ===
def hybrid_search(query, chunks, embedding_model, collection, k=5):
    query_embedding = embedding_model.encode([query])[0].tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k*2, len(chunks)),
        include=["documents", "metadatas", "distances"]
    )

    vector_chunks = vector_results["documents"][0]
    vector_metadatas = vector_results["metadatas"][0]

    keyword_matches = {}
    query_words = set(word.lower() for word in query.split() if len(word) > 3)

    for i, chunk in enumerate(vector_chunks):
        text = chunk.lower()
        score = sum(1 for word in query_words if word in text)
        keyword_matches[i] = score

    combined_results = []
    for i, (chunk, metadata) in enumerate(zip(vector_chunks, vector_metadatas)):
        combined_score = (k*2 - i) + keyword_matches.get(i, 0) * 2
        combined_results.append({
            "text": chunk,
            "page": metadata["page"],
            "score": combined_score
        })

    return sorted(combined_results, key=lambda x: x["score"], reverse=True)[:k]

# === Assemble text context from chunks ===
def assemble_context(ranked_chunks):
    formatted_chunks = []
    for chunk_info in ranked_chunks:
        formatted_chunk = f"[Page {chunk_info['page']}] {chunk_info['text']}"
        formatted_chunks.append(formatted_chunk)
    context = "\n\n".join(formatted_chunks)
    return context

# === Streamlit App ===
st.set_page_config(page_title="PDF Chat QA", layout="wide")
st.title("📄 SmartDoc Chat QA")

try:
    chroma_client = chromadb.EphemeralClient()
except:
    chroma_client = chromadb.Client()

uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

if uploaded_file:
    file_name = uploaded_file.name
    if "current_file" not in st.session_state or st.session_state["current_file"] != file_name:
        for key in ["pages_text", "chunks", "collection", "embedding_model", "tokenizer", "flan_model", "chat_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state["current_file"] = file_name

    if "pages_text" not in st.session_state:
        with st.spinner("🔍 Extracting text from PDF..."):
            st.session_state.pages_text = extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Text extraction complete! Found {len(st.session_state.pages_text)} pages.")

    if "chunks" not in st.session_state:
        with st.spinner("⚙️ Creating semantic chunks..."):
            st.session_state.chunks = improved_split_text(st.session_state.pages_text)
            st.success(f"✅ Created {len(st.session_state.chunks)} chunks!")

    if "embedding_model" not in st.session_state:
        with st.spinner("🧠 Loading embedding model..."):
            st.session_state.embedding_model = SentenceTransformer("all-mpnet-base-v2")
            st.success("✅ Embedding model loaded.")

    if "collection" not in st.session_state:
        with st.spinner("📦 Creating vector collection..."):
            try:
                chroma_client.delete_collection(name="pdf_chunks")
            except:
                pass
            chunks = st.session_state.chunks
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [{"page": chunk["page"]} for chunk in chunks]
            embeddings = st.session_state.embedding_model.encode(texts).tolist()

            collection = chroma_client.create_collection(name="pdf_chunks")
            collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=[f"id_{i}" for i in range(len(texts))])
            st.session_state.collection = collection
            st.success("✅ Vector database ready.")

    if "flan_model" not in st.session_state:
        with st.spinner("🚀 Loading FLAN-T5 model..."):
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            st.session_state.flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            st.success("✅ FLAN-T5 model loaded.")

    # === Chat History Setup ===
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the uploaded PDF..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("💡 Thinking..."):
            embedding_model = st.session_state.embedding_model
            collection = st.session_state.collection
            tokenizer = st.session_state.tokenizer
            model = st.session_state.flan_model
            chunks = st.session_state.chunks

            expand_prompt = f"Rewrite this question with additional related terms to improve search results: {prompt}"
            inputs = tokenizer(expand_prompt, return_tensors="pt", truncation=True, max_length=128)
            outputs = model.generate(**inputs, max_new_tokens=50)
            expanded_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

            search_query = f"{prompt} {expanded_query}"
            top_chunks = hybrid_search(search_query, chunks, embedding_model, collection, k=5)
            context = assemble_context(top_chunks)
            cited_pages = [chunk["page"] for chunk in top_chunks]

            final_prompt = (
                f"Answer this question based solely on the following context. "
                f"If the answer cannot be found in the context, say 'I cannot find information about this in the document.'\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Answer:"
            )

            inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=200)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            if cited_pages:
                st.caption(f"📌 Sources: Pages {', '.join(map(str, set(cited_pages)))}")
            with st.expander("📚 View Retrieved Context"):
                st.text_area("Used Context:", context, height=250)
