# 📄 SmartDoc QA

SmartDoc QA is a web-based document question-answering application that enables users to upload PDF documents, extract their content, and ask natural language questions to receive context-aware answers using state-of-the-art AI models.

---

## 📚 Table of Contents

- [📌 Description](#-description)  
- [✨ Features](#-features)  
- [⚙️ Installation](#️-installation)  
- [🚀 Usage](#-usage)
- [🧠 Powered By](#-powered by)

---

## 📌 Description

SmartDoc QA is an intelligent, interactive document analysis tool built with **Streamlit**, combining OCR (Optical Character Recognition), semantic embeddings, hybrid search, and large language models to enable users to query the content of their PDF documents directly. It offers an intuitive interface and fast responses derived entirely from the uploaded file.

---

## ✨ Features

- 📤 Upload and analyze PDF files
- 🔍 Extract text using OCR (via `doctr`)
- 🧠 Generate semantic chunks from document content
- 📌 Perform hybrid search (semantic + keyword)
- 🤖 Answer questions using FLAN-T5 language model
- 📄 View source page numbers and retrieved context
- 🖥️ Responsive and interactive web interface using Streamlit

---

## ⚙️ Installation

To run SmartDoc QA locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/smartdoc-qa.git

2. **Navigate to the project directory**
   ```bash
   cd smartdoc-qa

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the application**
   ```bash
   streamlit run rag.py

---

## 🚀 Usage
Once the application is running in your browser:

- 📥 Upload a PDF document
- 🔄 The app will automatically extract and chunk the content
- ❓ Enter your question in the input box
- ✅ Get an answer based solely on the document
- 📚 Expand the retrieved context to verify source content
- 🔗 View the cited pages used in generating the answer

---

## 🧠 Powered By

- Streamlit — for interactive UI
- doctr — for OCR text extraction
- ChromaDB — for vector storage and retrieval
- SentenceTransformers — for semantic embeddings
- HuggingFace Transformers — FLAN-T5 for question answering
