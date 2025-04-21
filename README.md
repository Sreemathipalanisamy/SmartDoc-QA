# 📄 SmartDoc QA

SmartDoc QA is a web-based document question-answering application that enables users to upload PDF documents, extract their content, and ask natural language questions to receive context-aware answers using state-of-the-art AI models.

---

## 📚 Table of Contents

- [📌 Description](#-description)  
- [✨ Features](#-features)  
- [⚙️ Installation](#️-installation)  
- [🚀 Usage](#-usage)

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
