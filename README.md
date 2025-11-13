# 🚀 Information Retrieval System using Gensim & Flask

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gensim](https://img.shields.io/badge/Gensim-TF--IDF-green)
![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete **Information Retrieval (IR)** system built using **Python**, **NLTK**, **Gensim**, and a **Flask-based web interface**.  
The system retrieves and ranks relevant documents using **TF-IDF** and **cosine similarity**, and provides evaluation metrics for each query.

---

# 📑 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Architecture](#-system-architecture)
5. [Folder Structure](#-folder-structure)
6. [How It Works](#-how-it-works)
7. [Installation](#-installation--setup)
8. [Usage](#-usage)
9. [Evaluation Metrics](#-evaluation-metrics)
10. [Future Improvements](#-future-enhancements)
11. [Author](#-author)

---

# 📘 Overview
This project demonstrates how to build a complete **Information Retrieval pipeline** using classical NLP techniques.  
It works on the **Movie Review Polarity Dataset** (`txt_sentoken`), consisting of **2000 labeled reviews** (POS/NEG).

The system includes:
- A fully functional **IR Engine**
- A **ranking module**
- A **precision/recall evaluation module**
- A **Flask web interface**

---

# ⚙ Features

### 🔹 Text Preprocessing
- Lowercasing  
- Tokenization  
- Stopword removal  
- Lemmatization  

### 🔹 TF-IDF Vectorization
- Vocabulary dictionary  
- Bag-of-Words model  
- Weighted TF-IDF vectors  

### 🔹 Similarity Search
- Cosine-similarity using Gensim’s `MatrixSimilarity`
- Top-k ranked document retrieval

### 🔹 Evaluation
- Precision  
- Recall  
- F1-Score  

### 🔹 GUI (Flask)
- Search bar  
- Ranked result table  
- Similarity scores  
- Evaluation metrics  

---

# 🧩 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| NLP | NLTK |
| Vectorization | Gensim (TF-IDF, Dictionary, BoW) |
| Evaluation | Scikit-Learn |
| Web App | Flask |

---

# 📐 System Architecture

```
User Query
     │
     ▼
[Preprocessing Module]
     │
     ▼
[TF-IDF Vectorizer] ← Corpus Preprocessed
     │
     ▼
[Similarity Engine]
     │
     ▼
[Ranking Module]
     │
     ▼
[Evaluation Module]
     │
     ▼
[Flask GUI Output]
```

---

# 📂 Folder Structure
```
information-retrieval/
│
├── IR_gensim.py           # Core IR engine
├── app.py                 # Flask web app
├── templates/
│   └── index.html         # GUI layout
├── txt_sentoken/
│   ├── pos/               # Positive movie reviews
│   └── neg/               # Negative movie reviews
└── README.md              # Documentation
```

---

# 🚀 How It Works

### **1. Corpus Loading**
Documents are loaded from POS and NEG folders via `PlaintextCorpusReader`.

### **2. Preprocessing**
Each document is:
- Lowercased  
- Tokenized  
- Cleaned  
- Lemmatized  

### **3. TF-IDF Model Creation**
- Build dictionary  
- Convert documents to Bag-of-Words  
- Apply TF-IDF weighting  

### **4. Similarity Computation**
The query is processed and compared to all documents using cosine similarity.

### **5. Ranking Output**
Top-10 documents are displayed with labels and similarity scores.

### **6. Evaluation**
Precision, Recall, and F1-Score are computed based on expected vs. retrieved sentiment label.

---

# 💻 Installation & Setup

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/Amr-Belal-77/information-retrieval-gensim.git
cd information-retrieval-gensim
```

### **2️⃣ Install Dependencies**
```bash
pip install nltk gensim scikit-learn flask
```

### **3️⃣ Run Console IR Engine**
```bash
python IR_gensim.py
```

### **4️⃣ Run Flask GUI**
```bash
python app.py
```

Open browser:
```
http://127.0.0.1:5000
```

---

# 🧪 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Accuracy of retrieved results |
| **Recall** | Coverage of relevant documents |
| **F1-Score** | Balance between precision & recall |

---

# 📈 Future Enhancements

- Add semantic embeddings: **Word2Vec**, **Doc2Vec**
- Integrate **BERT** for contextual retrieval
- UI improvement using **Bootstrap / React**
- Add ranking metrics: MAP, nDCG
- Add relevance feedback (Rocchio Algorithm)

---

# 👨‍💻 Author

**Amr Belal**  
Information Retrieval System — TF-IDF + Flask  
GitHub: *Amr-Belal-77*
