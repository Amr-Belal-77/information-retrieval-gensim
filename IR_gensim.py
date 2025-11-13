# import necessary libraries
import nltk
import gensim
import string

from nltk.corpus.reader import PlaintextCorpusReader

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim import corpora, models

from gensim.similarities import MatrixSimilarity

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 0. Load Corpus <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Load the corpus from the specified directory
directory=r"E:\uni\7th semester\NLP - Dr Abeer Hassan\project\information retrieval\txt_sentoken"
corpus = PlaintextCorpusReader(str(directory), r"(pos|neg)/.*\.txt")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1. Data Understanding <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# print("Number of documents:", len(corpus.fileids()))

pos_files = [f for f in corpus.fileids() if f.startswith("pos/")]
neg_files = [f for f in corpus.fileids() if f.startswith("neg/")]

# print("Positive files:", len(pos_files))
# print("Negative files:", len(neg_files))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2. Data Preprocessing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Download NLTK resources (first run will fetch; then it’s cached)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # |.convert to lowercase
    text = text.lower()
    # ||.word tokenization
    tokens = word_tokenize(text)
    # |||.remove punctuation & numbers
    tokens = [t for t in tokens if t.isalpha()]
    # |V.remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # V.lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


# cleaned docs
cleaned_docs = [clean_text(corpus.raw(f)) for f in corpus.fileids()]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3. Vectorization TF-IDF matrix <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dictionary = corpora.Dictionary(cleaned_docs)
# print(f" Number of unique words: {len(dictionary)}")

# Bag-of-Words for each doc
bow_corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]

# TF-IDF model and transformed corpus
tfidf_model = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf_model[bow_corpus]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4 . Similarity Computation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

index = MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5. Interactive Search Console <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == "__main__":
    # run the interactive console only when running this file directly
    print(" Similarity index built!\n")

    while True:
        query = input("Write what you want to search... (write EXIT if you want to close.)").strip()
        if query.lower() == "exit":
            print("Closing the search engine. Goodbye!")
            break
        
        # Prepare query exactly like documents
        query_tokens = clean_text(query)
        query_bow = dictionary.doc2bow(query_tokens)
        query_tfidf = tfidf_model[query_bow]

        # Similarity scores against entire corpus
        sims = index[query_tfidf]
        ranked_docs = sorted(list(enumerate(sims)), key=lambda x: -x[1])

        # Show top-10 results
        print("\n Top 10 Most Similar Documents:\n")
        top_docs = [corpus.fileids()[i] for i, _ in ranked_docs[:10]]
        for rank, (i, score) in enumerate(ranked_docs[:10], start=1):
            doc_name = corpus.fileids()[i]
            label = "POS" if "pos" in doc_name else "NEG"
            doc_text = corpus.raw(doc_name)[:150].replace("\n", " ")
            print(f"{rank}.  {doc_name} | [{label}] | Similarity: {score:.3f}")
            print(f"   {doc_text}...")
            print("-" * 80)

        # Simple per-query evaluation using a heuristic label from the query
        print("\n Evaluation for this query:")
        positive_keywords = ["good", "great", "amazing", "love", "wonderful", "excellent"]
        expected_label = "pos" if any(word in query.lower() for word in positive_keywords) else "neg"

        # Ground truth per result = whether file path contains expected label
        y_true = [1 if expected_label in doc else 0 for doc in top_docs]
        # All returned results are predicted as relevant (1)
        y_pred = [1] * len(top_docs)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Expected label: {expected_label.upper()}")
        print(f"Precision: {prec:.2f} | Recall: {rec:.2f} | F1-score: {f1:.2f}")
        print("=" * 90 + "\n")