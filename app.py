from flask import Flask, render_template, request
from gensim.similarities import MatrixSimilarity
from sklearn.metrics import precision_score, recall_score, f1_score

# نستورد كل الحاجات من الكود بتاعك
from IR_gensim import corpus, dictionary, tfidf_model, clean_text

app = Flask(__name__)

# نبني الـ index مرة واحدة
corpus_tfidf = tfidf_model[[dictionary.doc2bow(clean_text(corpus.raw(f))) for f in corpus.fileids()]]
index = MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

positive_keywords = ["good", "great", "amazing", "love", "wonderful", "excellent"]

def evaluate_results(query, top_docs):
    expected_label = "pos" if any(w in query.lower() for w in positive_keywords) else "neg"
    y_true = [1 if expected_label in doc else 0 for doc in top_docs]
    y_pred = [1] * len(top_docs)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return expected_label, prec, rec, f1

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    evaluation = None

    if request.method == "POST":
        query = request.form["query"]
        query_tokens = clean_text(query)
        query_bow = dictionary.doc2bow(query_tokens)
        query_tfidf = tfidf_model[query_bow]
        sims = index[query_tfidf]
        ranked_docs = sorted(list(enumerate(sims)), key=lambda x: -x[1])

        top_docs = [corpus.fileids()[i] for i, _ in ranked_docs[:10]]
        for i, score in ranked_docs[:10]:
            doc_name = corpus.fileids()[i]
            label = "POS" if "pos" in doc_name else "NEG"
            doc_text = corpus.raw(doc_name)[:200].replace("\n", " ")
            results.append({"doc": doc_name, "label": label, "score": round(float(score), 3), "text": doc_text})

        expected_label, prec, rec, f1 = evaluate_results(query, top_docs)
        evaluation = {
            "expected": expected_label.upper(),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2)
        }

    return render_template("index.html", results=results, evaluation=evaluation)

if __name__ == "__main__":
    app.run(debug=True)
