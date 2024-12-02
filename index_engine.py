from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
import re

def preprocess(query):
    query = re.sub(r"[^A-Za-z0-9 ]+", "", query)  
    return query.lower()

def connect():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["assignment4"]
    doc_collection = db["documents"]
    term_collection = db["terms"]
    return doc_collection, term_collection

def store_documents(doc_collection):
    doc_collection.delete_many({})
    docs = [
        {"_id": 1, "content": "After the medication, headache and nausea were reported by the patient."},
        {"_id": 2, "content": "The patient reported nausea and dizziness caused by the medication."},
        {"_id": 3, "content": "Headache and dizziness are common effects of this medication."},
        {"_id": 4, "content": "The medication caused a headache and nausea, but no dizziness was reported."}
    ]
    doc_collection.insert_many(docs)

def store_index(doc_collection, term_collection):
    term_collection.delete_many({})
    docs = list(doc_collection.find({}, {"_id": 1, "content": 1}))
    processed_docs = [preprocess(doc["content"]) for doc in docs]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(processed_docs)

    vocab = vectorizer.vocabulary_ 
    terms = []
    term_id = 1
    for term, pos in vocab.items():
        docs_with_term = []
        for doc_id, tfidf_val in enumerate(tfidf_matrix[:, pos].toarray().flatten()):
            if tfidf_val > 0:
                docs_with_term.append({"doc_id": docs[doc_id]["_id"], "tfidf": round(tfidf_val, 4)})

        terms.append({
            "_id": term_id, 
            "term": term,
            "pos": pos,
            "docs": docs_with_term
        })
        term_id += 1

    term_collection.insert_many(terms)

    return vectorizer, tfidf_matrix


def rank(query, vectorizer, tfidf_matrix, input_docs):

    preprocessed_query = preprocess(query)

    query_vector = vectorizer.transform([preprocessed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    ranked = similarity_scores.argsort()[::-1]
    return [
        {"content": input_docs[i]['content'], "score": similarity_scores[i]}
        for i in ranked
    ]

def fetch_documents(engine):
    
    docs_from_db = list(engine.find({}, {"_id": 0, "content": 1}))
    
    return docs_from_db

def main():
    doc_collection, term_collection = connect()
    
    store_documents(doc_collection)
    docs = fetch_documents(doc_collection)
    vectorizer, tfidf_matrix = store_index(doc_collection, term_collection)

    queries = [
        "nausea and dizziness",
        "effects",
        "nausea was reported",
        "dizziness",
        "the medication"
    ]

    for q_id, query in enumerate(queries, 1):
        print(f"Query {q_id}: {query}")
        ranked_docs = rank(query, vectorizer, tfidf_matrix, docs)
        for result in ranked_docs:
                print(f"{result['content']}, Score: {result['score']:.4f}")
        print()

if __name__ == "__main__":
    main()
