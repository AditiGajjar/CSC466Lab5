import os
import math
import string
from collections import defaultdict
import pandas as pd
from porterStemmer import PorterStemmer


# Add stopwords into an array
stopwords = []
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

def read_documents(root_dir):
    documents = {}
    doc_authors = {}
    stemmer = PorterStemmer()

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(subdir, file)
                author = os.path.basename(subdir)
                with open(file_path, 'r', encoding='utf-8') as f:
                    contents = f.read().split()
                    content = []
                    for term in contents:
                        if term.lower() in stopwords:
                            continue
                        else:
                            term = term.translate(str.maketrans('', '', string.punctuation))
                            stemmed_term = stemmer.stem(term, 0, len(term) - 1)
                            content.append(stemmed_term)
                    doc_id = file
                    documents[doc_id] = content
                    doc_authors[doc_id] = author
    return documents, doc_authors

def compute_tf(documents):
    tf = {}
    for doc_id, doc in documents.items():
        tf[doc_id] = {}
        doc_length = len(doc)
        for term in doc:
            if term not in tf[doc_id]:
                tf[doc_id][term] = 1 / doc_length
            else:
                tf[doc_id][term] += 1 / doc_length
    return tf

def compute_df(documents):
    df = defaultdict(int)
    for _, doc in documents.items():
        for term in doc:
            df[term] += 1
    return df

def compute_idf(df, N):
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(N / freq)
    return idf

def compute_tf_idf(tf, idf, documents):
    data = []
    for doc_id, doc_tf in tf.items():
        doc_data = {}
        for term in set.union(*[set(doc) for doc in documents.values()]):
            doc_data[term] = doc_tf.get(term, 0) * idf.get(term, 0)
        data.append(doc_data)
    df = pd.DataFrame(data, index=tf.keys())
    return df

def write_vectorized_output(vectorized_docs, output_filename):
    vectorized_docs.to_csv(output_filename)

def write_ground_truth(doc_authors, ground_truth_filename):
    with open(ground_truth_filename, 'w', encoding='utf-8') as f:
        for doc_id, author in doc_authors.items():
            f.write(f"{doc_id},{author}\n")

def main(root_dir, output_filename, ground_truth_filename):
    documents, doc_authors = read_documents(root_dir)
    tf = compute_tf(documents)
    df = compute_df(documents)
    idf = compute_idf(df, len(documents))
    tf_idf = compute_tf_idf(tf, idf, documents)
    
    write_vectorized_output(tf_idf, output_filename)
    write_ground_truth(doc_authors, ground_truth_filename)


if __name__ == "__main__":
    root_dir = "C50/C50Train"
    output_filename = "tfidf_vectorized_docs.csv"
    ground_truth_filename = "ground_truth.csv"
    main(root_dir, output_filename, ground_truth_filename)

# to run in terminal: python3 textVectorizer.py


