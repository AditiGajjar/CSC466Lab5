import os
import math
import string
from collections import defaultdict

# to do:
# 1. Fix splitting attributes (DONE)
# 2. Remove stopwords (DONE)
# 3. Implement stemming
# 4. Apply to all Authors in the C50Train

# Add stopwords into an array
stopwords = []
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

def read_documents(root_dir):
    documents = {}
    doc_authors = {}

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
                            pass
                        else:
                            x = term.translate(str.maketrans('', '', string.punctuation))
                            content.append(x)
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

def compute_tf_idf(tf, idf):
    tf_idf = {}
    for doc_id, doc_tf in tf.items():
        tf_idf[doc_id] = {}
        for term, tf_val in doc_tf.items():
            tf_idf[doc_id][term] = tf_val * idf.get(term, 0)
    return tf_idf

def write_vectorized_output(vectorized_docs, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as f:
        for doc_id, tf_idf_vector in vectorized_docs.items():
            output_line = f"{doc_id}," + ",".join([f"{word}:{tf_idf}" for word, tf_idf in tf_idf_vector.items()])
            f.write(output_line + "\n")

def write_ground_truth(doc_authors, ground_truth_filename):
    with open(ground_truth_filename, 'w', encoding='utf-8') as f:
        for doc_id, author in doc_authors.items():
            f.write(f"{doc_id},{author}\n")

def main(root_dir, output_filename, ground_truth_filename):
    documents, doc_authors = read_documents(root_dir)
    tf = compute_tf(documents)
    df = compute_df(documents)
    idf = compute_idf(df, len(documents))
    tf_idf = compute_tf_idf(tf, idf)
    
    write_vectorized_output(tf_idf, output_filename)
    write_ground_truth(doc_authors, ground_truth_filename)


# example implementation for debugging use the first author: Aaron Pressman
if __name__ == "__main__":
    root_dir = "C50/C50train/AaronPressman"
    output_filename = "tfidf_vectorized_docs.csv"
    ground_truth_filename = "ground_truth.csv"
    main(root_dir, output_filename, ground_truth_filename)

# to run in terminal: python3 textVectorizer.py
