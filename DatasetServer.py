import codecs
import json
import ir_datasets
from flask import Flask, request, jsonify
from flask_cors import CORS
from numpy import test
from OffileOperations import add_user_queries, load_user_queries
from QueryProcessing import expand_query_word2vec, query_expansion, get_relevant_queries
from TextPreprocessing import preprocess
from dataset_utils import load_antique_utils_we, load_quora_utils_we, load_quora_utils_tfidf, load_antique_utils_tfidf, load_quora_user_queries_utils_tfidf_doc_vectors, load_antique_user_queries_utils_inverted_index

app = Flask(__name__)
CORS(app, origins='http://127.0.0.1:5500')
CORS(app, origins='*')


def get_docs(relevant_docs, docstore):
    codecs.register_error("strict", codecs.ignore_errors)
    docs = docstore.get_many(relevant_docs)
    res = {}
    for doc_id, doc in docs.items():
        res[doc_id] = doc.text
    return res


def get_user_queries(relevant_queries, name):
    res = {}
    i = 0
    user_queries = load_user_queries(name)
    for query_id in relevant_queries:
        res[i] = user_queries[query_id]
        i += 1
    return res


load_antique_utils_we()
# load_antique_utils_tfidf()
load_quora_utils_we()
# load_quora_utils_tfidf()

load_antique_user_queries_utils_inverted_index()
load_quora_user_queries_utils_tfidf_doc_vectors()


@app.route('/query', methods=['POST'])
def post_query():
    query_data = request.get_json()
    query_text = query_data['query']
    dataset_name = query_data['dataset_name']  # Extract the additional parameter
    print(query_text, dataset_name)

    if dataset_name == "A":
        antique_dataset = ir_datasets.load("antique")
        antique_docstore = antique_dataset.docs_store()
        add_user_queries(query_text, dataset_name)
        expanded_results = query_expansion(query_text, dataset_name, 10)
        res = get_docs(expanded_results, antique_docstore)
        print("Antique", res)
    if dataset_name == "Q":
        quora_dataset = ir_datasets.load("beir/quora")
        quora_docstore = quora_dataset.docs_store()
        add_user_queries(query_text, dataset_name)
        expanded_results = query_expansion(query_text, dataset_name, 10)
        res = get_docs(expanded_results, quora_docstore)
        print("Quora", res)
    response = {'relevant_docs': res}
    return jsonify(response)



@app.route('/suggestions', methods=['POST'])
def post_suggestions():
    query_data = request.get_json()
    query_text = query_data['query']
    dataset_name = query_data['dataset_name']  # Extract the additional parameter
    print(query_text, dataset_name)

    if dataset_name == "A":
        ppq = preprocess(query_text)
        expanded_query = expand_query_word2vec(ppq, dataset_name)
        relevant_queries = get_relevant_queries(query_text, dataset_name)
        queries = get_user_queries(relevant_queries, dataset_name)
        result = []
        result.append(expanded_query)
        for query in queries:
            result.append(queries[query])
    if dataset_name == "Q":
        ppq = preprocess(query_text)
        expanded_query = expand_query_word2vec(ppq, dataset_name)
        relevant_queries = get_relevant_queries(query_text, dataset_name)
        queries = get_user_queries(relevant_queries, dataset_name)
        result = []
        result.append(expanded_query)
        for query in queries:
            result.append(queries[query])
    response = {'suggestions': result}
    return response


@app.route('/lastSearches', methods=['POST'])
def post_last_searches():
    query_data = request.get_json()
    dataset_name = query_data['dataset_name']  # Extract the additional parameter
    print(dataset_name)

    if dataset_name == "A":
        res = []
        user_queries = load_user_queries("A")
        i = 0
        for user_query in user_queries:
            if (i >= len(user_queries) - 5):
                res.append(user_queries[user_query])
            i += 1
    if dataset_name == "Q":
        res = []
        user_queries = load_user_queries("Q")
        i = 0
        for user_query in user_queries:
            if (i >= len(user_queries) - 5):
                res.append(user_queries[user_query])
            i += 1
    response = {'suggestions': res}
    return response


if __name__ == '__main__':
    app.run()
