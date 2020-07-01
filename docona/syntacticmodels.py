#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: syntactic models 
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

from helper import convert_to_document_references, exists_citation_link_between
from resources import datafortraining, documentID_to_data, documentID_to_index, sampledocuments, documentID_to_tokenized, data_to_tokenized
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import pickle
import os
import os.path
from os import path
import csv
import time
import math
import operator

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def gettfidfmodel():
    tfidf_data = None  
    if os.path.exists(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), "tfidfmodel.pickle")):
        # Model exists
        file = open(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), "tfidfmodel.pickle"), 'rb')
        tfidf_data = pickle.load(file)
        file.close()
    else:
        # No model. Train one.
        tfidfvect = TfidfVectorizer(use_idf=True)
        tfidf_data = tfidfvect.fit_transform(datafortraining)
        pickle.dump(tfidf_data, open(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), "tfidfmodel.pickle"), 'wb'))
    return tfidf_data

def getngrammodel(n):
    tfidf_data = None  
    if os.path.exists(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), str(n)+"grammodel.pickle")):
        # Model exists
        file = open(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), str(n)+"grammodel.pickle"), 'rb')
        tfidf_data = pickle.load(file)
        file.close()
    else:
        # No model. Train one.
        tfidfvect = TfidfVectorizer(analyzer='word', ngram_range=(n,n), use_idf=True)
        tfidf_data = tfidfvect.fit_transform(datafortraining)
        pickle.dump(tfidf_data, open(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), str(n)+"grammodel.pickle"), 'wb'))

    return tfidf_data

def find_similar(tfidf_matrix, index, top_n):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

def lookup_similar_documents_tfidf_based(sample_documents, n, model,methodname):
    results = []
    num_samples = len(sample_documents)
    count = 1
    for item in sample_documents:
        start = time.time()
        print(str(count) + "/" + str(num_samples) + "...", end='', flush=True)
        count = count + 1
        index = documentID_to_index[item]                                                   # Look up this documents index in the TFIDF matrix
        similar_documents = find_similar(model, index, n)                                   # Look up top n similar documents for this document
        similar_documents_references = convert_to_document_references(similar_documents)
        for reference in similar_documents_references:
            results.append([item,reference[0],reference[1],methodname,exists_citation_link_between(item,reference[0])])
        end = time.time()
        timetaken = end-start
        print(str(timetaken) + "s")
    return results

def lookup_similar_documents_jaccard(sample_documents, n, methodname):
    results = []

    num_samples = len(sample_documents)
    count = 1
    for item in sample_documents:
        start = time.time()
        print(str(count) + "/" + str(num_samples) + "...", end='', flush=True)
        count = count + 1
        current_dict = {}
        for k,v in documentID_to_data.items():
            if k != item:
                current_sim_val = jaccard_similarity(documentID_to_tokenized[item], data_to_tokenized[v])
                current_dict[k] = current_sim_val
        sorted_dict = sorted(current_dict.items(), key=operator.itemgetter(1))
        topn = sorted_dict[-n:]
        for reference in topn:
            results.append([item,reference[0],reference[1], methodname, exists_citation_link_between(item,reference[0])])
        end = time.time()
        timetaken = end-start
        print(str(timetaken) + "s")
    return results

# ### Main function
def dosyntacticsimilaritychecks(methodname,model):
    results = []

    if (methodname == "tfidf") or ("gram" in methodname):
        results = lookup_similar_documents_tfidf_based(sampledocuments, 20, model, methodname)
    elif (methodname == "jaccard"):
        results = lookup_similar_documents_jaccard(sampledocuments, 20, methodname)

    if os.path.exists('../outputdata/results.csv') == False:
        results.insert(0,['source_document','similar_document','similarity_score','method','citation_link'])
        
    with open('../outputdata/results.csv', 'a', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(results)