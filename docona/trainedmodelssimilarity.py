#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: do similarity checks based on trained models
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

import pickle
import logging
import time
from datetime import datetime
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import os
import os.path
from os import path
import csv
import pandas as pd
from helper import convert_to_document_references, exists_citation_link_between
from resources import documentID_to_data, documentID_to_index, sim, sampledocuments

# ### Look up top n similar documents per sample document (doc2vec embeddings + cosine similarity)
def lookup_similar_documents_docvec_cosine(sample_documents, n, model):
    results = []

    num_samples = len(sample_documents)
    count = 1
    for item in sample_documents:
        start = time.time()
        print(str(count) + "/" + str(num_samples) + "...", end='', flush=True)
        count = count + 1
        similar_documents = model.docvecs.most_similar(documentID_to_index[item], topn=n)
        similar_documents_references = convert_to_document_references(similar_documents)
        for reference in similar_documents_references:
            method = "doc2vec_cosine"
            results.append([item,reference[0].replace(".txt",""),reference[1],method,exists_citation_link_between(item,reference[0])])
        end = time.time()
        timetaken = end-start
        print(str(timetaken) + "s")
    return results

# ### Look up top n similar documents per sample document (word2vec embeddings + word mover's distance)
def lookup_similar_documents_word2vec_wmd(sample_documents):
    results = []
    
    num_samples = len(sample_documents)
    count = 1
    for item in sample_documents:
        start = time.time()
        print(str(count) + "/" + str(num_samples) + "...", end='', flush=True)
        count = count + 1
        similar_documents = sim[documentID_to_data[item]]
        similar_documents_references = convert_to_document_references(similar_documents)
        for reference in similar_documents_references:
            method = "word2vec_wmd"
            if (str(item) != str(reference[0])):
                results.append([item,reference[0],reference[1],method,exists_citation_link_between(item,reference[0])])
        end = time.time()
        timetaken = end-start
        print(str(timetaken) + "s")
    return results

# ### Main function
def dosimilaritychecks(modeltype,model,distancemeasure):
    results = []
    if (modeltype == "doc2vec" and distancemeasure == "cosine"):
        print("doc2vec + cosine distance")
        print()
        results = lookup_similar_documents_docvec_cosine(sampledocuments,20, model)
        print()
    elif (modeltype == "word2vec" and distancemeasure == "wmd"):
        print("word2vec + word movers distance")
        print()
        results = lookup_similar_documents_word2vec_wmd(sampledocuments)
        print()
    if os.path.exists('../outputdata/results.csv') == False:
        results.insert(0,['source_document','similar_document','similarity_score','method','citation_link'])
        
    with open('../outputdata/results.csv', 'a', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(results)
