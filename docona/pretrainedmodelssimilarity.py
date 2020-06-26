#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: computes similarity between all documents in the corpus
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

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

# ### Look up top n similar documents per sample document (googlenews doc2vec embeddings + cosine similarity)
def lookup_similar_documents_docvec_cosine(sample_documents, n, model, pretrained_embedding_name):
    results = []

    for item in sample_documents:
        similar_documents = model.docvecs.most_similar(documentID_to_index[item], topn=n)
        similar_documents_references = convert_to_document_references(similar_documents)
        for reference in similar_documents_references:
            method = pretrained_embedding_name + "_doc2vec_wmd"
            results.append([item,reference[0].replace(".txt",""),reference[1],method,exists_citation_link_between(item,reference[0])])

    return results

# ### Look up top n similar documents per sample document (googlenews word2vec embeddings + word mover's distance)
def lookup_similar_documents_word2vec_wmd(sample_documents, pretrained_embedding_name):
    results = []
    
    for item in sample_documents:
        similar_documents = sim[documentID_to_data[item]]
        similar_documents_references = convert_to_document_references(similar_documents)
        for reference in similar_documents_references:
            method = pretrained_embedding_name + "_word2vec_wmd"
            if (str(item) != str(reference[0])):
                results.append([item,reference[0],reference[1],method,exists_citation_link_between(item,reference[0])])

    return results

# ### Main function
def dosimilaritychecks(modeltype,pretrained_embedding_name,model,distancemeasure):
    results = []
    if (modeltype == "doc2vec" and distancemeasure == "cosine"):
        results = lookup_similar_documents_docvec_cosine(sampledocuments, 20, model, pretrained_embedding_name)
    elif (modeltype == "word2vec" and distancemeasure == "wmd"):
        results = lookup_similar_documents_word2vec_wmd(sampledocuments, pretrained_embedding_name)

    if os.path.exists('../outputdata/results.csv') == False:
        results.insert(0,['source_document','similar_document','similarity_score','method','citation_link'])
        
    with open('../outputdata/results.csv', 'a', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(results)
