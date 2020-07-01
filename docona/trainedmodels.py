#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: train models from provided documents
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.similarities import WmdSimilarity
from gensim.test.utils import get_tmpfile
import os
import os.path
from os import path
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import time
import re 
import logging
from datetime import datetime
import csv
import pandas as pd
from utility import cleantoken

def getdoc2vecmodel():
    model = None
    model_exists = False
                
    if os.path.exists(os.path.join(os.path.realpath('..'), "inputdata")):
        for fname in os.listdir(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources")):
            if str(fname) == "doc2vec.model":
                # Existing model
                model_file = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), str(fname)))
                model = Doc2Vec.load(model_file)
                model_exists = True

    if not model_exists:
        # No existing model
        documents = pickle.load( open( "../inputdata/resources/documents.pickle", "rb" ) )
        model = Doc2Vec(vector_size=256, min_count=2, epochs=30)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.init_sims(replace=True)
        model_file = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), "doc2vec.model"))
        model.save(model_file)
    
    return model

def getword2vecmodel():
    model = None
    model_exists = False
                
    if os.path.exists(os.path.join(os.path.realpath('..'), "inputdata")):
        for fname in os.listdir(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources")):
            if str(fname) == "word2vec.model":
                # Existing model
                model_file = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), str(fname)))
                model = Word2Vec.load(model_file)
                model_exists = True

    if not model_exists:
        # No existing model
        documents = pickle.load( open( "../inputdata/resources/documents.pickle", "rb" ) )
        texts = []
        for doc in documents:
            texts.append(doc.words)
        model = Word2Vec(texts, size=256, window=5, min_count=2, workers=4)
        model.train(texts, total_examples=model.corpus_count,epochs=30)        
        model.init_sims(replace=True)
        model_file = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), "word2vec.model"))
        model.save(model_file)
        sim_matrix = WmdSimilarity(texts, model, num_best=21)
        with open('../inputdata/resources/word2vecsimilaritymatrix.pickle', 'wb') as f:
            pickle.dump(sim_matrix, f)

    return model
