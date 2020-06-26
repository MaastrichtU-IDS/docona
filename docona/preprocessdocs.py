#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: preprocess documents
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
import os
import os.path
from os import path
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,  word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import time
import logging
from datetime import datetime
import csv
import pandas as pd
from utility import cleantoken, stemSentence

def preprocess():
    textpath = "../inputdata/fulltexts/"
    index_to_documentID = {}
    documentID_to_index = {}
    data_to_documentID = {}
    documentID_to_data = {}
    documentID_to_tokenized = {}
    data_to_tokenized = {}

    tokenize = lambda doc: doc.lower().split(" ")

    # Import stopwords 
    stopwords_full = []
    if os.path.exists(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), "stopwords.csv")):
        stopwords_full = pd.read_csv("../inputdata/resources/stopwords.csv", header = None)[0].values          
    # with open(stopwordsfile, "rb") as f:
    #     tmp = pickle.load(f)
    #     stopwords_full.extend(list(tmp))
    #     stopwords_full.extend(stopwords.words('english'))
            
    # stopwords_full = list(set(stopwords_full))

    porter=PorterStemmer()

    # Import files and define mapping between document IDS and full texts   
    starttime = time.time()
    files = []
    datafortraining = []
    word_list = set()
    index = 0
    for r, d, f in os.walk(textpath):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
                documentIDnum = os.path.basename(file)
                with open (textpath+file, "r", encoding="utf-8") as myfile:
                    data = myfile.read().replace('\n', ' ')
                    data_word_tokens = word_tokenize(data)
                    filtered_document = [w for w in data_word_tokens if not w in stopwords_full] 
                    filtered_document = []
                    data = ''
                    for w in data_word_tokens: 
                        if w.lower() not in stopwords_full:
                            tokens = cleantoken(w.lower())
                            for item in tokens:
                                if len(wn.synsets(item)) > 0:
                                    filtered_document.append(item)
                                    data = data + item + ' '
                                
                    for word in filtered_document:
                        word_list.add(word)
                            
                    sent_text = nltk.sent_tokenize(data)
                    data = ""
                    for sentence in sent_text:
                        data += stemSentence(sentence, porter)

                    datafortraining.append(data)
                    documentID_to_index[file.replace(".txt","")] = index 
                    data_to_documentID[data] = file.replace(".txt","")
                    documentID_to_data[file.replace(".txt","")] = data      # Use this for TFIDF
                    index_to_documentID[index] = file.replace(".txt","")
                    tknzd = tokenize(data)
                    data_to_tokenized[data] = tknzd
                    documentID_to_tokenized[file.replace(".txt","")] = tknzd
                    index += 1
    
    with open('../inputdata/resources/documentID_to_tokenized.pickle', 'wb') as f:
        pickle.dump(documentID_to_tokenized, f)

    with open('../inputdata/resources/data_to_tokenized.pickle', 'wb') as f:
        pickle.dump(data_to_tokenized, f)

    with open('../inputdata/resources/datafortraining.pickle', 'wb') as f:
        pickle.dump(datafortraining, f)

    endtime = time.time()

    documents = [TaggedDocument(text, [i]) for i, text in enumerate(datafortraining)]
    with open('../inputdata/resources/documents.pickle', 'wb') as f:
        pickle.dump(documents, f)

    wl = list(word_list)
    with open('../outputdata/uniquewords.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for w in wl:
            writer.writerow([w])

    processedtextpath = "../inputdata/processedtexts/"

    if not path.exists(processedtextpath):
        try:
            os.mkdir(processedtextpath)
        except OSError:
            print ("Creation of the directory %s failed" % processedtextpath)
        #else:
            #print ("Successfully created the directory %s " % processedtextpath)

    with open('../inputdata/resources/index_to_documentID.pickle', 'wb') as f:
        pickle.dump(index_to_documentID, f)

    with open('../inputdata/resources/documentID_to_data.pickle', 'wb') as f:
        pickle.dump(documentID_to_data, f)

    with open('../inputdata/resources/documentID_to_index.pickle', 'wb') as f:
        pickle.dump(documentID_to_index, f)

    for key, val in documentID_to_data.items():
        file_tmp = open(processedtextpath + key + ".txt","w",encoding='utf-8',newline='') 
        file_tmp.write(val) 
        file_tmp.close()
