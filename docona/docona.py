#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Main client script - execute this script to run the pipeline, extend this script to add new models
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

print()
print("Document Citation and Content Analysis (DoConA)")
print("-----------------------------------------------")
print()
print("1. Preprocessing documents...", end = '', flush=True)
# Preprocessing full texts
from preprocessdocs import preprocess, preprocessingdone
if not preprocessingdone():
	preprocess()
print("Done!")
print()
print("2. Executing semantic measures")
print("a) Training corpus models...", end = '', flush=True)
# Train Doc2Vec and Word2Vec models on full texts of documents
from trainedmodels import getdoc2vecmodel,getword2vecmodel
doc2vecmodel = getdoc2vecmodel()
# word2vecmodel = getword2vecmodel()
print("Done!")
print("b) Similarity checks: corpus models...", end = '', flush=True)
print()
# # Trained models: check document similarity
from trainedmodelssimilarity import dosimilaritychecks
dosimilaritychecks("doc2vec",doc2vecmodel,"cosine")
# dosimilaritychecks("word2vec",word2vecmodel,"wmd")
print()
print("Done!")

# # --------------------------------------------- #
# # --- ADD CUSTOM PRETRAINED MODEL CODE HERE --- #
# # --------------------------------------------- #
# print("c) Adapting GoogleNews pretrained model...", end = '', flush=True)
# # # GoogleNews pretrained load / train
# from pretrainedmodels import getdoc2vecmodel,getword2vecmodel
# googledoc2vecmodel = getdoc2vecmodel("GoogleNews-vectors-negative300.bin","googlenewsdoc2vec.model")									# Only accepts gensim pretrained .model files 
# googleword2vecmodel = getword2vecmodel("GoogleNews-vectors-negative300.bin","googlenewssimilaritymatrix",binaryfile=True)				# Accepts binary or keyed vector format files
# print("Done!")
# print("d) Adapting Law2Vec pretrained model...", end = '', flush=True)
# # # Law2Vec pretrained load / train
# law2vecdoc2vecmodel = getdoc2vecmodel("Law2Vec.200d.txt","law2vecdoc2vec.model")														# Only accepts gensim pretrained .model files 
# law2vecword2vecmodel = getword2vecmodel("Law2Vec.200d.txt","law2vecsimilaritymatrix",binaryfile=False)									# Accepts binary or keyed vector format files
# print("Done!")
# print("e) Similarity checks: GoogleNews models...", end = '', flush=True)
# # # GoogleNews: check document similarity
# from pretrainedmodelssimilarity import dosimilaritychecks
# dosimilaritychecks("doc2vec", "googlenews", googledoc2vecmodel, "cosine")
# dosimilaritychecks("word2vec", "googlenews", googleword2vecmodel, "wmd")
# print("Done!")
# print("f) Similarity checks: Law2Vec models...", end = '', flush=True)
# # # Law2Vec: check document similarity
# dosimilaritychecks("doc2vec", "law2vec", law2vecdoc2vecmodel, "cosine")
# dosimilaritychecks("word2vec", "law2vec", law2vecword2vecmodel, "wmd")
# print("Done!")
# print()

print("3. Executing syntactic measures")
print("a) Training TFIDF and Ngram models...", end = '', flush=True)
# # TFIDF, Ngram models load / train
from syntacticmodels import gettfidfmodel,getngrammodel,dosyntacticsimilaritychecks
tfidfmodel = gettfidfmodel()
ngrammodel = getngrammodel(5)
print("Done!")
print("b) Similarity checks: TFIDF, Ngram, Jaccard...", end = '', flush=True)
# # TFIDF, Ngram, Jaccard: check document similarity
dosyntacticsimilaritychecks("tfidf",tfidfmodel)
dosyntacticsimilaritychecks("5gram",ngrammodel)
dosyntacticsimilaritychecks("jaccard",model=None)
print("Done!")
print()
print("4. Analysing results")
from analyseresults import analyse
analyse()
print("Done!")
print()
print("-- FINISHED --")
