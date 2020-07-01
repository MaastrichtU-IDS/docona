#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: trains / loads relevant models for applying to the similarity computation
# # TO DO: create a separate config file for training parameters and pass this into the getdoc2vecmodel() and getword2vecmodel() functions
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors
import gensim.models as g
from gensim.test.utils import get_tmpfile
from gensim.similarities import WmdSimilarity
import pickle
import os
import os.path

def getdoc2vecmodel(modelinputfilename,modeloutputfilename):
    model_exists = False
    # Check if there is an existing doc2vec model trained based on the input pretrained embeddings in modelinputfilename
    for fname in os.listdir(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources")):
        if str(fname) == modeloutputfilename:
            # Existing model
            model_file = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), str(fname)))
            model = Doc2Vec.load(model_file)
            model_exists = True

    if not model_exists:
        documents = pickle.load(open( "../inputdata/resources/documents.pickle", "rb" ))
        # Train doc2vec model by using pretrained googlenews word vectors to initialise words in the corpus
        vector_size = 300
        window_size = 5
        min_count = 2
        sampling_threshold = 1e-5
        negative_size = 5
        training_epochs = 20
        dm = 0
        hs = 0
        worker_count = 4
         # Load pretrained word embeddings
        pretrained_emb = os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), modelinputfilename)
        # Model output file
        fname = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), modeloutputfilename))
        # Train doc2vec model
        model = g.Doc2Vec(documents, vector_size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=hs, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, epochs=training_epochs)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.init_sims(replace=True)
        # Save model
        model.save(fname)

    return model

def getword2vecmodel(modelinputfilename,wmdsimilaritymatrixfilename,binaryfile):
    documents = pickle.load( open( "../inputdata/resources/documents.pickle", "rb" ) )
    fname = get_tmpfile(os.path.join(os.path.join(os.path.join(os.path.realpath('..'), "inputdata"), "resources"), modelinputfilename))
    model = KeyedVectors.load_word2vec_format(fname, binary=binaryfile)
    texts = []
    for doc in documents:
        texts.append(doc.words)
    sim_matrix = WmdSimilarity(texts, model, num_best=21)
    with open('../inputdata/resources/'+wmdsimilaritymatrixfilename+'.pickle', 'wb') as f:
        pickle.dump(sim_matrix, f)
    return model

