#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: access to generated resources from the pipeline
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

import pickle
import pandas as pd

# Load resources
documentID_to_data = pickle.load( open( "../inputdata/resources/documentID_to_data.pickle", "rb" ) )
index_to_documentID = pickle.load( open( "../inputdata/resources/index_to_documentID.pickle", "rb" ) )
datafortraining = pickle.load( open( "../inputdata/resources/datafortraining.pickle", "rb" ) )
documentID_to_tokenized = pickle.load( open( "../inputdata/resources/documentID_to_tokenized.pickle", "rb" ) )
data_to_tokenized = pickle.load( open( "../inputdata/resources/data_to_tokenized.pickle", "rb" ) )
documentID_to_index = pickle.load( open( "../inputdata/resources/documentID_to_index.pickle", "rb" ) )
sim = pickle.load( open( "../inputdata/resources/word2vecsimilaritymatrix.pickle", "rb" ) )
citations = pd.read_csv('../inputdata/citations.csv')
sampledocuments = pd.read_csv("../inputdata/sample.csv", header = None)[0].values