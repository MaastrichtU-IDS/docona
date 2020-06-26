#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: SPECIFIC helper methods for the pipeline
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/

from resources import index_to_documentID, citations

# Path to full texts of documents
textpath = "../inputdata/fulltexts/"

# Helper functions for document similarity checking
def convert_to_document_references(simmatrix):
    global index_to_documentID
    result = []
    for item in simmatrix:
        document_reference = index_to_documentID[item[0]] # convert to document reference
        similarity_value = item[1]
        result.append((document_reference,similarity_value))
    return result

def exists_citation_link_between(documentIDnumber1,documentIDnumber2):
    global citations
    relevantsource1 = citations[citations['source'] == documentIDnumber1]
    relevantsource2 = citations[citations['source'] == documentIDnumber2]
    if (documentIDnumber2 in relevantsource1['target'].tolist()) or (documentIDnumber1 in relevantsource2['target'].tolist()):
        return True
    return False