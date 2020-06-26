#!/usr/bin/env python
# # Document Content and Citation Analysis (DoConA)
# # Module: GENERIC helper methods for the pipeline
# GNU AGPLv3 - https://choosealicense.com/licenses/agpl-3.0/
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Generic methods 
def stemSentence(sentence, porter):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def representsint(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def cleantoken(token):
    result = []
    token2 = token.replace(".", " ")
    token2 = token2.replace("\\", " ")
    token2 = token2.replace("/", " ")
    token2 = token2.replace(":", " ")
    token2 = token2.replace(";"," ")
    token2 = token2.replace("-"," ")
    token2 = token2.replace("*"," ")
    token2 = token2.replace("°"," ")
    tokens = token2.split(" ")
                            
    if (len(tokens) >= 1):
        for item in tokens:
            if representsint(item):                        # Is an integer 
                if (len(item) == 4):                       # Could be a year e.g. 1996
                    result.append(item)                    # Keep it
            else:
                if ((len(item) > 1) and (len(item) < 25)): # Remove URLs (> 25 characters)
                    result.append(item)

    return result