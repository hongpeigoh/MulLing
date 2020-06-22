from pyemd import emd, emd_with_flow
import numpy as np
from src import processing
from gensim.corpora.dictionary import Dictionary

'''
For full docs, refer to
https://github.com/RaRe-Technologies/gensim/blob/27bbb7015dc6bbe02e00bb1853e7952ac13e7fe0/gensim/models/deprecated/keyedvectors.py
for WMD and emd implementation.

This package is implemented to suit the word2vec format of the MulLing repository

Legend
-------------
doc1: Raw sentence corpora of document 1
doc2: Raw sentence corpora of document 2
lang1: Language of document 1
lang2: Language of document 2
vecs: Dictionary of word vectors, accessed by vecs[lang]
'''

def wmdsimilarity(doc1, doc2, lang1, lang2, vecs, with_flow=False):
    tok1 = list(processing.tokenize(lang1, doc1, include_stopwords=True))
    tok2 = list(processing.tokenize(lang2, doc2, include_stopwords=True))
    
    print(tok1, tok2)
    
    dictionary = Dictionary(documents=[tok1, tok2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(tok1)
    docset2 = set(tok2)
    
    print(dictionary, docset1, docset2)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = np.sqrt(np.sum((vecs[lang1][t1] - vecs[lang2][t2])**2))

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        print('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(tok1)
    d2 = nbow(tok2)

    # Compute WMD.
    if with_flow:
        emd = emd_with_flow(d1, d2, distance_matrix)
        return {
            'tokens': list(dictionary.values()),
            'pdf1':list(d1),
            'pdf2':list(d2),
            'wmd': emd[0],
            'flow': emd[1],
            'dist_matrix': distance_matrix.tolist()
        }
    else:
        return {
            'tokens':list(dictionary.values),
            'pdf1':list(d1),
            'pdf2':list(d2),
            'wmd':emd(d1, d2, distance_matrix),
            'dist_matrix': distance_matrix.tolist()
        }