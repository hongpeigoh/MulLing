import numpy as np
import os.path
import pickle
import math
from scipy import spatial
from collections import defaultdict
from dataclasses import dataclass
from pkgs.FastText import FastVector
from pkgs import LASER
from src import processing

@dataclass(unsafe_hash=True)
class MulLingVectors:
    def __init__(self, method=1, langs=['en', 'zh', 'ms', 'ta']):
        """
        Methods
        --------------------
        Method 1: BWE-Agg-Add
        Method 2: BWE-Agg-IDF
        Method 3: Pre-loaded document vectors (BWE-Agg-Add)
        Method 4: Pre-loaded document vectors (BWE-Agg-IDF)
        Method 5: LASER Sentence Embeddings

        Langs
        --------------------
        Implemented:
        - English (en)
        - Simplified Chinese (zh)
        - Bahasa Melayu (ms)
        - Tamil (ta)
        """
        methods = [1, 2, 3, 4, 5]
        assert method in methods, TypeError('Invalid method type. Use integer from 1 to 4 instead.')
        self.method = method
        self.langs = langs
        self.vecs = {}
        self.docs = {}
        self.docvecs = defaultdict(list)
        self.metadocvecs = defaultdict(list)
        self.idfs = {}
        self.kdtrees = {}
        self.lasers = {}
        self.means = {}

        if self.method==3 or self.method==4:
            paths = {
                'docvecs': 'pickle/docvecs2.pkl',
                'idfs': 'pickle/idfs2.pkl',
                'metadocvecs': 'pickle/metadocvecs.pkl',
            }
            print('Loading ' + ', '.join(list(paths.keys())) + ' from disk')
            self.docvecs = pickle.load( open(paths['docvecs'], 'rb'))
            self.idfs = pickle.load( open(paths['idfs'], 'rb'))
            self.metadocvecs = pickle.load( open(paths['metadocvecs'], 'rb'))

        if self.method==5:
            paths = {
                'metadocvecs': 'pickle/metadocvecs.pkl',
                'laser': 'pickle/laser.pkl'
            }
            print('Loading ' + ', '.join(list(paths.keys())) + ' from disk')
            self.lasers = pickle.load(open(paths['laser'], 'rb'))
            self.metadocvecs = pickle.load( open(paths['metadocvecs'], 'rb'))

        for lang in self.langs:
            # Load word vectors and document corpora
            self.load(lang)

            # Calculate document vectors using BWE-Agg-Add
            if self.method == 1 or self.method == 3:
                if lang in self.docvecs and self.method==3:
                    print('The {} document vectors are already loaded!'.format(lang))
                else:
                    self.calculate_docvecs(lang)
            
            # Calculate document vectors using BWE-Agg-IDF
            elif self.method == 2 or self.method == 4:
                if lang in self.docvecs and lang in self.idfs and self.method==4:
                    print('The {} document vectors and inverse document frequencies are already loaded!'.format(lang))
                else:
                    self.calculate2_docvecs(lang)

            # Calculate document vectors using LASER
            elif self.method == 5:
                if lang in self.lasers:
                    print('The {} LASER vectors are already loaded!'.format(lang))
                else:
                    self.calculate3_docvecs(lang)
            
            # Calculate document vectors using document meta-data (titles)
            if lang in self.metadocvecs:
                print('The {} meta-document vectors are already loaded!'.format(lang))
            else:
                self.calculate_metadocvecs(lang)

            #Set up KD-Trees for subsequent queries
            # print('Instancing {} KD-Tree'.format(lang))
            # self.kdtrees[lang] = spatial.KDTree(list(map(lambda x: x/np.linalg.norm(x), self.docvecs[lang])))
            # print('KD-tree loaded!')
        print('All vector dictionaries loaded!')
        
    # Load the selected language into the class 'MulLingVectors'
    def load(self, lang:str):
        if self.method != 5:
            # Load aligned word vectors
            if not os.path.isfile('vecs/nb-{}-dump.txt'.format(lang)):
                raise IOError('File vecs/nb-{}-dump.txt does not exist'.format(lang))
            else:
                print('Importing FastText Vectors for {}'.format(lang))
                self.vecs[lang] = FastVector(vector_file = 'vecs/nb-{}-dump.txt'.format(lang))
        
        # Load monolingual corpora
        if not os.path.isfile('articles/articles-{}.pkl'.format(lang)):
            raise IOError('File articles/articles-{}.pkl does not exist'.format(lang))
        else:
            print('Importing articles from articles/articles-{}.pkl'.format(lang))
            self.docs[lang] = pickle.load(open('articles/articles-{}.pkl'.format(lang), 'rb'))

    def calculate_docvecs(self, lang:str):
        # Calculate summation of document vectors by vector addition.
        print('Calculating document vectors')
        self.docvecs[lang] = list()
        
        for doc in self.docs[lang]:
            d_tokens = processing.tokenize(lang, doc[1])
            d_tokens_vecs = []
            for token in list(d_tokens):
                try:
                    d_tokens_vecs.append(self.vecs[lang][token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            self.docvecs[lang].append(sum(np.array(vec) for vec in d_tokens_vecs))

    def calculate2_docvecs(self, lang:str):
        # Calculate Inverse Document Frequency (IDF) on first pass
        print('Calculating IDFs')
        N = len(self.docs[lang])
        dfs = dict()
        for doc in self.docs[lang]:
            d_tokens = processing.tokenize(lang,doc[1])
            d_unique_tokens = list(set(d_tokens))
            for token in d_unique_tokens:
                if token in dfs:
                    dfs[token] += 1
                else:
                    dfs[token] = 1
        
        self.idfs[lang] = dict(zip(dfs, map(lambda x : (math.log(N/x)), dfs.values())))

        # Calculating BWE-Agg-IDF using weighted vector addition on second pass.
        print('Calculating document vectors')
        for doc in self.docs[lang]:
            d_tokens = processing.tokenize(lang, doc[1])
            d_tokens_count = dict(Counter(list(d_tokens)))
            d_tokens_vecs = []
            for token in d_tokens_count:
                try:
                    d_tokens_vecs.append(np.array(self.vecs[lang][token])*self.idfs[lang][token]*d_tokens_count[token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            vecs = sum(np.array(vec) for vec in d_tokens_vecs)
            if isinstance(vecs, (list, tuple, np.ndarray)):
                self.docvecs[lang].append(sum(np.array(vec) for vec in d_tokens_vecs))
            else:
                self.docs[lang].remove(doc)

    def calculate_metadocvecs(self, lang:str):
        # Calculate summation of meta-document vectors by vector addition.
        print('Calculating meta-document vectors')
        self.metadocvecs[lang] = list()
        
        for doc in self.docs[lang]:
            t_tokens = processing.tokenize(lang, doc[0])
            t_tokens_vecs = []
            for token in list(t_tokens):
                try:
                    t_tokens_vecs.append(self.vecs[lang][token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            self.metadocvecs[lang].append(sum(np.array(vec) for vec in t_tokens_vecs))

    def calculate3_docvecs(self, lang:str):
        # Calculate document vectors using LASER sentence embeddings
        print('Loading document vectors using LASER')
        self.lasers[lang] = list()

        for doc in self.docs[lang]:
            text = doc[1].replace('\n',' ')
            vec = LASER.get_vect(text)
            if isinstance( vec, (list,tuple,np.ndarray)):
                self.lasers[lang].append(vec)
            else:
                self.docs[lang].remove(doc)
