import numpy as np
import os.path
import pickle
import math
from scipy import spatial
from collections import defaultdict, Counter
from dataclasses import dataclass
from pkgs.FastText import FastVector
from pkgs import LASER
from src import get, processing, query

@dataclass(unsafe_hash=True)
class MulLingVectors:
    def __init__(self, method=1, langs=['en', 'zh', 'ms', 'ta'], kdtree=False):
        """
        Methods
        --------------------
        Method 1: BWE-Agg-Add
        Method 2: Pre-loaded document vectors (BWE-Agg-Add)
        Method 3: BWE-Agg-IDF
        Method 4: Pre-loaded document vectors (BWE-Agg-IDF)
        Method 5: LASER Sentence Embeddings
        Method 6: Pre-loaded LASER Sentence Embeddings
        Method 7: LASER Embeddings on Article Title
        Method 8: Pre-loaded LASER Sentence Embeddings on Article Title
        Method 10: Pre-loaded Multi-model (BWE-Agg-Add, BWE-Agg-IDF, LASER)
        Langs
        --------------------
        Implemented:
        - English (en)
        - Simplified Chinese (zh)
        - Bahasa Melayu (ms)
        - Tamil (ta)
        """
        methods = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        assert method in methods, TypeError('Invalid method type. Use integer from 1 to 6 instead.')
        self.method = method
        self.langs = langs
        self.vecs = {}
        self.docs = {}
        self.docvecs = {
            'baa': defaultdict(list),
            'bai': defaultdict(list),
            'meta': defaultdict(list),
            'lasers': defaultdict(list),
            'metalasers': defaultdict(list)
        }
        self.idfs = {}
        self.kdtrees = {}
        self.mean = {}

        if self.method %2==0:
            self.paths = {
                'docvecs_baa': 'pickle/docvecs_baa_new.pkl',
                'docvecs_bai': 'pickle/docvecs_new.pkl',
                'idfs': 'pickle/idfs_new.pkl',
                'metadocvecs': 'pickle/metadocvecs_new.pkl',
                'lasers': 'pickle/lasers_new.pkl',
                'metalasers' : 'pickle/metalaser.pkl'
            }
            
            if 'docvecs_baa' in self.paths:
                self.docvecs['baa'] = pickle.load( open(self.paths['docvecs_baa'], 'rb'))
                print('Loaded document vectors(BWE-Agg-Add)')
            if 'docvecs_bai' in self.paths:
                self.docvecs['bai'] = pickle.load( open(self.paths['docvecs_bai'], 'rb'))
                print('Loaded document vectors(BWE-Agg-IDF)')
            if 'metadocvecs' in self.paths:
                self.docvecs['meta'] = pickle.load( open(self.paths['metadocvecs'], 'rb'))
                print('Loaded document vectors for titles')
            if self.method == 6 or self.method == 10:
                self.docvecs['lasers'] = pickle.load( open(self.paths['lasers'], 'rb'))
                print('Loaded document vectors(LASER)')
            if self.method == 8 or self.method == 10:
                self.docvecs['metalasers'] = pickle.load( open(self.paths['metalasers'], 'rb'))
                print('Loaded document vectors for titles in LASER')
            print('Pre-made attributes are loaded')

        for lang in self.langs:
            # Load word vectors and document corpora
            self.load(lang)

            # Calculate document vectors using BWE-Agg-Add
            if self.method in [1,2,10]:
                if lang in self.docvecs['baa']:
                    print('The {} BWE-Agg-Add document vectors are already loaded!'.format(lang))
                else:
                    self.calculate_docvecs(lang)
                self.dv = self.docvecs['baa']
            
            # Calculate document vectors using BWE-Agg-IDF
            elif self.method in [3,4,10]:
                if lang in self.docvecs['bai'] and lang in self.idfs:
                    print('The {} BWE-Agg-IDF document vectors and inverse document frequencies are already loaded!'.format(lang))
                else:
                    self.calculate2_docvecs(lang)
                self.dv = self.docvecs['baa']

            # Calculate documents vectors using LASER
            elif self.method in [5,6,10]:
                if lang in self.docvecs['lasers']:
                    print('The {} LASER document vectors are already loaded'.format(lang))
                else:
                    self.calculate3_docvecs(lang)
            
            # Calculate document vectors using LASER on titles
            elif self.method in [7,8,10]:
                if lang in self.docvecs['metalasers']:
                    print('The {} LASER meta-document vectors are already loaded'.format(lang))
                else:
                    self.calculate4_docvecs(lang)
            
            # Calculate document vectors using document meta-data (titles)
            if lang in self.docvecs['meta']:
                print('The {} meta-document vectors are already loaded!'.format(lang))
            else:
                self.calculate_metadocvecs(lang)

            # if kdtree:
            #     #Set up KD-Trees for subsequent queries
            #     print('Instancing {} KD-Tree'.format(lang))
            #     try:
            #         self.kdtrees[lang] = spatial.KDTree(list(map(lambda x: x/np.linalg.norm(x), self.docvecs[lang])))
            #         print('KD-tree loaded!')
            #     except:
            #         print('KD-tree failed to load, continuing without KD-tree')
        print('All vector dictionaries loaded!')
        
    # Load the selected language into the class 'MulLingVectors'
    def load(self, lang:str):
        # Load aligned word vectors
        if not os.path.isfile('dump/{}/wordvecs.txt'.format(lang)):
            raise IOError('File dump/{}/wordvecs.txt does not exist'.format(lang))
        else:
            print('Importing FastText Vectors for {}'.format(lang))
            self.vecs[lang] = FastVector(vector_file = 'dump/{}/wordvecs.txt'.format(lang))
        
        # Load monolingual corpora
        if not os.path.isfile('dump/{}/articles.pkl'.format(lang)):
            raise IOError('File dump/{}/articles.pkl does not exist'.format(lang))
        else:
            print('Importing articles from dump/{}/articles.pkl'.format(lang))
            self.docs[lang] = pickle.load(open('dump/{}/articles.pkl'.format(lang), 'rb'))

    def calculate_docvecs(self, lang:str):
        # Calculate summation of document vectors by vector addition.
        print('Calculating document vectors')
        
        for doc in self.docs[lang]:
            d_tokens = processing.tokenize(lang, doc[1])
            d_tokens_vecs = []
            for token in list(d_tokens):
                try:
                    d_tokens_vecs.append(self.vecs[lang][token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            self.docvecs['baa'][lang].append(sum(np.array(vec) for vec in d_tokens_vecs))

    def calculate2_docvecs(self, lang:str, multi=False):

        if 'idfs' in self.paths:
            self.idfs = pickle.load( open(self.paths['idfs'], 'rb'))
            print('Loaded IDFs')
        else:
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
                self.docvecs['bai'][lang].append(sum(np.array(vec) for vec in d_tokens_vecs))
            else:
                self.docs[lang].remove(doc)

    def calculate_metadocvecs(self, lang:str):
        # Calculate summation of meta-document vectors by vector addition.
        print('Calculating meta-document vectors')
        self.docvecs['meta'][lang] = list()
        
        for doc in self.docs[lang]:
            t_tokens = processing.tokenize(lang, doc[0])
            t_tokens_vecs = []
            for token in list(t_tokens):
                try:
                    t_tokens_vecs.append(self.vecs[lang][token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            self.docvecs['meta'][lang].append(sum(np.array(vec) for vec in t_tokens_vecs))

    def calculate3_docvecs(self, lang:str):
        # Calculate document vectors using LASER Sentence Embeddings
        print('Loading document vectors using LASER')
        self.docvecs['lasers'][lang] = list()

        for doc in self.docs[lang]:
            text = doc[1].replace('\n', ' ')
            vec = LASER.get_vect(text, lang=lang)
            if isinstance(vec, (list, tuple, np.ndarray)):
                self.docvecs['lasers'][lang].append(vec)
            else:
                self.docs[lang].remove(doc)

    def calculate4_docvecs(self, lang:str):
        # Calculate document vectors using LASER Sentence Embeddings on article titles
        print('Loading document vectors using LASER-meta')
        self.docvecs['metalasers'][lang] = list()

        for doc in self.docs[lang]:
            text = doc[0].replace('\n', ' ')
            vec = LASER.get_vect(text, lang=lang if lang != 'zh' else 'en')
            if isinstance(vec, (list, tuple, np.ndarray)):
                self.docvecs['lasers'][lang].append(vec)
            else:
                self.docs[lang].remove(doc)

if __name__ == "__main__":
    from src import get, query, processing

    print('Hello World')
