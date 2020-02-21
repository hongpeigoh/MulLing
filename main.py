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
from annoy import AnnoyIndex

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
        Method 100: Pre-loaded Multi-model (BWE-Agg-Add, BWE-Agg-IDF, LASER)
        Langs
        --------------------
        Implemented:
        - English (en)
        - Simplified Chinese (zh)
        - Bahasa Melayu (ms)
        - Tamil (ta)
        """
        methods = [1, 2, 3, 4, 5, 6, 7, 8, 100]
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
                'docvecs_baa': 'dump/pickle/docvecs_baa_new.pkl',
                'docvecs_bai': 'dump/pickle/docvecs_new.pkl',
                'idfs': 'dump/pickle/idfs_new.pkl',
                'metadocvecs': 'dump/pickle/metadocvecs_new.pkl',
                'lasers': 'dump/pickle/lasers_new.pkl',
                'metalasers' : 'dump/pickle/metalasers_new.pkl'
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
            if self.method == 6 or self.method == 100:
                self.docvecs['lasers'] = pickle.load( open(self.paths['lasers'], 'rb'))
                print('Loaded document vectors(LASER)')
            if self.method == 8 or self.method == 100:
                self.docvecs['metalasers'] = pickle.load( open(self.paths['metalasers'], 'rb'))
                print('Loaded document vectors for titles in LASER')
            print('Pre-made attributes are loaded')

        for lang in self.langs:
            # Load word vectors and document corpora
            self.load(lang)

            # Calculate document vectors using BWE-Agg-Add
            if self.method in [1,2,100]:
                if lang in self.docvecs['baa']:
                    print('The {} BWE-Agg-Add document vectors are already loaded!'.format(lang))
                else:
                    self.calculate_docvecs(lang)
                self.dv = self.docvecs['baa']
            
            # Calculate document vectors using BWE-Agg-IDF
            elif self.method in [3,4,100]:
                if lang in self.docvecs['bai'] and lang in self.idfs:
                    print('The {} BWE-Agg-IDF document vectors and inverse document frequencies are already loaded!'.format(lang))
                else:
                    self.calculate2_docvecs(lang)
                self.dv = self.docvecs['baa']

            # Calculate documents vectors using LASER
            elif self.method in [5,6,100]:
                if lang in self.docvecs['lasers']:
                    print('The {} LASER document vectors are already loaded'.format(lang))
                else:
                    self.calculate3_docvecs(lang)
            
            # Calculate document vectors using LASER on titles
            elif self.method in [7,8,100]:
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
            if lang!='ta':
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

        bigtext = []
        for doc in self.docs[lang]:
            text = doc[1].replace('\n', ' ')
            bigtext.append(text)
        bigtext = '\n'.join(bigtext)
        bigvects = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
        
        for vec in bigvects:
            self.docvecs['metalasers'][lang].append(vec)

        '''
        ----- Old Code -----
        for doc in self.docs[lang]:
            text = doc[1].replace('\n', ' ')
            vec = LASER.get_vect(text, lang=lang)
            if isinstance(vec, (list, tuple, np.ndarray)):
                self.docvecs['lasers'][lang].append(vec)
            else:
                self.docs[lang].remove(doc)
        '''

    def calculate4_docvecs(self, lang:str):
        # Calculate document vectors using LASER Sentence Embeddings on article titles
        print('Loading document vectors using LASER-meta')
        self.docvecs['metalasers'][lang] = list()

        bigtext = []
        for doc in self.docs[lang]:
            text = doc[0].replace('\n', ' ')
            bigtext.append(text)
        bigtext = '\n'.join(bigtext)
        bigvects = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
        
        for vec in bigvects:
            self.docvecs['metalasers'][lang].append(vec)
        '''
        ----- Old Code -----
        for doc in self.docs[lang]:
            text = doc[0].replace('\n', ' ')
            vec = LASER.get_vect(text, lang=lang if lang != 'zh' else 'en')
            if isinstance(vec, (list, tuple, np.ndarray)):
                self.docvecs['metalasers'][lang].append(vec)
            else:
                self.docs[lang].remove(doc)
        '''



# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #




@dataclass(unsafe_hash=True)
class MulLingVectorsAnnoy:
    def __init__(self, methods = ['baa','bai','meta','laser','metalaser'], paths = ['1', '1', '1', '1', '1'], langs=['en', 'zh', 'ms', 'ta']):
        """
        Methods
        --------------------
        Method 1: BWE-Agg-Add (baa)
        Method 2: BWE-Agg-IDF (bai)
        Method 3: LASER Sentence Embeddings (laser)
        Method 4: LASER Embeddings on Article Title (metalaser)
        Method 5: BWE-Agg-Add on Article Title (meta)
        --------------------
        Implemented:
        - English (en)
        - Simplified Chinese (zh)
        - Bahasa Melayu (ms)
        - Tamil (ta)
        """
        self.langs = langs
        self.vecs = {}
        self.docs = {}
        self.idfs = {}
        self.docvecs = {}
        self.paths = {}
        self.mean = {}

        # Check if number of methods and number of paths are equal
        assert len(methods) == len(paths), "Number of methods isn't equal to number of paths!"

        # Loads all possible indexes
        for method in methods:
            assert method in ['baa','bai','meta','laser','metalaser'], "%s is not a valid model" % method
            self.docvecs[method] = defaultdict(list)
        for index in range(len(methods)):
            self.paths[methods[index]] = paths[index]  

        # Indexing all necessary attributes
        for lang in self.langs:
            # Load word vectors and document corpora
            self.load(lang)
            print('Language dependencies loaded.')
            for model in self.paths:
                # Set Dimension
                dim = 1024 if model == 'laser' or model =='metalaser' else 300

                # Load Annoy Index
                self.docvecs[model][lang] = AnnoyIndex(dim, 'angular')
                if os.path.isfile('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model])):
                    self.docvecs[model][lang].load('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model]))
                    print('Loading document vectors from dump/annoy/%s/%s%s.ann.' % (lang, model, self.paths[model]))
                else:
                    print('Saved annoy index not found, calculating from scratch.')
                    self.calculate(model, lang)

        # Clear memory
        self.idfs = {}
        print('All models are loaded.')
        
    # Load the selected language into the class
    def load(self, lang:str):
        # Load aligned word vectors
        if not os.path.isfile('dump/{}/wordvecs.txt'.format(lang)):
            raise IOError('File dump/{}/wordvecs.txt does not exist'.format(lang))
        else:
            print('Importing FastText Vectors for {}'.format(lang))
            if lang != 'ta':
                self.vecs[lang] = FastVector(vector_file = 'dump/{}/wordvecs.txt'.format(lang))

        # Load monolingual corpora
        if lang in self.docs:
            pass
        elif not os.path.isfile('dump/{}/articles.pkl'.format(lang)):
            raise IOError('File dump/{}/articles.pkl does not exist'.format(lang))
        else:
            print('Importing articles from dump/{}/articles.pkl'.format(lang))
            self.docs[lang] = pickle.load(open('dump/{}/articles.pkl'.format(lang), 'rb'))

    def calculate(self, model:str, lang:str):
        if model == 'baa':
            # Calculate summation of document vectors by vector addition.
            print('Calculating document vectors using Bilingual Word Embeddings (Vector Addition)')
            
            for index, doc in enumerate(self.docs[lang]):
                d_tokens = processing.tokenize(lang, doc[1])
                d_tokens_vecs = []
                for token in list(d_tokens):
                    try:
                        d_tokens_vecs.append(self.vecs[lang][token])
                    except:
                        pass
                        #raise KeyError('{} not in {} dictionary'.format(token, lang))
                vecs = sum(np.array(vec) for vec in d_tokens_vecs)
                if isinstance(vecs, (list, tuple, np.ndarray)):
                    self.docvecs['baa'][lang].add_item(index, vecs)



        if model =='bai':
            if os.path.isfile('dump/pickle/idfs_new.pkl'):
                # Load IDFs
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
            print('Calculating document vectors using Bilingual Word Embeddings (TF-IDF)')
            for index, doc in enumerate(self.docs[lang]):
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
                    self.docvecs['bai'][lang].add_item(index, vecs)



        if model == 'meta':
            # Calculate summation of meta-document vectors by vector addition.
            print('Calculating meta-document vectors')            
            for index, doc in enumerate(self.docs[lang]):
                t_tokens = processing.tokenize(lang, doc[0])
                t_tokens_vecs = []
                for token in list(t_tokens):
                    try:
                        t_tokens_vecs.append(self.vecs[lang][token])
                    except:
                        pass
                        #raise KeyError('{} not in {} dictionary'.format(token, lang))
                vecs = sum(np.array(vec) for vec in t_tokens_vecs)
                if isinstance(vecs, (list, tuple, np.ndarray)):
                    self.docvecs['meta'][lang].add_item(index, vecs)



        if model == 'laser':
            # Calculate document vectors using LASER Sentence Embeddings
            print('Loading document vectors using LASER')
            bigtext = []
            for doc in self.docs[lang]:
                text = doc[1].replace('\n', ' ')
                bigtext.append(text)
            bigtext = '\n'.join(bigtext)
            bigvects = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
            
            for index, vecs in enumerate(bigvects):
                self.docvecs['lasers'][lang].add_item(index, vecs)



        if model == 'metalaser':
            # Calculate document vectors using LASER Sentence Embeddings on article titles
            print('Loading document vectors using LASER-meta')
            bigtext = []
            for doc in self.docs[lang]:
                text = doc[0].replace('\n', ' ')
                bigtext.append(text)
            bigtext = '\n'.join(bigtext)
            bigvects = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
            
            for index, vecs in enumerate(bigvects):
                self.docvecs['metalasers'][lang].add_item(index, vecs)

        self.docvecs[model][lang].build(math.floor(math.log(len(self.docs[lang]))))
        self.docvecs[model][lang].save('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model]))

if __name__ == "__main__":
    from src import get, query, processing

    print('Hello World')
