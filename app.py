import numpy as np
import os.path
import pickle
import math
from nltk.tokenize import sent_tokenize
from scipy import spatial
from collections import defaultdict, Counter
from dataclasses import dataclass
from pkgs.FastText import FastVector
from pkgs import LASER
from src import get, processing, query
from annoy import AnnoyIndex

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
app = Flask(__name__)
cors = CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['CORS_HEADERS'] = 'Content-Type'

@dataclass(unsafe_hash=True)
class MulLingVectorsAnnoy:
    def __init__(self, methods = ['baa','bai','meta','laser','metalaser','senlaser'], paths = ['1', '1', '1', '1', '1','1'], langs=['en', 'zh', 'ms', 'ta']):
        """
        Methods
        --------------------
        Method 1: BWE-Agg-Add (baa)
        Method 2: BWE-Agg-IDF (bai)
        Method 3: LASER Sentence Embeddings by Document (laser)
        Method 4: LASER Embeddings on Article Title (metalaser)
        Method 5: LASER Sentence Embeddings by Sentences (senlaser)
        Method 6: BWE-Agg-Add on Article Title (meta)
        --------------------
        Implemented Languages:
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
        self.s2d = {}
        self.paths = {}
        self.mean = {}

        # Check if number of methods and number of paths are equal
        assert len(methods) == len(paths), "Number of methods isn't equal to number of paths!"

        # Loads all possible indexes
        for method in methods:
            assert method in ['baa','bai','meta','laser','metalaser','senlaser'], "%s is not a valid model" % method
            self.docvecs[method] = defaultdict(list)
            self.s2d[method] = defaultdict(list)
        for index in range(len(methods)):
            self.paths[methods[index]] = paths[index]  

        # Indexing all necessary attributes
        for lang in self.langs:
            # Load word vectors and document corpora
            self.load(lang)
            print('Language dependencies loaded.')
            for model in self.paths:
                # Set Dimension
                dim = 1024 if 'laser' in model else 300

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
                    self.docvecs[model][lang].add_item(index, vecs)



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
                    self.docvecs[model][lang].add_item(index, vecs)



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
                    self.docvecs[model][lang].add_item(index, vecs)



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
                self.docvecs[model][lang].add_item(index, vecs)



        if model == 'metalaser':
            # Calculate document vectors using LASER Sentence Embeddings on article titles
            print('Loading document vectors using LASER-meta')
            bigtext = []
            for doc in self.docs[lang]:
                text = doc[0].replace('\n', ' ')
                bigtext.append(text)
            bigtext = '\n'.join(bigtext)
            bigvecs = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
            
            for index, vec in enumerate(bigvecs):
                self.docvecs[model][lang].add_item(index, vec)



        if model == 'senlaser':
            # Calculate document vectors using LASER Sentence Embeddings on individual sentences
            print('Loading document vectors using LASER on individual sentences')
            self.s2d[model][lang] = []
            count = 0

            for index, doc in enumerate(self.docs[lang]):
                sens = '\n'.join(sent_tokenize(doc[1]))
                vecs = LASER.get_vect(sens, lang=lang if lang == 'ta' else 'en')
                for vec in vecs:
                    self.docvecs[model][lang].add_item(count, vec)
                    self.s2d[model][lang].append(index)
                    count += 1

        self.docvecs[model][lang].build(math.floor(math.log(len(self.docs[lang]))))
        self.docvecs[model][lang].save('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model]))

            

@app.route('/')
def approot():
    return "<strong>Hello World!</strong><br/><p>The loaded languages are: %</p> " % (','.join(langs))

@app.route('/query_mono')
@cross_origin()
def appquery():
    q = str(request.args.get('q'))
    model = str(request.args.get('model'))
    lang = str(request.args.get('lang'))
    k = int(request.args.get('k'))
    print(
        '    Query:          %s\n' %q,
        '   Model:          %s\n' %model,
        '   Language:       %s\n' %lang,
        '   No. of Results: %i' %k)
    try:
        results = query.monolingual_annoy_query(app_object, q, model, lang, k)
    except:
        print(q, model, lang, k)

    return jsonify(
        allresults= get.jsonall(app_object, results),
        scoredresults= get.jsonscoredresults(app_object, results),
        results= get.jsonresults(app_object, results)
    )

@app.route('/query_multi')
@cross_origin()
def multiappquery():
    q = str(request.args.get('q'))
    model = str(request.args.get('model'))
    lang = str(request.args.get('lang'))
    k = int(request.args.get('k'))
    normalize = bool(request.args.get('normalize'))
    print(
        '    Query:           %s\n' %q,
        '   Model:           %s\n' %model,
        '   Language:        %s\n' %lang,
        '   No. of Results : %i\n' %k,
        '   Normalize Top L: %s' %str(normalize))

    if lang=='null':
        results = query.mulling_annoy_query(app_object, q, model, k, L= math.ceil(k/4), normalize_top_L=normalize, multilingual= True)
    else:
        results = query.mulling_annoy_query(app_object, q, model, k, L= math.ceil(k/4), normalize_top_L= normalize, multilingual= False, lang_= lang)

    return jsonify(
        allresults= get.jsonall(app_object, results),
        scoredresults= get.jsonscoredresults(app_object, results),
        results= get.jsonresults(app_object, results)
    )


if __name__ == "__main__":
    methods = ['baa','bai','meta','laser','metalaser']
    paths = [1, 1, 1, 1, 1]
    langs = ['en','zh','ms','ta']

    app_object = MulLingVectorsAnnoy(methods=methods, paths=paths, langs=langs)
    app.run(port=5050, host='0.0.0.0')
