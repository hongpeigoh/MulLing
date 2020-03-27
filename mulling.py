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
            assert method in ['baa','bai','meta','laser','metalaser','senlaser','senbai'], "%s is not a valid model" % method
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
                if 'sen' in model:
                    if os.path.isfile('dump/%s/s2d%s.pkl' % (lang, model)) and os.path.isfile('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model])):
                        self.s2d[model][lang] = pickle.load(open('dump/%s/s2d%s.pkl'%(lang, model),'rb'))
                        print('Loading sentence to document index from dump/%s/s2d%s.pkl'%(lang, model))
                        self.docvecs[model][lang].load('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model]))
                        print('Loading document vectors from dump/annoy/%s/%s%s.ann.' % (lang, model, self.paths[model]))
                    else:
                        print('Saved annoy index not found, calculating from scratch.')
                        self.calculate(model, lang)
                else:
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
            #if lang != 'ta':
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
                self.idfs = pickle.load( open('dump/pickle/idfs_new.pkl', 'rb'))
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
            print('Calculating sentence vectors using Bilingual Word Embeddings (TF-IDF)')
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
            print('Calculating document vectors using LASER')
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
            print('Calculating document vectors using LASER-meta')
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
            print('Calculating document vectors using LASER on individual sentences')
            self.s2d[model][lang] = []
            count = 0
            
            for index, doc in enumerate(self.docs[lang]):
                if lang == 'zh':
                    sens = '\n'.join(list(processing.zh_sent_tokenize(doc[1])))
                else:
                    sens = '\n'.join(sent_tokenize(doc[1]))
                vecs = LASER.get_vect(sens, lang=lang if lang == 'ta' else 'en')
                for vec in vecs:
                    self.docvecs[model][lang].add_item(count, vec)
                    self.s2d[model][lang].append(index)
                    count += 1

            # Saving sentence to document index
            pickle.dump(self.s2d[model][lang], open('dump/%s/s2dsenlaser.pkl'%lang,'wb'))



        # Tends to crash memory when the corpora size is too large
        # if model == 'senlaser':
        #     # Calculate document vectors using LASER Sentence Embeddings on individual sentences
        #     print('Calculating document vectors using LASER on individual sentences')
        #     self.s2d[model][lang] = []
        #     bigtext = []
            
        #     for index, doc in enumerate(self.docs[lang]):
        #         sens = sent_tokenize(doc[1].replace('\n', ' '))
        #         bigtext += sens
        #         for _ in sens:
        #             self.s2d[model][lang].append(index)
        #     bigtext= '\n'.join(bigtext)
        #     print(lang, self.s2d[model][lang][-1])
        #     bigvecs = LASER.get_vect(bigtext, lang=lang if lang == 'ta' else 'en')

        #     for index, vec in enumerate(bigvecs):
        #         self.docvecs[model][lang].add_item(index, vec)        

        #     # Saving sentence to document index
        #     pickle.dump(self.s2d[model][lang], open('dump/%s/s2dsenlaser.pkl' % lang,'wb'))




        if model == 'senbai':
            # Calculate sentence vectors using TF-IDF
            print('Calculating document vectors using TF-IDF on individual sentences')
            self.s2d[model][lang] = []
            count = 0

            if self.idfs:
                print('IDFs already loaded')
            elif os.path.isfile('dump/pickle/idfs_new.pkl'):
                # Load IDFs
                self.idfs = pickle.load( open('dump/pickle/idfs2_new.pkl', 'rb'))
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
                sens = sent_tokenize(doc[1]) if lang!='zh' else list(processing.zh_sent_tokenize(doc[1]))
                for sentence in sens:
                    s_tokens = processing.tokenize(lang, sentence)
                    s_tokens_count = dict(Counter(list(s_tokens)))
                    s_tokens_vecs = []
                    for token in s_tokens_count:
                        try:
                            s_tokens_vecs.append(np.array(self.vecs[lang][token])*self.idfs[lang][token]*s_tokens_count[token])
                        except:
                            pass
                            #raise KeyError('{} not in {} dictionary'.format(token, lang))
                    vecs = sum(np.array(vec) for vec in s_tokens_vecs)
                    if isinstance(vecs, (list, tuple, np.ndarray)):
                        self.docvecs[model][lang].add_item(count, vecs)
                    self.s2d[model][lang].append(index)
                    count += 1
            
            # Saving sentence to document index
            pickle.dump(self.s2d[model][lang], open('dump/%s/s2dsenbai.pkl'%lang ,'wb'))


        self.docvecs[model][lang].build(math.floor(math.log(len(self.docs[lang]))))
        self.docvecs[model][lang].save('dump/annoy/%s/%s%s.ann' % (lang, model, self.paths[model]))
        print('Annoy Index for %s model and %s lang saved and loaded!' % (model, lang))