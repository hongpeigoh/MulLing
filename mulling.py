import numpy as np
import os.path
import pickle
import math
from nltk.tokenize import sent_tokenize
from collections import defaultdict, Counter
from dataclasses import dataclass
from pkgs.FastText import FastVector
from pkgs import LASER
from src import get, processing, query
from annoy import AnnoyIndex

@dataclass(unsafe_hash=True)
class MulLingVectorsAnnoy:
    def __init__(
            self,
            models = ['baa', 'bai', 'meta', 'laser', 'metalaser', 'senlaser', 'senbai'],
            path   = './dump',
            langs  = ['en', 'zh', 'ms', 'ta'],
            wordvecs_as_annoy = False):
        """
        Initialises dataclass and models.
        --------------------
        Models: list(str)
        Model 1: BWE-Agg-Add on Article Text(baa)
        Model 2: BWE-Agg-IDF on Article Text (bai)
        Model 3: LASER Sentence Embeddings on Article Text (laser)
        Model 4: LASER Embeddings on Article Title (metalaser)
        Model 5: LASER Sentence Embeddings on Article Sentences (senlaser)
        Model 6: BWE-Agg-Add on Article Title (meta)
        Model 7: BWE-Agg-IDF on Article Sentences
        --------------------
        Path: str
        Specify the path of the data file. Default: './dump'
        --------------------
        Implemented Languages: list(str)
        - English (en)
        - Simplified Chinese (zh)
        - Bahasa Melayu (ms)
        - Tamil (ta)
        --------------------
        Wordvecs as Annoy: bool
        Specify whether to load word vectors as Annoy Index. Default: False.
        Typically true for models involving clustering, such as the flask version.
        """
        self.path = path
        self.langs = langs
        self.vecs = {}
        self.docs = {}
        self.idfs = {}
        self.docvecs = {}
        self.s2d = {}
        self.mean = {}

        # Loads all possible indexes
        for model in models:
            assert model in ['baa','bai','meta','laser','metalaser','senlaser','senbai'], "%s is not a valid model" % model
            self.docvecs[model] = defaultdict(list)
            self.s2d[model] = defaultdict(list)

        for lang in self.langs:
            # Load word vectors and document corpora
            self.load(lang, wordvecs_as_annoy=wordvecs_as_annoy)
            print('Language dependencies loaded.')

            for model in models:
                dim = 1024 if 'laser' in model else 300

                # Load Annoy Index
                self.docvecs[model][lang] = AnnoyIndex(dim, 'angular')
                if os.path.isfile('%s/%s/%s.ann' % (self.path, lang, model)):
                    if 'sen' in model and os.path.isfile('%s/%s/s2d%s.pkl' % (self.path, lang, model)):
                        self.s2d[model][lang] = pickle.load( open('%s/%s/s2d%s.pkl' % (self.path, lang, model),'rb'))
                        print('Loading sentence to document index from %s/%s/s2d%s.pkl.' % (self.path, lang, model))
                    self.docvecs[model][lang].load('%s/%s/%s.ann' % (self.path, lang, model))
                    print('Loading document vectors from %s/%s/%s.ann.' % (self.path, lang, model))

                # Calculate Annoy Index
                else:
                    print('Saved annoy index not found, calculating from scratch.')
                    self.calculate(model, lang, (self.path, lang, model))

        # Clear memory
        #self.idfs = {}

        print('All models are loaded.')
        
    def load(self, lang, wordvecs_as_annoy=False):
        """
        Loads necessary prerequisites for specific language.
        --------------------
        Language (to load): str
        - English (en)
        - Simplified Chinese (zh)
        - Bahasa Melayu (ms)
        - Tamil (ta)
        --------------------
        Wordvecs as Annoy: bool
        Specify whether to load word vectors as Annoy Index. Default: False.
        Typically true for models involving clustering, such as the flask version.
        """
        # Loads aligned word vectors
        if not os.path.isfile('%s/%s/wordvecs.txt' % (self.path, lang)):
            raise IOError('File %s/%s/wordvecs.txt does not exist' % (self.path, lang))
        else:
            print('Importing FastText Vectors for %s' % lang)
            self.vecs[lang] = FastVector(vector_file = '%s/%s/wordvecs.txt' % (self.path, lang))

        # Loads optional wordvecs Annoy Index and IDF pickle
        if wordvecs_as_annoy:
            self.vecs['annoy/' + lang] = AnnoyIndex(300, 'angular')
            if os.path.isfile('%s/%s/wordvecs.ann' % (self.path, lang)):
                # Loads wordvecs as .ann if it exists
                self.vecs['annoy/' + lang].load('%s/%s/wordvecs.ann' % (self.path, lang))
                print('Loading word vectors from %s/%s/wordvecs.ann. If you are not using our flask version, consider deloading this module' % (self.path, lang))

            else:
                # Re-saves wordvecs as .ann
                for index, word in enumerate(self.vecs[lang].id2word):
                    self.vecs['annoy/' + lang].add_item(index, self.vecs[lang][word])
                self.vecs['annoy/' + lang].build(math.floor(math.log(len(self.vecs[lang].id2word))))
                self.vecs['annoy/' + lang].save('%s/%s/wordvecs.ann' % (self.path, lang))
                print('Annoy Index is saved and loaded in %s for %s lang wordvecs' % (self.path, lang))

            if os.path.isfile('%s/pickle/idfs.pkl' % self.path):
                # Loads IDFs
                self.idfs = pickle.load( open('%s/pickle/idfs.pkl' % self.path, 'rb'))
                for lang_ in self.idfs:
                    self.idfs[lang_] = defaultdict(lambda: float('inf'), self.idfs[lang_])
                print('Loaded IDFs')
                
            else:
                print('IDFs need to be calculated!')


        # Load monolingual corpora
        if os.path.isfile('%s/%s/articles.pkl' % (self.path, lang)):
            print('Importing articles from %s/%s/articles.pkl.' % (self.path, lang))
            self.docs[lang] = pickle.load(open('%s/%s/articles.pkl' % (self.path, lang), 'rb'))
        else:
            raise IOError('File %s/%s/articles.pkl does not exist.' % (self.path, lang))

    def calculate(self, model, lang, directory):
        """
        Calculates document vectors/sentence vectors for the corpora given the model name.
        --------------------
        Models: list(str)
        Model 1: BWE-Agg-Add on Article Text(baa)
        Model 2: BWE-Agg-IDF on Article Text (bai)
        Model 3: LASER Sentence Embeddings on Article Text (laser)
        Model 4: LASER Embeddings on Article Title (metalaser)
        Model 5: LASER Sentence Embeddings on Article Sentences (senlaser)
        Model 6: BWE-Agg-Add on Article Title (meta)
        Model 7: BWE-Agg-IDF on Article Sentences
        """
        if model == 'baa':
            print('Calculating document vectors using Bilingual Word Embeddings (Vector Addition) on Article Text.')
            
            for index, doc in enumerate(self.docs[lang]):
                d_tokens = processing.tokenize(lang, doc[1])
                d_tokens_vecs = []
                for token in list(d_tokens):
                    try:
                        d_tokens_vecs.append(self.vecs[lang][token])
                    except:
                        pass
                vecs = sum(np.array(vec) for vec in d_tokens_vecs)
                if isinstance(vecs, (list, tuple, np.ndarray)):
                    self.docvecs[model][lang].add_item(index, vecs)

        if model =='bai':
            # First pass
            if os.path.isfile('%s/pickle/idfs.pkl' % self.path):
                self.idfs = pickle.load( open('%s/pickle/idfs.pkl' % self.path, 'rb'))
                print('Loaded IDFs')
            else:
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
                pickle.dump( self.idfs, open('%s/pickle/idfs.pkl' % self.path, 'wb'))
                print('IDFs saved at %s/pickle/idfs.pkl' % self.path)

            # Second pass
            print('Calculating document vectors using Bilingual Word Embeddings (TF-IDF) on Article Text.')
            for index, doc in enumerate(self.docs[lang]):
                d_tokens = processing.tokenize(lang, doc[1])
                d_tokens_count = dict(Counter(list(d_tokens)))
                d_tokens_vecs = []
                for token in d_tokens_count:
                    try:
                        d_tokens_vecs.append(np.array(self.vecs[lang][token])*self.idfs[lang][token]*d_tokens_count[token])
                    except:
                        pass
                vecs = sum(np.array(vec) for vec in d_tokens_vecs)
                if isinstance(vecs, (list, tuple, np.ndarray)):
                    self.docvecs[model][lang].add_item(index, vecs)

        if model == 'meta':
            print('Calculating document vectors using Bilingual Word Embeddings (Vector Addition) on Article Title.')            
            for index, doc in enumerate(self.docs[lang]):
                t_tokens = processing.tokenize(lang, doc[0])
                t_tokens_vecs = []
                for token in list(t_tokens):
                    try:
                        t_tokens_vecs.append(self.vecs[lang][token])
                    except:
                        pass
                vecs = sum(np.array(vec) for vec in t_tokens_vecs)
                if isinstance(vecs, (list, tuple, np.ndarray)):
                    self.docvecs[model][lang].add_item(index, vecs)

        if model == 'laser':
            print('Calculating document vectors using LASER Sentence Embeddings  on Article Text.')
            bigtext = []
            for doc in self.docs[lang]:
                text = doc[1].replace('\n', ' ')
                bigtext.append(text)
            bigtext = '\n'.join(bigtext)
            bigvects = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
            
            for index, vecs in enumerate(bigvects):
                self.docvecs[model][lang].add_item(index, vecs)

        if model == 'metalaser':
            print('Calculating document vectors using LASER Sentence Embeddings on Article Titles.')
            bigtext = []
            for doc in self.docs[lang]:
                text = doc[0].replace('\n', ' ')
                bigtext.append(text)
            bigtext = '\n'.join(bigtext)
            bigvecs = LASER.get_vect(bigtext, lang=lang if lang != 'zh' else 'en')
            
            for index, vec in enumerate(bigvecs):
                self.docvecs[model][lang].add_item(index, vec)

        if model == 'senlaser':
            print('Calculating document vectors using LASER Sentence Embeddings on Individual Sentences in Article.')
            self.s2d[model][lang] = []
            count = 0
            bigtext = []
            for index, doc in enumerate(self.docs[lang]):
                sens = list(processing.zh_sent_tokenize(doc[1])) if lang == 'zh' else list(sent_tokenize(doc[1]))
                bigtext += sens
                for _ in sens:
                    self.s2d[model][lang].append(index)

                if index%100 == 99 or index==len(self.docs[lang]):
                    vecs = LASER.get_vect('\n'.join(bigtext), lang=lang if lang == 'ta' else 'en')
                    for vec in vecs:
                        self.docvecs[model][lang].add_item(count, vec)
                        count += 1
                    bigtext = []
            pickle.dump(self.s2d[model][lang], open('%s/%s/s2d%s.pkl' % directory,'wb'))

        if model == 'senbai':
            self.s2d[model][lang] = []
            count = 0

            # First pass
            if lang in self.idfs:
                print('IDFs already loaded')
            elif os.path.isfile('%s/pickle/idfs.pkl' % self.path):
                self.idfs = pickle.load( open('%s/pickle/idfs.pkl' % self.path, 'rb'))
                print('Loaded IDFs')
            else:
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
                pickle.dump( self.idfs, open('%s/pickle/idfs.pkl' % self.path, 'wb'))
                print('IDFs saved at %s/pickle/idfs.pkl' % self.path)

            # Second pass
            print('Calculating document vectors using Bilingual Word Embeddings (TF-IDF) on Individual Sentences in Article.')
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
                    vecs = sum(np.array(vec) for vec in s_tokens_vecs)
                    if isinstance(vecs, (list, tuple, np.ndarray)):
                        self.docvecs[model][lang].add_item(count, vecs)
                    self.s2d[model][lang].append(index)
                    count += 1
            
            pickle.dump(self.s2d[model][lang], open('%s/%s/s2d%s.pkl' % directory ,'wb'))


        self.docvecs[model][lang].build(math.floor(math.log(len(self.docs[lang]))))
        self.docvecs[model][lang].save('%s/%s/%s.ann' % directory)
        print('Annoy Index is saved and loaded in %s for %s lang and %s model!' % directory)