from pkgs.FastText import FastVector
import pickle
import numpy as np
import math
import statistics
from scipy import spatial
import os.path
import string
import codecs
import spacy
import jieba as nlp_zh
import malaya as nlp_ms
from pkgs.HindiTokenizer import Tokenizer as nlp_hi
from collections import defaultdict, Counter
from googletrans import Translator
gtrans = Translator()
nlp_en = spacy.load("en_core_web_md")

class MulLingVectors:
    def __init__(self, method=1, langs=['en', 'zh', 'ms', 'ta']):
        """
        Method 1: BWE-Agg-Add
        Method 2: BWE-Agg-IDF
        Method 3: Pre-loaded document vectors (BWE-Agg-IDF)
        """
        self.method = method
        self.langs = langs
        self.vecs = {}
        self.docs = {}
        self.docvecs = defaultdict(list)
        self.metadocvecs = defaultdict(list)
        self.stopwords = {}
        self.idfs = {}
        self.kdtrees = {}

        if self.method == 3:
            paths = {
                docvecs: 'pickle/docvecs2.pkl'        # input('Insert path of doc-vecs (default is \'pickle/docvecs2.pkl\'): ')
                idfs: 'pickle/idfs2.pkl'              # input('Insert path of idfs (default is \'pickle/idfs2.pkl\'): ')
                metadocvecs: 'pickle/metadocvecs.pkl' # input('Insert path of doc-vecs (default is \'pickle/metadocvecs.pkl\'): ')
                #kdtrees: 'pickle/kdtrees.pkl'        # input('Insert path of idfs (default is \'pickle/kdtrees.pkl\'): ')
            }
            self.docvecs = pickle.load( open(paths[docvecs], 'rb'))
            self.idfs = pickle.load( open(paths[idfs], 'rb'))
            self.metadocvecs = pickle.load( open(paths[metadocvecs], 'rb'))
            #self.kdtrees = pickle.load( open(paths[kdtrees], 'rb'))

        for lang in self.langs:
            self.load(lang)
            if self.method == 1:
                self.calculate_docvecs(lang)
                self.calculate_metadocvecs(lang)
            elif self.method == 2:
                self.calculate2_docvecs(lang)
                self.calculate_metadocvecs(lang)
            elif self.method == 3:
                if lang in self.docvecs and lang in self.idfs:
                    print('The {} document vectors and inverse document frequencies are already loaded!'.format(lang))
                else:
                    self.calculate2_docvecs(lang)
                if lang in self.metadocvecs:
                    print('The {} meta-document vectors are already loaded!'.format(lang))
                else:
                    self.calculate_metadocvecs(lang)
            else: raise TypeError('Invalid method type')

            #Set up KD-Tree for subsequent queries
            print('Instancing {} KD-Tree'.format(lang))
            self.kdtrees[lang] = spatial.KDTree(list(map(lambda x: x/np.linalg.norm(x), self.docvecs[lang])))
            print('KD-tree loaded!')
        print('All vector dictionaries loaded!')
        
    # Load the selected language into the class 'MulLingVectors'
    def load(self, lang:str):
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

        # Load stopwords for NLP modules
        if lang == 'zh':
            f = codecs.open('stopwords/stopwords-zh.txt', encoding='utf-8')    # https://github.com/stopwords-iso/stopwords-zh
            self.stopwords[lang] = [str(line)[0] for line in f][1:]
            f.close()
        elif lang == 'ms':
            f = codecs.open('stopwords/stopwords-ms.txt', encoding='utf-8')    # https://github.com/stopwords-iso/stopwords-ms
            self.stopwords[lang] = [repr(line)[1:-5] for line in f][1:]
            f.close()
        elif lang == 'ta':
            f = codecs.open('stopwords/stopwords-ta.txt', encoding='utf-8')    # https://github.com/AshokR/TamilNLP/blob/master/Resources/TamilStopWords.txt
            self.stopwords[lang] = [repr(line)[1:-5] for line in f][1:]
            f.close()



    def calculate_docvecs(self, lang:str):
        # Calculate summation of document vectors by vector addition.
        print('Calculating document vectors')
        self.docvecs[lang] = list()
        
        for doc in self.docs[lang]:
            d_tokens = self.tokenize(lang, doc[1])
            d_tokens_vecs = []
            for token in list(d_tokens):
                try:
                    d_tokens_vecs.append(self.vecs[lang][token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            self.docvecs[lang].append(sum(np.array(vec) for vec in d_tokens_vecs))

    def calculate_metadocvecs(self, lang:str):
        # Calculate summation of meta-document vectors by vector addition.
        print('Calculating meta-document vectors')
        self.metadocvecs[lang] = list()
        
        for doc in self.docs[lang]:
            t_tokens = self.tokenize(lang, doc[0])
            t_tokens_vecs = []
            for token in list(t_tokens):
                try:
                    t_tokens_vecs.append(self.vecs[lang][token])
                except:
                    pass
                    #raise KeyError('{} not in {} dictionary'.format(token, lang))
            self.metadocvecs[lang].append(sum(np.array(vec) for vec in t_tokens_vecs))      

    def calculate2_docvecs(self, lang:str):
        # Calculate Inverse Document Frequency (IDF) on first pass
        print('Calculating IDFs')
        N = len(self.docs[lang])
        dfs = dict()
        for doc in self.docs[lang]:
            d_tokens = self.tokenize(lang,doc[1])
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
            d_tokens = self.tokenize(lang, doc[1])
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
    


    # Tokenizes Input
    def tokenize(self, lang, doc):
        if lang == 'en':
            d = nlp_en(doc.replace('\n', ' '))
            d_tokens = filter(lambda x : (x.is_stop == False and x.is_punct == False and x.pos_!='NUM'), d)
            return map(lambda x: str(x).lower(), d_tokens)
        elif lang == 'zh':
            d = nlp_zh.cut(doc, cut_all=False)
            return filter(lambda x: (x not in self.stopwords[lang] and x not in string.punctuation), d)
        elif lang == 'ms':
            d = nlp_ms.preprocessing.SocialTokenizer().tokenize(doc)
            d_tokens = filter(lambda x: (x not in self.stopwords[lang] and x not in string.punctuation), d)
            return map(lambda x: str(x).lower(), d_tokens)
        elif lang == 'ta':
            d = nlp_hi(doc)
            d.tokenize()
            return filter(lambda x: (x not in self.stopwords[lang] and x not in string.punctuation), d.tokens)

    def mulling_vectorize(self, q):
        Q, tokens, token_vecs = dict(), dict(), list()
        # To tokenize and add spaces to chinese text in multilingual query
        for word in q.split():
            if word != ' ':
                lang = gtrans.detect(word).lang[:2].replace('id','ms')
                if lang in self.langs:
                    if lang in Q:
                        Q[lang].append(word)
                    else:
                        Q[lang] = [word]
                else:
                    word_ = gtrans.translate(word, dest='en').text
                    if self.langs[0] in Q:
                        Q[self.langs[0]].append(word_)
                    else:
                        Q[self.langs[0]] = [word_]
        
        # To vectorise the query by using the language of each query token. Vectors are then added to form the query vector
        for lang in Q:
            tokens[lang] = list(self.tokenize(lang,' '.join(Q[lang])))
            for token in tokens[lang]:
                try:
                    token_vecs.append(self.vecs[lang][token])
                except:
                    print(tokens)
                    raise KeyError('{} cannot be found in {} dictionary'.format(token, lang))
        return sum(np.array(vec) for vec in token_vecs)



    '''
    Input Monolingual Query, searched using brute force on documents in query language.
    Instances two linked arrays: Cosine Similarity Ranked Relevance List and 
    Article Index Ranked Relevance List, by searching the k nearest neighbours.
    These results are then transposed to form an array of tuples 
    (CosineSimilarity, ArticleIndex, Language)
    '''
    def simple_query(self, q, lang, k):
        cs_rrl = [0 for _ in range(k)]
        ai_rrl = [0 for _ in range(k)]
        q_tokens = self.tokenize(lang, q)
        q_tokens_vecs = []
        for token in list(q_tokens):
            try:
                q_tokens_vecs.append(self.vecs[lang][token])
            except:
                raise KeyError('{} cannot be found in dictionary'.format(token))
        q_vecs = sum(np.array(vec) for vec in q_tokens_vecs)

        for index, vec in enumerate(self.docvecs[lang]):
            r = np.dot( q_vecs, vec) / ( np.linalg.norm(q_vecs) * np.linalg.norm(vec) )
            m = 0
            while(m < k):
                if r < cs_rrl[m]:
                    m += 1 
                else: 
                    break
            if m < k:
                cs_rrl.insert(m, r)
                ai_rrl.insert(m, index)
                cs_rrl, ai_rrl = cs_rrl[:-1], ai_rrl[:-1]
        return list((cs, ai, lang) for cs,ai in list(zip(cs_rrl, ai_rrl)))

    '''
    Input Monolingual Query, searched using brute force on all documents in all languages.
    Instances two linked arrays: Cosine Similarity Ranked Relevance List and 
    Article Index Ranked Relevance List, by searching the k nearest neighbours.
    These results are then transposed to form an array of tuples 
    (CosineSimilarity, ArticleIndex, Language)

    Credits: Avery
    '''
    def simple_raw_score_merge_query(self, q, lang, k):
        cs_rrl = [0 for _ in range(k)]
        ai_rrl = [0 for _ in range(k)]
        q_tokens = self.tokenize(lang, q)
        q_tokens_vecs = []
        for token in list(q_tokens):
            try:
                q_tokens_vecs.append(self.vecs[lang][token])
            except:
                raise KeyError('{} cannot be found in dictionary'.format(token))
        q_vecs = sum(np.array(vec) for vec in q_tokens_vecs)

        for lang2 in self.langs:
            for index, vec in enumerate(self.docvecs[lang2]):
                r = np.dot( q_vecs, vec) / ( np.linalg.norm(q_vecs) * np.linalg.norm(vec) )
                m = 0
                while(m < k):
                    if r < cs_rrl[m]:
                        m += 1 
                    else: 
                        break
                if m < k:
                    cs_rrl.insert(m, r)
                    ai_rrl.insert(m, index)
                    cs_rrl, ai_rrl = cs_rrl[:-1], ai_rrl[:-1]
            return list((cs, ai, lang2) for cs,ai in list(zip(cs_rrl, ai_rrl)))

    '''
    Input Monolingual Query, searched using SciPy's KDTrees on documents in query language.
    The k nearest neighbours are returned in an array of tuples
    (CosineSimilarity, ArticleIndex, Language)
    ''' 
    def kdtree_query(self, q, lang, k):
        q_tokens = self.tokenize(lang,q)
        q_tokens_vecs = []
        for token in list(q_tokens):
            try:
                q_tokens_vecs.append(self.vecs[lang][token])
            except:
                raise KeyError('{} cannot be found in dictionary'.format(token))
        q_vecs = sum(np.array(vec) for vec in q_tokens_vecs)

        return list((cs, int(ai), lang) for cs, ai in np.column_stack(self.kdtrees[lang].query(q_vecs/np.linalg.norm(q_vecs), k=k)).tolist())

    '''
    Input Query vectors (query has undergone tokenization and vectorization),searched 
    using brute force on documents in query language. Top k results are returned in
    (CosineSimilarity, ArticleIndex, Language)
    ''' 
    def vec_query(self, q_vecs, lang, k):
        cs_rrl = [0 for _ in range(k)]
        ai_rrl = [0 for _ in range(k)]
        for index, vec in enumerate(self.docvecs[lang]):
            r = np.dot( q_vecs, vec) / ( np.linalg.norm(q_vecs) * np.linalg.norm(vec) )
            m = 0
            while(m < k):
                if r < cs_rrl[m]: m += 1 
                else: break
            if m < k:
                cs_rrl.insert(m, r)
                ai_rrl.insert(m, index)
                cs_rrl, ai_rrl = cs_rrl[:-1], ai_rrl[:-1]
        return list((cs, ai, lang) for cs,ai in list(zip(cs_rrl, ai_rrl)))

    '''
    Input Query vectors (query has undergone tokenization and vectorization),searched 
    using KDTrees on documents in query language. Top k results are returned in
    (CosineSimilarity, ArticleIndex, Language)
    ''' 
    def vec_kdtree_query(self, q_vecs, lang, k):
        return list((cs, int(ai), lang) for cs, ai in np.column_stack(self.kdtrees[lang].query(q_vecs/np.linalg.norm(q_vecs), k=k)).tolist())



    '''
    Input Monolingual Title Query, searched using brute force on documents in query language.
    Results are similarly returned as (CosineSimilarity, ArticleIndex, Language)
    '''
    def title_query(self, q_vecs, lang, k):
        cs_rrl = [0 for _ in range(k)]          # Cosine Similarity Ranked Relevance List
        ai_rrl = [0 for _ in range(k)]          # Article Index Ranked Relevance List
        for index, vec in enumerate(self.metadocvecs[lang]):
            r = np.dot( q_vecs, vec) / ( np.linalg.norm(q_vecs) * np.linalg.norm(vec) )   # Cosine Similarity
            m = 0
            while(m < k):
                if r < cs_rrl[m]: m += 1 
                else: break
            if m < k:
                cs_rrl.insert(m, r)
                ai_rrl.insert(m, index)
                cs_rrl, ai_rrl = cs_rrl[:-1], ai_rrl[:-1]
        return list((cs, ai, lang) for cs,ai in list(zip(cs_rrl, ai_rrl)))



    '''
    Input Multilingual Query
    k : Number of ranked search results returned as (CosineSimilarity, ArticleIndex, Language)
    L : Maximum number of search results returned from any particular language. 
        Only valid for large k ≥ 30
    kdtree = True : uses KDTrees instead of brute force to search. 
                    Cosine Similarity (Descending) is replaced with Distance (Ascending)
    normalize_top_L = True : normalizes the returned results by the mean cosine similarity of the top L results
    '''
    def mulling_query(self, q, k, L=-1, kdtree = True, normalize_top_L = False):
        if L == -1:
            L = k
        elif k<30:
            L = k
        elif k> len(self.langs)*L:
            raise ValueError('The number of search results cannot be displayed as L is too small!')
        vecs = self.mulling_vectorize(q=q)

        results = []
        if normalize_top_L:
            mean = dict()
            if kdtree:
                for lang in self.langs:
                    results += self.vec_kdtree_query(vecs, lang, L)
                    mean[lang] = statistics.mean([results[-i-1][0] for i in range(l)])
                # Get first item, normalized by the average cosine similarity of the top l results of that language
                results = map( lambda x: (x[0]/mean[x[2]] , x[1], x[2]) , results)
                return sorted(results, key=self._getitem)[:k]
            else:
                for lang in self.langs:
                    results += self.vec_query(vecs, lang, L)
                    mean[lang] = statistics.mean([results[-i-1][0] for i in range(l)])
                # Get first item, normalized by the average cosine similarity of the top l results of that language
                results = map( lambda x: ( x[0]/mean[x[2]] , x[1], x[2]) , results)
                return sorted(results, key=self._getitem)[:-k-1:-1]

        else:
            if kdtree:
                for lang in self.langs:
                    results += self.vec_kdtree_query(vecs, lang, L)
                return sorted(results, key=self._getdenormalizeditem)[:k]
            else:
                for lang in self.langs:
                    results += self.vec_query(vecs, lang, L)
                return sorted(results, key=self._getnormalizeditem)[:-k-1:-1]



    # Get first item
    def _getitem(self, tuple_):
        return tuple_[0]

    # Get first item, the cosine similarity/ shortest distance (1st element), normalized by a factor of 1/lg(N) where N is the size of the language's document corpora to reduce 
    def _getnormalizeditem(self, tuple_):
        return tuple_[0] / math.log(len(self.docs[tuple_[2]]))

    # Get first item, the cosine similarity/ shortest distance (1st element), normalized by a factor of *lg(N) where N is the size of the language's document corpora to reduce
    def _getdenormalizeditem(self, tuple_):
        return tuple_[0] * math.log(math.log(len(self.docs[tuple_[2]])))

    # Get first item, normalized by the average cosine similarity of the top k results
    def _getnormalized_2l_item(self, tuple_):
        return tuple_[0] / statistics.mean()

    # Get article using the language (2nd element) and the article index (3rd element)
    def _getarticle(self, tuple_):
        return self.docs[tuple_[2]][tuple_[1]]

    # Get readable results of enumerated titles from a query:
    def _getresults(self, query_results):
        for cs, ai, lang in query_results:
            print(my_object.docs[lang][ai][0])



    # Mean Square Cosine Similarity, only test for non KD-Tree queries.
    def _evaluate_precision(self, results, k, l=-1):
        return statistics.mean([cs**2 for cs, ai, lang in results])

    # Half-L Recall
    def _evaluate_recall(self, results, q_vecs, k, l=-1):
        list1 = [(ai, lang) for cs, ai, lang in results]
        list2 = []
        for lang in self.langs:
            list2 += [(ai, lang) for cs, ai, lang in self.title_query(q_vecs, lang, k=l//2)]
        return len(Union(list1,list2))/len(list(2))

        
