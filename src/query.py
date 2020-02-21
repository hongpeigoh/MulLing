import numpy as np
import statistics
from . import processing, get
from pkgs import LASER

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
    q_tokens = processing.tokenize(lang, q)
    q_tokens_vecs = []
    for token in list(q_tokens):
        try:
            q_tokens_vecs.append(self.vecs[lang][token])
        except:
            raise KeyError('{} cannot be found in dictionary'.format(token))
    q_vecs = sum(np.array(vec) for vec in q_tokens_vecs)

    for index, vec in enumerate(self.dv[lang]):
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
    q_tokens = processing.tokenize(lang, q)
    q_tokens_vecs = []
    for token in list(q_tokens):
        try:
            q_tokens_vecs.append(self.vecs[lang][token])
        except:
            raise KeyError('{} cannot be found in dictionary'.format(token))
    q_vecs = sum(np.array(vec) for vec in q_tokens_vecs)

    results = []
    for lang_ in self.langs:
        cs_rrl = [0 for _ in range(k)]
        ai_rrl = [0 for _ in range(k)]
        for index, vec in enumerate(self.dv[lang_]):
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
        results += list((cs, ai, lang_) for cs,ai in list(zip(cs_rrl, ai_rrl)))
    return sorted(results, key=get.item)[:-k-1:-1]

'''
Input Monolingual Query, searched using SciPy's KDTrees on documents in query language.
The k nearest neighbours are returned in an array of tuples
(CosineSimilarity, ArticleIndex, Language)
''' 
def kdtree_query(self, q, lang, k):
    q_tokens = processing.tokenize(lang,q)
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
    for index, vec in enumerate(self.dv[lang]):
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
    for index, vec in enumerate(self.docvecs['meta'][lang]):
        if not isinstance(vec, (list, tuple, np.ndarray)):
            pass
        else:
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
def mulling_query(self, q, k, L=-1, kdtree = False, normalize_top_L = False):
    if L == -1:
        L = k
    elif k> len(self.langs)*L:
        raise ValueError('The number of search results cannot be displayed as L is too small!')
    vecs = processing.vectorize(self,input_=q)

    results = []
    if normalize_top_L:
        self.mean = {}
        if kdtree:
            for lang in self.langs:
                results += vec_kdtree_query(self,vecs, lang, L)
                self.mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])
            # Get first item, normalized by the average cosine similarity of the top l results of that language
            return sorted(results, key=lambda tuple_: get.normalizedtopLitem(self, tuple_))[:k]
        else:
            for lang in self.langs:
                results += vec_query(self,vecs, lang, L)
                self.mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])
            # Get first item, normalized by the average cosine similarity of the top l results of that language
            return sorted(results, key=lambda tuple_: get.normalizedtopLitem(self, tuple_))[:-k-1:-1]

    else:
        if kdtree:
            for lang in self.langs:
                results += vec_kdtree_query(self,vecs, lang, L)
            return sorted(results, key=lambda tuple_: get.denormalizeditem(self,tuple_))[:k]
        else:
            for lang in self.langs:
                results += vec_query(self, vecs, lang, L)
            return sorted(results, key=lambda tuple_: get.normalizeditem(self,tuple_))[:-k-1:-1]

def laser_query(self, q, lang, k):
    q_vecs = LASER.get_vect(q)[0]

    cs_rrl = [0 for _ in range(k)]          # Cosine Similarity Ranked Relevance List
    ai_rrl = [0 for _ in range(k)]          # Article Index Ranked Relevance List
    for index, vec in enumerate(self.docvecs['laser'][lang]):
        r = np.dot( q_vecs, vec[0]) / ( np.linalg.norm(q_vecs) * np.linalg.norm(vec[0]) )
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

def laser_mulling_query(self, q, k, L=-1, normalize_top_L = False):
    if L == -1:
        L = k
    elif k> len(self.langs)*L:
        raise ValueError('The number of search results cannot be displayed as L is too small!')
    
    self.mean = {}
    results = list()
    for lang in self.langs:
        results += laser_query(self, q, lang, L)
        self.mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])
    if normalize_top_L:
        return sorted(results, key=lambda tuple_: get.normalizedtopLitem(self, tuple_))[:k]
    else:
        sorted(results, key=lambda tuple_: get.normalizeditem(self,tuple_))[:-k-1:-1]

def ensemble_query(self, ensemble, q, lang='en' k, L=-1, normalize_top_L = True, multilingual = True):
    if L == -1:
        L = k
    elif k> len(self.langs)*L:
        raise ValueError('The number of search results cannot be displayed as L is too small!')
    
    if multilingual:
        vec1 = processing.vectorize(self, q)
    else:
        vec1 = processing.vectorize_lang(self, q, lang)
    vec2 = LASER.get_vect(q)[0]
    q_vecs = {
        'baa': vec1,
        'bai': vec1,
        'laser': vec2,
        'metalaser': vec2
    }

    self.mean = {}
    results = []
    for lang_ in self.langs:
        cs_rrl = [0 for _ in range(L)]
        ai_rrl = [0 for _ in range(L)]
        for index in range(len(self.docs[lang_])):
            r = 0
            for method_index, method in enumerate(['baa','bai','laser','metalaser']):
                vec = self.docvecs[method][lang_].get_item_vector(index)
                r += ensemble[method_index] * get.cosine_sim(vec, q_vecs[method])
            
            m = 0
            while(m < L):
                if r < cs_rrl[m]:
                    m += 1 
                else: 
                    break
            if m < L:
                cs_rrl.insert(m, r)
                ai_rrl.insert(m, index)
                cs_rrl, ai_rrl = cs_rrl[:-1], ai_rrl[:-1]
        results += list((cs, ai, lang_) for cs,ai in list(zip(cs_rrl, ai_rrl)))
        self.mean[lang_] = statistics.mean([results[-i-1][0] for i in range(L)])
    if normalize_top_L:
        return sorted(results, key=lambda tuple_: get.normalizedtopLitem(self, tuple_))[:-k-1:-1]
    else:
        return sorted(results, key=lambda tuple_: get.normalizeditem(self,tuple_))[:-k-1:-1]


def monolingual_annoy_query(self, q, model, lang, k):
    if model == 'laser' or model == 'metalaser':
        q_vecs = LASER.get_vect(q)[0]
    elif model == 'baa' or 'bai':
        q_vecs = processing.vectorize_lang(self, q, lang)              
    return list((cs, ai, lang) for ai, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, k, include_distances=True))))

def mulling_annoy_query(self, q, model, k, L=-1, normalize_top_L=True, multilingual=True):
    if L == -1:
        L = k
    elif k> len(self.langs)*L:
        raise ValueError('The number of search results cannot be displayed as L is too small!')
    
    if model == 'laser' or model == 'metalaser':
        q_vecs = LASER.get_vect(q)[0]
    else:
        if multilingual:
            q_vecs = processing.vectorize(self, q)
        else:
            q_vecs = processing.vectorize_lang(self, q, 'en')
    results = list()
    for lang in self.langs:
        results += list((cs, ai, lang) for ai, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, L, include_distances=True))))
        self.mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])
    if normalize_top_L:
        return sorted(results, key=lambda tuple_: get.normalizedtopLitem(self, tuple_))[:k]
    else:
        return sorted(results, key=lambda tuple_: get.normalizeditem(self,tuple_))[:k]