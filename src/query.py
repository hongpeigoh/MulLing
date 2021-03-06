import numpy as np
import statistics
from . import processing, get
from pkgs import LASER
from sklearn.cluster import MeanShift

def monolingual_annoy_query(self, q, model, lang, k, clustering=False):
    """
    Monolingual/Multilingual Query
    --------------------
    q: str
    Input Query
    --------------------
    model: str
    One of the initialised models in MulLingVectorsAnnoy() class. See ../mulling.py for more.
    --------------------
    lang: str
    Language of input query
    --------------------
    k: int
    Number of results to generate
    --------------------
    clustering: bool
    Returns results with clusters and most similar keywords by cosine similarity and idfs if true
    --------------------
    Returns:
    <generator> of k tuples each in the form of:
    1. cosine_similarity a.k.a. angular distance
    2. article_index (queried by MulLingVectorsAnnoy().docs[lang][ai]); or
       sentence_index for sentence models (queried by MulLingVectorsAnnoy().docs[lang][ai] where ai = MulLingVectorsAnnoy().s2d[model][lang][si])
    3. language of returned tuple
    4. clustering_labels.
       = -1 for queries without clustering
       = int for queries with clustering. Result tuples in the same cluster share the same clustering_labels, starting from 0.
    5. cluster_keywords
       top 5 most frequently used words among top 20 most similar words for each cluster center

    """
    # Vectorize query
    q_vecs = LASER.get_vect(q)[0] if 'laser' in model else processing.vectorize_lang(self, q, lang) 
    
    # Find and return results in ranked list
    if not clustering:
        if 'sen' in model:
            for si, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, k, include_distances=True))):
                yield (cs, self.s2d[model][lang][si], lang, si, -1)
        else:
            for ai, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, k, include_distances=True))):
                yield (cs, ai, lang, -1)
    else:
        if 'sen' in model:
            tmp = list([cs, self.s2d[model][lang][si], lang, si] for si, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, k, include_distances=True))))
        else:
            tmp = list([cs, ai, lang, -1] for ai, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, k, include_distances=True))))
        vecs = np.array([self.docvecs[model][result[2]].get_item_vector(result[1]) for result in tmp])
        # Mean Shift Clusters
        clustering = MeanShift().fit(vecs)
        if 'annoy/%s'%lang in self.vecs:
            keywords = []
            for cc in clustering.cluster_centers_:
                keywords.append(sorted( list(self.vecs[lang].id2word[wi] for wi in self.vecs['annoy/%s'%lang].get_nns_by_vector(cc, 20)), key=lambda x: self.idfs[lang][x])[:5])
        for i in range(k):
            yield tuple(tmp[i] + [clustering.labels_[i], keywords[clustering.labels_[i]]])
            


def mulling_annoy_query(self, q, model, k, L=-1, normalize_top_L=True, multilingual=True, lang_='en', olangs= None, clustering=False):
    """
    Refer to above for key variables and tuples returned
    --------------------
    L: int
    Maximum number of results to generate for any given output language. Set to -1 if no criteria.
    --------------------
    normalize_top_L: bool
    Returns results sorted weighing aginst the average cosine similarity of results in each output language
    --------------------
    multilingual: bool, and lang_:str
    multilingual should be set to true if the input query is of an unknown language/dual language and
    lang_ should be set to the input query's language if known and closest language among initialised languages (See ../mulling.py) otherwise.
    --------------------
    olangs: list(str)
    List of languages to be included in output, should be a subset of initialised languages.
    """
    if L == -1:
        L = k
    elif k> len(self.langs)*L:
        raise ValueError('The number of search results cannot be displayed as L is too small!')
    if olangs == None:
        olangs = self.langs
    else:
        for lang in olangs:
            if lang not in self.langs:
                raise KeyError('The specified output language is not accepted!')
    
    # Vectorize query
    if 'laser' in model:
        q_vecs = LASER.get_vect(q)[0]
    else:
        if multilingual:
            q_vecs = processing.vectorize(self, q)
        else:
            q_vecs = processing.vectorize_lang(self, q, lang_)

    # Find results
    results = list()
    for lang in olangs:
        if not clustering:
            if 'sen' in model:
                results += list((cs, self.s2d[model][lang][si], lang, si) for si, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, L, include_distances=True))))
            else:
                results += list((cs, ai, lang, -1) for ai, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, L, include_distances=True))))
        else:
            if 'sen' in model:
                tmp = list([cs, self.s2d[model][lang][si], lang, si] for si, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, L, include_distances=True))))
            else:
                tmp = list([cs, ai, lang, -1] for ai, cs in list(zip(*self.docvecs[model][lang].get_nns_by_vector(q_vecs, L, include_distances=True))))
            vecs = np.array([self.docvecs[model][result[2]].get_item_vector(result[1]) for result in tmp])
            # Mean Shift Clusters
            clustering = MeanShift().fit(vecs)
            if 'annoy/en' in self.vecs:
                keywords = []
                for cc in clustering.cluster_centers_:
                    keywords.append(sorted( list(self.vecs['en'].id2word[wi] for wi in self.vecs['annoy/en'].get_nns_by_vector(cc, 20)), key=lambda x: self.idfs['en'][x])[:5])
            for i in range(L):
                results.append(tuple(tmp[i] + [clustering.labels_[i], keywords[clustering.labels_[i]]]))
        self.mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])

    # Return results in ranked list
    if normalize_top_L:
        return sorted(results, key=lambda tuple_: get.normalizedtopLitem(self, tuple_))[:k]
    else:
        return sorted(results, key=lambda tuple_: get.normalizeditem(self,tuple_))[:k]



def ensemble_query(self, ensemble, q, k, lang='en', L=-1, normalize_top_L = True, multilingual = True):
    """
    Query for testing ensemble of models. Mostly deprecated due to long runtime.
    Refer to above for variables and returned tuples (no clustering).
    """
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



'''
OLD STUFF

Input Monolingual Query, searched using brute force on documents in query language.
Instances two linked arrays: Cosine Similarity Ranked Relevance List and 
Article Index Ranked Relevance List, by searching the k nearest neighbours.
These results are then transposed to form an array of tuples 
(CosineSimilarity, ArticleIndex, Language)

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


Input Monolingual Query, searched using brute force on all documents in all languages.
Instances two linked arrays: Cosine Similarity Ranked Relevance List and 
Article Index Ranked Relevance List, by searching the k nearest neighbours.
These results are then transposed to form an array of tuples 
(CosineSimilarity, ArticleIndex, Language)

Credits: Avery

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


Input Monolingual Query, searched using SciPy's KDTrees on documents in query language.
The k nearest neighbours are returned in an array of tuples
(CosineSimilarity, ArticleIndex, Language)

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


Input Query vectors (query has undergone tokenization and vectorization),searched 
using brute force on documents in query language. Top k results are returned in
(CosineSimilarity, ArticleIndex, Language)
 
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


Input Query vectors (query has undergone tokenization and vectorization),searched 
using KDTrees on documents in query language. Top k results are returned in
(CosineSimilarity, ArticleIndex, Language)

def vec_kdtree_query(self, q_vecs, lang, k):
    return list((cs, int(ai), lang) for cs, ai in np.column_stack(self.kdtrees[lang].query(q_vecs/np.linalg.norm(q_vecs), k=k)).tolist())


Input Monolingual Title Query, searched using brute force on documents in query language.
Results are similarly returned as (CosineSimilarity, ArticleIndex, Language)

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

Input Multilingual Query
k : Number of ranked search results returned as (CosineSimilarity, ArticleIndex, Language)
L : Maximum number of search results returned from any particular language. 
    Only valid for large k ≥ 30
kdtree = True : uses KDTrees instead of brute force to search. 
                Cosine Similarity (Descending) is replaced with Distance (Ascending)
normalize_top_L = True : normalizes the returned results by the mean cosine similarity of the top L results

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
'''