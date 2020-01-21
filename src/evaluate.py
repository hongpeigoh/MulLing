import os,sys
import statistics
from scipy.stats import friedmanchisquare
import pandas as pd
import random
import numpy as np
from IPython.display import display
from . import processing, get, query

# Mean Square Cosine Similarity, only test for non KD-Tree queries.
def _evaluate_precision(self, results):
    return statistics.mean([cs**2 for cs, ai, lang in results])

# Half-L Recall
def _evaluate_recall(self, results, q_vecs, L=-1):
    list1 = [(ai, lang) for cs, ai, lang in results]
    list2 = []
    for lang in self.langs:
        list2 += [(ai, lang2) for cs, ai, lang2 in query.title_query(self, q_vecs, lang, k=L//2)]
    return len(set(list1).intersection(set(list2)))/len(list2)

# Pipeline evaluator
class Evaluator:
    def __init__(self, inpath , k, L, MulLingObjects, methods=[1,2,3]):
        if L == -1:
            L = k
        elif k> len(MulLingObjects[0].langs)*L:
            raise ValueError('The number of search results cannot be displayed as L is too small!')
        
        self.k = k
        self.L = L
        self.outpath = 'dump/evaluator.csv'
        self.queries = list()
        if inpath[-4:] == '.csv':
            with open(inpath, encoding='utf-8') as f:
                self.n_queries = f.readline()[:-1]
                for line in f:
                    self.queries.append(line[:-1])
        method_list = ['BWE-Agg-Add','BWE-Agg-IDF','LASER']
        self.methods = list(map(lambda x: method_list[x-1], methods))
        self.table = pd.DataFrame(columns=['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)]).set_index('Queries')
        self.data = dict() 
        for query_ in self.queries:
            self.data[query_] = pd.DataFrame(columns=['Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)])
            #Title Row
            results = list()
            mean = dict()
            vec = processing.vectorize(MulLingObjects[0], query_)
            if not isinstance(vec, (list, tuple, np.ndarray)):
                self.queries.remove(query_)
            for lang in MulLingObjects[0].langs:
                results += query.title_query(MulLingObjects[0], vec, lang, L)
                mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])
            results = map( lambda x: (x[0]/mean[x[2]] , x[1], x[2]) , results)
            results = list(sorted(results, key=get._getitem)[:-k-1:-1])

            new_row = pd.Series(dict(zip(self.data[query_].columns, ['Title']+ [(ai, lang) for cs,ai, lang in results])))
            self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

            for index, method_ in enumerate(self.methods):
                if method_=='BWE-Agg-Add' or method_=='BWE-Agg-IDF':
                    results = list()
                    for i in range(self.k):
                        ai, lang = self.data[query_]['Article{}/Ranking'.format(i+1)][0]
                        d_vecs = MulLingObjects[index].docvecs[lang][ai]
                        print(d_vecs)
                        results.append(np.dot( d_vecs, vec) / ( np.linalg.norm(d_vecs) * np.linalg.norm(vec) ))
                    
                    new_row = pd.Series(dict(zip(self.data[query_].columns, [method_] + results)))
                    self.data[query_] = self.data[query_].append(new_row, ignore_index=True)


            self.data[query_]['Queries'] = query_

    def export_to_csv(self, outpath=''):
        self.data.to_csv(outpath, encoding='utf-8')

    def evaluate(self, method):
        pass

    def tabulate(self):
        for query_ in self.data:
            self.table = self.table.append(self.data[query_])
        self.table = self.table[['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(self.k)]]
        display(self.table)


if __name__ == '__main__':
    sys.path.insert(0,'..')
    from .main import *
    my_MulLing_objs = dict()
    my_MulLing_objs[1] = MulLingVectors(method=1)
    my_Evaluator = Evaluator('dump/queries.csv', )

