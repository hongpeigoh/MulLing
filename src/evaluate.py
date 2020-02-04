import time
import os,sys
import statistics
import functools
import math
from scipy.stats import friedmanchisquare
import scikit_posthocs
import pandas as pd
import random
import numpy as np
from IPython.display import display
from . import processing, get, query
from pkgs import LASER

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
    def __init__(self, inpath , k, L, MulLingObject, methods=[1,2,3]):
        if L == -1:
            L = k
        elif k> len(MulLingObject.langs)*L:
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
        
        # Methods
        # baa: BWE-Agg-Add
        # bai: BWE-Agg-IDF
        # lasers: Language Agnostic SEntence Representations
        method_list = ['baa','bai','lasers']
        self.methods = list(map(lambda x: method_list[x-1], methods))
        self.table = pd.DataFrame(columns=['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)]).set_index('Queries')
        self.data = dict() 
        for query_ in self.queries:
            self.data[query_] = pd.DataFrame(columns=['Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)])
            #Title Row
            results = list()
            mean = dict()
            vec = processing.vectorize_lang(MulLingObject, query_, 'en')
            if not isinstance(vec, (list, tuple, np.ndarray)):
                self.queries.remove(query_)
            for lang in MulLingObject.langs:
                results += query.title_query(MulLingObject, vec, lang, L)
                mean[lang] = statistics.mean([results[-i-1][0] for i in range(L)])
            results = map( lambda x: (x[0]/mean[x[2]] , x[1], x[2]) , results)
            results = list(sorted(results, key=get._getitem)[:-k-1:-1])

            new_row = pd.Series(dict(zip(self.data[query_].columns, ['Title']+ [(ai, lang) for cs,ai, lang in results])))
            self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

            for index, method_ in enumerate(self.methods):
                if method_ == 'baa':
                    name = 'BWE-Agg-Add'
                elif method_ == 'bai':
                    name = 'BWE-Agg-IDF'
                elif method_ == 'lasers':
                    name = 'LASER'
                    vec = LASER.get_vect(query_)[0]
                results = list()
                for i in range(self.k):
                    ai, lang = self.data[query_]['Article{}/Ranking'.format(i+1)][0]
                    if method_ == 'lasers':
                        d_vecs = MulLingObject.docvecs[method_][lang][ai][0]
                    else:
                        d_vecs = MulLingObject.docvecs[method_][lang][ai]
                    results.append(np.dot( d_vecs, vec) / ( np.linalg.norm(d_vecs) * np.linalg.norm(vec) ))

                new_row = pd.Series(dict(zip(self.data[query_].columns, [name] + results)))
                self.data[query_] = self.data[query_].append(new_row, ignore_index=True)
                

            self.data[query_]['Queries'] = query_

    def export_to_csv(self, outpath=''):
        self.table.to_csv(outpath, encoding='utf-8')

    def evaluate(self, method):
        pass

    def tabulate(self):
        self.ensemble = []
        for query_ in self.data:
            f_data1, f_data2, f_data3 = self.data[query_].iloc[1:4,1:-1].values.tolist()
            stat, p =friedmanchisquare(f_data1, f_data2, f_data3)
            ph_data = scikit_posthocs.posthoc_nemenyi_friedman(np.array([f_data1, f_data2, f_data3]).T)
            ph_min = [math.sqrt(-1*np.prod(ph_data[i])) for i in range(len(ph_data[0]))]
            ensemble = [n/sum(ph_min) for n in ph_min]
            self.data[query_]['Non-parametric (Friedman, Nemenyi)'] = ['p=%.3f'%p] + ensemble
            self.table = self.table.append(self.data[query_])
            if p>=0.05:
                self.ensemble.append(ensemble)
        self.table = self.table[['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(self.k)] + ['Non-parametric (Friedman, Nemenyi)']].set_index('Queries', append=True).swaplevel(0,1)
        display(self.table)
        self.ensembles = np.asarray(self.ensemble).T.tolist()
        self.ensemble = [statistics.mean(n) for n in self.ensembles]
        print('Ensemble is %.3fx of BAA, %.3fx of BAI and %.3fx of LASER' % (self.ensemble[0],self.ensemble[1],self.ensemble[2]))

    def print_results(self):
        display(self.table)
        print('Ensemble is %.3fx of BAA, %.3fx of BAI and %.3fx of LASER' % (self.ensemble[0],self.ensemble[1],self.ensemble[2]))
        print(self.ensembles)




if __name__ == '__main__':
    sys.path.insert(0,'..')
    from .main import *
    my_MulLing_objs = dict()
    my_MulLing_objs[1] = MulLingVectors(method=1)
    my_Evaluator = Evaluator('dump/queries.csv', )

