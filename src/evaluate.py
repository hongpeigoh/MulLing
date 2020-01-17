import os,sys
import statistics
import pandas as pd
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
    def __init__(self, inpath , test=[1,2,3]):
        self.outpath = 'dump/evaluator.csv'
        if inpath[-4:] == '.csv':
            self.queries = pd.read_csv(inpath, encoding='utf-8')
        self.methods = ['BWE-Agg-Add','BWE-Agg-IDF','LASER']
        self.methods = list(filter(lambda x: self.methods[x-1], test))

    def export_to_csv(self, outpath=''):
        self.queries.to_csv(outpath, encoding='utf-8')

    def evaluate(self, method):
        pass


if __name__ == '__main__':
    sys.path.insert(0,'..')
    from .main import *
    my_MulLing_objs = dict()
    my_MulLing_objs[1] = MulLingVectors(method=1)
    my_Evaluator = Evaluator('dump/queries.csv', )

