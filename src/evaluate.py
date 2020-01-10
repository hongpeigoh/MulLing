import statistics
from . import porcessing, get

# Mean Square Cosine Similarity, only test for non KD-Tree queries.
def _evaluate_precision(self, results, k, l=-1):
    return statistics.mean([cs**2 for cs, ai, lang in results])

# Half-L Recall
def _evaluate_recall(self, results, q_vecs, k, l=-1):
    list1 = [(ai, lang) for cs, ai, lang in results]
    list2 = []
    for lang in self.langs:
        list2 += [(ai, lang) for cs, ai, lang in processing.title_query(q_vecs, lang, k=l//2)]
    return len(Union(list1,list2))/len(list(2))