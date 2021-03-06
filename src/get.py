import statistics
import math
import numpy as np
from nltk.tokenize import sent_tokenize
from . import evaluate, processing

def cosine_sim(v1, v2):
    """
    Returns cosine similarity of vectors v1 and v2 (same dimensions)
    """
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))



"""
Methods for parsing individual results
"""
# Get first item
def item(tuple_):
    return tuple_[0]

# Get first item, the cosine similarity/ shortest distance (1st element), normalized by a factor of 1/lg(N) where N is the size of the language's document corpora to reduce 
def normalizeditem(self, tuple_):
    return tuple_[0] / math.log(len(self.docs[tuple_[2]]))

# Get first item, the cosine similarity/ shortest distance (1st element), normalized by a factor of *lg(N) where N is the size of the language's document corpora to reduce
def denormalizeditem(self, tuple_):
    return tuple_[0] * math.log(math.log(len(self.docs[tuple_[2]])))

# Get first item, normalized by the average cosine similarity of the top L results.
def normalizedtopLitem(self, tuple_):
    return tuple_[0] / self.mean[tuple_[2]]



"""
Methods for parsing articles
"""
# Get article using the language (2nd element) and the article index (3rd element)
def article(self, tuple_):
    return self.docs[tuple_[2]][tuple_[1]]

# Print prettified article in Jupyter
def printarticle(self, tuple_):
    print('\n\n\033[1m'+'\033[0m\n\n'.join(article(self, tuple_)))



"""
Methods for printing readable results
"""
# Get readable results of enumerated titles from a query:
def results(self, query_results):
    for index, result in enumerate(query_results):
        print('%i. %s' % (index+1, self.docs[result[2]][result[1]][0]))

# Get readable results of enumerated titles from a query:
def scoredresults(self, query_results):
    for index, result in enumerate(query_results):
        print('%i. %f %s' % (index+1, result[0], self.docs[result[2]][result[1]][0]))

# Get readable results of enumerated titles and articles from a query:
def all(self, query_results):
    for index, result in enumerate(query_results):
        print('\033[1m%i. %s\033[0m' % (index+1, self.docs[result[2]][result[1]][0]))
        print(self.docs[result[2]][result[1]][1][:200] + '...\n')

# Get readable results for sentence model:
def sentence(self, result, model):
    sennum = result[3] - self.s2d[model][result[2]].index(result[1])
    if result[2] == 'zh':
        return list(processing.zh_sent_tokenize(self.docs[result[2]][result[1]][1]))[sennum]
    else:
        return sent_tokenize(self.docs[result[2]][result[1]][1])[sennum]

# Get parsed results in json for flask container:
def json(self, query_results, model):
    for index, result in enumerate(query_results):
        rank = str(index+1)
        score = str(result[0])
        clustering = str(result[4])
        keywords = result[5]
        title = self.docs[result[2]][result[1]][0]
        text = self.docs[result[2]][result[1]][1]
        if result[3] >= 0:
            try:
                best_sen = sentence(self, result, model)
                yield ('\t'.join([rank, score, clustering, ','.join(keywords), title, text, best_sen]))
            except:
                yield ('\t'.join([rank, score, clustering, ','.join(keywords), title, text]))
        else:
            yield ('\t'.join([rank, score, clustering, ','.join(keywords), title, text]))

# Get accuracy using evaluate.py:
def accuracy(self, results, q, k, angular=True):
    precision = evaluate.evaluate_precision(self, results, angular=angular)
    recall = evaluate.evaluate_recall(self, results, q=q, k=k)
    print('Precision: %.3f' % precision)
    print('Recall: %.3f' % recall)
    print('F Score: %.3f' % (2*(precision*recall)/(precision+recall)))
