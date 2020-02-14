import statistics
import math

# Get first item
def _getitem(tuple_):
    return tuple_[0]

# Get first item, the cosine similarity/ shortest distance (1st element), normalized by a factor of 1/lg(N) where N is the size of the language's document corpora to reduce 
def _getnormalizeditem(self, tuple_):
    return tuple_[0] / math.log(len(self.docs[tuple_[2]]))

# Get first item, the cosine similarity/ shortest distance (1st element), normalized by a factor of *lg(N) where N is the size of the language's document corpora to reduce
def _getdenormalizeditem(self, tuple_):
    return tuple_[0] * math.log(math.log(len(self.docs[tuple_[2]])))

# Get first item, normalized by the average cosine similarity of the top L results.
def _getnormalizedtopLitem(self, tuple_):
    return tuple_[0] / self.mean[tuple_[2]]

# Get article using the language (2nd element) and the article index (3rd element)
def _getarticle(self, tuple_):
    return self.docs[tuple_[2]][tuple_[1]]

# Get readable results of enumerated titles from a query:
def _getresults(self, query_results):
    for index, result in enumerate(query_results):
        print('%i. %s' % (index+1, self.docs[result[2]][result[1]][0]))
