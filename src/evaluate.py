import os.path
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
    return statistics.mean([(1 - (cs**2)/2)**2 for cs, ai, lang in results])**0.5

# Half-L Recall
def _evaluate_recall(self, results, q, k):
    list1 = [(ai, lang) for cs, ai, lang in results]
    list2 = [(ai, lang) for cs, ai, lang in query.mulling_annoy_query(self, q, 'meta', k=k)]
    return len(set(list1).intersection(set(list2)))/len(list2)

def cossim(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

# Pipeline evaluator
class Evaluator:
    def __init__(self, inpath , k, L, MulLingObject, methods=[1,2,3,4]):
        if L == -1:
            L = k
        elif k> len(MulLingObject.langs)*L:
            raise ValueError('The number of search results cannot be displayed as L is too small!')
        
        self.k = k
        self.L = L
        self.outpath = 'dump/evaluator.csv'
        self.queries = list()
        if not os.path.isfile(inpath):
            raise IOError('File {} does not exist'.format(inpath))
        else:
            with open(inpath, encoding='utf-8') as f:
                self.n_queries = f.readline()[:-1]
                for line in f:
                    self.queries.append(line[:-1])
        method_list = ['baa','bai','laser','metalaser']
        self.methods = list(map(lambda x: method_list[x-1], methods))
        self.table = pd.DataFrame(columns=['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)] + ['Non-parametric (Friedman, Nemenyi)']).set_index('Queries')

        # Batch query for LASER
        bigtext = '\n'.join(self.queries)
        bigvects = LASER.get_vect(bigtext, lang='en')
        smallvects = list()
        for query_ in self.queries:
            smallvects.append(processing.vectorize_lang(MulLingObject, query_, 'en'))

        # Creating a dataframe for each query in the dataset
        self.data = dict()

        # Progress Tracking
        total = len(self.queries)
        progress = {total//5-1:20, 2*total//5-1:40, 3*total//5-1:60, 4*total//5-1:80, total-1:100}

        for index_, query_ in enumerate(self.queries):
            self.data[query_] = pd.DataFrame(columns=['Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)])
            try:
            # Querying relatively relevant articles using search by metadata sorted and normalized by top L
                results = query.mulling_annoy_query(MulLingObject, query_, 'meta', k=k, L=L, normalize_top_L=True, multilingual=False)
            except:
                print(query_)
            # Appending the relevant articles to the dataframe
            new_row = pd.Series(dict(zip(self.data[query_].columns, ['Title']+ [(ai, lang) for cs,ai, lang in results])))
            self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

            # Calculating cosine similarity between query and above articles
            for index, method_ in enumerate(self.methods):
                if method_ == 'baa':
                    name = 'BWE-Agg-Add'
                    vec = smallvects[index_]
                elif method_ == 'bai':
                    name = 'BWE-Agg-IDF'
                    vec = smallvects[index_]
                elif method_ == 'laser':
                    name = 'LASER'
                    vec = bigvects[index_]
                elif method_ == 'metalaser':
                    name = 'LASER-meta'
                    vec = bigvects[index_]
                results = list()
                for i in range(self.k):
                    ai, lang = self.data[query_]['Article{}/Ranking'.format(i+1)][0]
                    d_vecs = MulLingObject.docvecs[method_][lang].get_item_vector(ai)
                    results.append(cossim(vec, d_vecs))
                
                # Appending cosine similarities to the dataframe
                new_row = pd.Series(dict(zip(self.data[query_].columns, [name] + results)))
                self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

            self.data[query_]['Queries'] = query_

            # Progress Status
            if index_ in progress:
                print(progress[index_], '% of queries processed.')

    '''
    Using Friedman's non-parametric chi-squared test, we evaluate the performance of the models
    comparatively. First, we determine whether they are statistically likely to rank the relevant
    articles similarly as given by the p-value (p>0.05 suggests they are similarly ranked which
    verifies the assumption that these articles are relevant in all models).

    If the models are shown to rank the relevant articles similarly, we then conduct pairwise
    comparisons of the models using Nemenyi's tests which show how similar the rankings are. By
    taking the geometric mean of all Nemenyi test p-values of pairwise compared models, we
    effectively reward models that rank results similarly and punish those that rank them
    differently.

    Using these geometric means, we then generate an ensemble of values for the given query that
    would be representative to how well each model has ranked the articles. We then take the mean
    ensemble preserve it.
    '''
    def tabulate(self, ensemble=None):
        if not ensemble:
            self.ensemble = []
            for query_ in self.data:
                f_data = self.data[query_].iloc[1:(len(self.methods)+1),1:-1].values.tolist()
                p =friedmanchisquare(*f_data)[1]
                ph_data = scikit_posthocs.posthoc_nemenyi_friedman(np.array(f_data).T)
                ph_min = [math.sqrt(-1*np.prod(ph_data[i])) for i in range(len(ph_data[0]))]
                ensemble = [n/sum(ph_min) for n in ph_min]
                self.data[query_]['Non-parametric (Friedman, Nemenyi)'] = ['p=%.3f'%p] + ensemble
                self.table = self.table.append(self.data[query_])
                if p>=0.05:
                    self.ensemble.append(ensemble)
            self.table = self.table[['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(self.k)] + ['Non-parametric (Friedman, Nemenyi)']].set_index('Queries', append=True).swaplevel(0,1)
            self.ensembles = np.asarray(self.ensemble).T.tolist()
            self.print_results()
        else:
            self.ensemble = []
            for query_ in self.data:
                # Appending Ensemble model as new row to each query
                results = [sum( a*b for a,b in zip(ensemble, c)) for c in np.array(self.data[query_].iloc[1:(len(self.methods)+1),1:-2].values.tolist()).T]
                new_row = pd.Series(dict(zip(self.data[query_].columns, ['Ensemble-model'] + results + [query_, '0'])))
                self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

                f_data = self.data[query_].iloc[1:(len(self.methods)+2),1:-2].values.tolist()
                p =friedmanchisquare(*f_data)[1]
                ph_data = scikit_posthocs.posthoc_nemenyi_friedman(np.array(f_data).T)
                ph_min = [math.sqrt(-1*np.prod(ph_data[i])) for i in range(len(ph_data[0]))]
                ensemble = [n/sum(ph_min) for n in ph_min]
                self.data[query_]['Non-parametric (Friedman, Nemenyi) (New)'] = ['p=%.3f'%p] + ensemble
                self.table = self.table.append(self.data[query_])
                if p>=0.05:
                    self.ensemble.append(ensemble)
            self.table = self.table[['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(self.k)] + ['Non-parametric (Friedman, Nemenyi)']].set_index('Queries', append=True).swaplevel(0,1)
            self.ensembles = np.asarray(self.ensemble).T.tolist()
            self.print_results()

    # Print table of results and print ensemble 
    def print_results(self):
        display(self.table)
        print('Ensemble is %.3fx of BAA, %.3fx of BAI, %.3fx of LASER and %.3fx of meta LASERs' % (statistics.mean(self.ensembles[0]),statistics.mean(self.ensembles[1]),statistics.mean(self.ensembles[2]),statistics.mean(self.ensemble[3])))

    # Export the output table to csv
    def export_to_csv(self, outpath=''):
        self.table.to_csv(outpath, encoding='utf-8')


if __name__ == '__main__':
    pass

