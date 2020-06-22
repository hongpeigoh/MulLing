import os.path
from statistics import mean
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

# Mean Square Cosine Similarity.
def evaluate_precision(self, results, angular=True):
    if angular:
        return mean([(1 - (result[0]**2)/2)**2 for result in results])**0.5
    else:
        return mean([(result[0]**2) for result in results])**0.5

# Half-L Recall
def evaluate_recall(self, results, q, k, L=-1):
    if L == -1:
        L = k//len(self.langs) + 1
    list1 = [(result[1], result[2]) for result in results]
    list2 = [(result[1], result[2]) for result in query.mulling_annoy_query(self, q, 'bai', k=len(self.langs)*k, L=len(self.langs)*L)]
    return len(set(list1).intersection(set(list2)))/len(list1)

def cossim(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

# Pipeline evaluator
class Evaluator:
    def __init__(self, inpath , k, L, MulLingObject, methods=[1,2,3,4,5,6]):
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
        method_list = ['baa','bai','laser','metalaser','senlaser','senbai']
        self.methods = list(map(lambda x: method_list[x-1], methods))
        self.table = pd.DataFrame(columns=['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(k)] + ['Non-parametric (Friedman, Nemenyi)']).set_index('Queries')

        # Batch query for LASER
        bigtext = '\n'.join(self.queries)
        bigvects = LASER.get_vect(bigtext, lang='en')
        smallvects = list()
        for query_ in self.queries:
            smallvects.append(processing.vectorize_lang(MulLingObject, query_, 'en', raise_error=False))

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
            new_row = pd.Series(dict(zip(self.data[query_].columns, ['Title']+ [(result[1], result[2]) for result in results])))
            self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

            # Calculating cosine similarity between query and above articles
            for index, method_ in enumerate(self.methods):
                if method_ == 'baa':
                    name = 'BWE-Agg-Add'
                elif method_ == 'bai':
                    name = 'BWE-Agg-IDF'
                elif method_ == 'laser':
                    name = 'LASER'
                elif method_ == 'metalaser':
                    name = 'LASER-meta'
                elif method_ == 'senlaser':
                    name = 'LASER-sentences'
                elif method_ == 'senbai':
                    name = 'BWE-Agg-IDF-sentence'
                
                vec = bigvects[index_] if 'laser' in method_ else smallvects[index_]
                results = list()
                for i in range(self.k):
                    ai, lang = self.data[query_]['Article{}/Ranking'.format(i+1)][0]
                    if 'sen' not in method_:
                        d_vecs = MulLingObject.docvecs[method_][lang].get_item_vector(ai)
                        results.append(cossim(vec, d_vecs))
                    else:
                        sis = [MulLingObject.s2d[method_][lang].index(ai)]
                        while MulLingObject.s2d[method_][lang][sis[-1]+1] == ai:
                            sis.append(sis[-1]+1)
                            if sis[-1]+1 == len(MulLingObject.s2d[method_][lang]):
                                break
                        results.append(max([cossim(vec, MulLingObject.docvecs[method_][lang].get_item_vector(si)) for si in sis]))
                
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
    def tabulate(self, ensemble=False):
        offset = 2 if ensemble else 1
        column_name = 'Non-parametric (Friedman, Nemenyi) (New)' if ensemble else 'Non-parametric (Friedman, Nemenyi)'
        self.weights = []
        for query_ in self.data:
            if ensemble:
                #  Appending Ensemble model as new row to each query
                results = [sum( a*b for a,b in zip(self.ensemble, c)) for c in np.array(self.data[query_].iloc[1:(len(self.methods)+1),1:-2].values.tolist()).T]
                new_row = pd.Series(dict(zip(self.data[query_].columns, ['Ensemble-model'] + results + [query_, '0'])))
                self.data[query_] = self.data[query_].append(new_row, ignore_index=True)

            f_data = self.data[query_].iloc[1:(len(self.methods)+offset),1:-1*offset].values.tolist()
            p =friedmanchisquare(*f_data)[1]
            ph_data = scikit_posthocs.posthoc_nemenyi_friedman(np.array(f_data).T)
            #ph_min = [math.sqrt(-1*np.prod(ph_data[i])) for i in range(len(ph_data[0]))]
            ph_min = [max(ph_data[i]) for i in range(len(ph_data[0]))]
            weights = [n/sum(ph_min) for n in ph_min]
            self.data[query_][column_name] = ['p=%f'%p] + weights
            self.table = self.table.append(self.data[query_])
            self.weights.append(weights)
        if ensemble:
            self.table = self.table[['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(self.k)] + ['Non-parametric (Friedman, Nemenyi)'] + ['Non-parametric (Friedman, Nemenyi) (New)']].set_index('Queries', append=True).swaplevel(0,1)
        else:
            self.table = self.table[['Queries', 'Model'] + ['Article{}/Ranking'.format(i+1) for i in range(self.k)] + ['Non-parametric (Friedman, Nemenyi)']].set_index('Queries', append=True).swaplevel(0,1)
        self.ensembles = np.asarray(self.weights).T.tolist()
        return self.print_results()

    # Print table of results and print ensemble 
    def print_results(self, ensemble=False, table=True):
        if table:
            display(self.table)
        if not ensemble:
            self.ensemble = [mean(self.ensembles[i]) for i in range(len(self.methods))]
            output = tuple([len(self.ensembles[0])] + [self.ensemble[i] for i in range(len(self.methods))])
            #print('Using %i valid results, the calculated ensemble is %.3fx of BAA, %.3fx of BAI, %.3fx of LASER, %.3fx of meta LASERs, %.3fx of sentence LASER and %.3f of sen BAI.' % output)
            return output
        else:
            self.ensemble = [mean(self.ensembles[i]) for i in range(len(self.methods)+1)]
            output = tuple([len(self.ensembles[0])] + [self.ensemble[i] for i in range(len(self.methods)+1)])
            #print('Using %i valid results, the calculated ensemble is %.3fx of BAA, %.3fx of BAI, %.3fx of LASER, %.3fx of meta LASERs, %.3fx of sentence LASER, %.3f of sen BAI and %.3f of the ensemble model' % output)
            if max(self.ensemble) == self.ensemble[-1]:
                print('Ensemble model is valid.')
            return output

    # Export the output table to csv
    def export_to_csv(self, outpath=''):
        self.table.to_csv(outpath, encoding='utf-8')


if __name__ == '__main__':
    pass

