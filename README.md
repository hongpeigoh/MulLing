# Multilingual Information Retrieval (MulLing)

**MulLing** is a Python library used to evaluate and enhance Information Retrieval Methods in a multilingual setting. Its objectives are:

- To explore state-of-the-art Multilingual Information Retrieval methods.
- To provide a meta-analysis of existing techniques for IR evaluation.
- To suggest strong alternatives in evaluating IR in a multilingual context for accuracy.

We include various add-ons and features that allow for implementation of augmented Multilingual IR methods.

We provide a detailed description of the performance of our IR evaluation in [1] and include an interactive Python notebook for trying our source code.

# Dependencies

We run on the following dependencies:

- **Python 3.7.4**, run on Anaconda 2019.10, with the following modules: **NumPy** 1.16.5, 
- [**FastText**](https://github.com/facebookresearch/fastText) for Multilingual word embeddings. (installed)
- [**SpaCy**](https://github.com/explosion/spaCy) for Natural Language Processing (NLP) toolkit in English (en).  
`pip install spacy && python -m spacy download en_core_web_md`
- [**jieba**](https://github.com/fxsjy/jieba) for NLP toolkit in Simplified-Chinese (zh).  
`pip install jieba`
- [**malaya**](https://github.com/huseinzol05/Malaya) for NLP toolkit in Bahasa Melayu (ms).  
`pip install malaya`
- [**HindiTokenizer**](https://github.com/taranjeet/hindi-tokenizer) for NLP tooklit in Indic Languages (hi, ta). (installed)
- **SciPy's KDTrees** for basic nearest neighbours search.
- [**googletrans**](https://pypi.org/project/googletrans/) as Machine Translation (MT) and Language detection toolkit.  
`pip install googletrans`

The data provided is built with:
- [**ConceptNet Numberbatch's**](https://github.com/commonsense/conceptnet-numberbatch) multilingual word embeddings. This data contains semantic vectors from ConceptNet Numberbatch, by Luminoso Technologies, Inc. You may redistribute or modify the data under the terms of the CC-By-SA 4.0 license.
- Web-crawled data-set from Singapore-based newspapers: **The Straits Times** (en), **Lianhe Zaobao** (zh), **Berita Harian** (ms) and **Tamil Murasu** (ta).

# Get pre-calculated evaluation datasets and monolingual word embeddings

We obtain aligned multilingual word embeddings from Conceptnet's Numberbatch ([https://github.com/commonsense/conceptnet-numberbatch](https://github.com/commonsense/conceptnet-numberbatch)). These are sorted in a singular multilingual vector space. A crawler will be necessary to process the file to extract tokens of a particular language. An edited code sample by Facebook's Fasttext team shown below may be used.

```
import gzip

class FastVector:
    def __init__(self, vector_file='', outpath = ''):
        self.word2id = {}
        self.id2word = []

        print('Reading word vectors from %s' % vector_file)
        with gzip.open(vector_file, 'rt', encoding="utf8") as f:
            (self.n_words, self.n_dim) = \
                (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.embed = np.zeros((_____, self.n_dim))          # number of tokens in language
            k=0
            for i, line in enumerate(f):
                elems = line.rstrip('\n').split(' ')
                if elems[0][3:5] == "_____":                     # language
                    self.word2id[elems[0][6:]] = k
                    self.embed[k] = elems[1:self.n_dim+1]
                    self.id2word.append(elems[0][6:])
                    k+=1
        self.export(outpath)
    
    def export(self, outpath):
        fout = open(outpath, "w", encoding="utf8")
        fout.write(str(_____) + " " + str(self.n_dim) + "\n")   # number of tokens in language
        for token in self.id2word:
            vector_components = ["%.6f" % number for number in self[token]]
            vector_as_string = " ".join(vector_components)

            out_line = token + " " + vector_as_string + "\n"
            fout.write(out_line)
        fout.close()

if __name__ == '__main__':
    lang_dictionary = FastVector(vector_file='numberbatch-19.08.txt.gz', outpath='wordvecs.txt')
```

If you wish to train an aligned monolingual word embedding yourself, please refer to MUSE's github to learn how Procrustes Alignment may be used effectively. Since this is a multilingual task and not a crosslingual task, Procrustes Refinment should be avoided (i.e. `--n_refinement 0`) as it would edit both the source and target space.

Should you wish to evaluate the current methods, the corpora used is provided below. A pre-calculated version of the test set is also included should the computation be too demanding for your system.

### Necessary Data

| Language           | Articles      | Word vectors  | Stopwords     |
| -------------      | ------------- | ------------- | ------------- |
| English (en)       | Link          | Link          | Link          |
| Mandarin (zh)      | Link          | Link          | Link          |
| Bahasa Melayu (ms) | Link          | Link          | Link          |
| Tamil (ta)         | Link          | Link          | Link          |

### Additional Data

| File                | Articles       |
| ----------------    | -------------- |
| Document Vectors    | Link           |
| LASER Vectors       | Link           |
| IDFs                | Link           |
| Doc Vectors (Title) | Link           |

# Pre-processing

Included in our built-in methods are an implementation of Litschko et. al's Bilingual Word Embeddings (Aggregate Addition and Aggregate IDF-weighted Addition), and Artexte and Schwenke's Language Agnostic SEntence Representations (LASER).

To install the docker container for LASER, please refer to their documentation on ([https://github.com/facebookresearch/LASER](https://github.com/facebookresearch/LASER)) and start the docker container on port 8050 (public): 80 (private).

We have implemented 4 Multilingual IR models in the above respository. They are labelled BWE-Agg-Add (BAA), BWE-Agg-IDF (BAI), LASER, LASER-meta. We have also a simple model using meta-data used for diagnosis.

### Bilingual Word Embeddings (BWE)

For **BWE-Agg-Add (BAA)**, words are represented as vectors through word embeddings using FastText's FastVectors, a Bag-of-Concepts model is employed to represent a document as a sum of all individual vectors. Similarity is embedded in the 300-dimension vector space in the form of nearest neighbours.

Similarly for **BWE-Agg-IDF (BAI)**, employing the Bag-of-Concepts model with FastVector word embeddings allow us to calculate the document vector as a weighted sum of all token vectors, where the weight is determined by its [Term-Frequeuncy × Inverse-Document-Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (TF-IDF). Similarity is also calculated through cosine similarity (nearest neighbours) on vectors in a 300-dimension vector space.

### LASER

Using Facebook's **LASER** LSTM model, sentences are embedded using a 1024-dimension vector. By calculating the document as a stream of sentences, we can use the model to find the closest representation of the document through the sentence model. Similarity here is calculated through cosine similarity in a 1024-dimension vector space

For **LASER-meta**, the document title is processed as the sentence instead of the full document to reduce computational load.

# Search methodology

Refer to the interactive Jupyter Notebook to see the implementation of relevant searches.

### **Monolingual Queries**

| Query                     | Function          | Input Arguments                                         |
| --------------------------| ------------------| --------------------------------------------------------|
| Brute Force (BAA, BAI)    | `simple_query`    | `self:MulLingObject`, `q:str`, `lang:str`, `k:int`      |
| Brute Force (BAA, BAI)    | `vec_query`       | `self:MulLingObject`, `q_vecs:list`, `lang:str`, `k:int`|
| KD-Tree (BAA, BAI)        | `kdtree_query`    | `self:MulLingObject`, `q:str`, `lang:str`, `k:int`      |
| KD-Tree (BAA, BAI)        | `vec_kdtree_query`| `self:MulLingObject`, `q_vecs:list`, `lang:str`, `k:int`|
| Metadata Brute Force (BAA)| `title_query`     | `self:MulLingObject`, `q_vecs:list`, `lang:str`, `k:int`|
| LASER                     | `laser_query`     | `self:MulLingObject`, `q:str`, `lang:str`, `k:int`      |

**Specifications:**  
`self` : Instanced MulLing Vector Object with loaded models
`q` : Input Query  
`q_vecs` : Input Query in vectors, calculated by `processing.vectorize(MulLingObject, input_=q)` or `LASER.get_vect(q=q)`
`lang` : Language to search in, currently implemented in English (`en`), Simplified Chinese (`zh`), Bahasa Melayu (`ms`) and Tamil (`ta`)  
`k` : Number of results to show  
**Returns:**  
A list of `k` 3-tuples, sorted in decreasing cosine similarity. Each tuple is given as `(cosine_similarity, article_index, language)`

### **Multilingual Queries**

| Query                                           | Function                      | Input Arguments                                                                       |
| ------------------------------------------------| ------------------------------| --------------------------------------------------------------------------------------|
| Un-normalized Brute Force (BAA, BAI)            | `simple_raw_score_merge_query`| `self:MulLingObject`, `q:str`, `lang:str`, `k:int`                                    |
| Brute Force/KD Tree with normalization (BAA/BAI)| `mulling_query`               | `self:MulLingObject`, `q:str`, `k:int`, `L:int`, `kdtree:bool`, `normalize_top_L:bool`|
| Brute Force with normalization (LASER)          | `laser_mulling_query`         | `self:MulLingObject`, `q:str`, `k:int`, `L:int`, `normalize_top_L:bool`               |

**Specifications:**  
`L` : Number of results allowable for each language. Must be more than `k/langs.len()`  
`kdtree` : Decides if the results are searched on SciPy's kd-trees instead of brute force  
`normalize_top_L` : Decides if results will be normalized by the average cosine similarity of the top L results for each language.  
**Returns:**  
A list of `k` 3-tuples, sorted in decreasing cosine similarity. Each tuple is given as `(cosine_similarity, article_index, language)`

# Evaluation methodology

To compare the models, the following two methods are included.

1. Non parametric method
2. Parametric method

For the non-parametric method, we can proceed without assumptions regarding the distribution of the corpora in the n-dimensional space. Using Friedman's non-parametric chi-squared test, we evaluate the performance of the models comparatively. First, we determine whether they are distributed similarly i.e. statistically likely to rank the relevant articles similarly as given by the p-value (p>0.05 suggests they are similarly ranked which verifies the assumption that these articles are relevant in all models). 
 
If the models are shown to rank the relevant articles similarly, we then conduct pairwise comparisons of the models using Nemenyi's tests which show how similar the rankings are. By taking the geometric mean of all Nemenyi test p-values of pairwise compared models, we effectively reward models that rank results similarly and punish those that rank them differently. 
 
Using these geometric means, we then generate an ensemble of values for the given query that would be representative to how well each model has ranked the articles. We then take the mean ensemble preserve it.

For the parametric method, we will first normalize the distribution of document vectors in the vector space and check whether it satisfies the assumption of having the same chi-squared distribution, which is necessary for parametric testing. Next, we will use the ensemble of models generated from the non-parametric method as a control/baseline to evaluate the models.

# Application

Provided in the repository is an [interactive Jupyter Notebook](Debug.ipynb) that will showcase some of the functionalities provided by the source code. These include but are not limited to:

1. Instancing the Model
2. Sample Queries
3. Evaluating the Model
4. Developing an Ensemble of Models after evaluation
5. Re-evaluating the Model using the baseline model
6. Refining the model.

# References