# Multilingual Information Retrieval (MulLing)

**MulLing** is a Python library used to evaluate and enhance Information Retrieval Methods in a multilingual setting. Its objectives are:

- To explore state-of-the-art Multilingual Information Retrieval methods.
- To provide a meta-analysis of existing techniques for IR evaluation.
- To suggest strong alternatives in evaluating IR in a multilingual context for accuracy.

Included in this repository are various add-ons and features that allow for implementation of augmented Multilingual IR methods. Detailed descriptions of the performance of our IR evaluation in [1] and an interactive Python notebook are also included.

# Dependencies and Requirements

MulLing runs on the following dependencies:

- **Python 3.7.4**, run on Anaconda 2019.10, with **NumPy** 1.16.5, 
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
- [**annoy**](https://github.com/spotify/annoy) for similarity search index.  
`pip install annoy`


The data provided is built with:
- [**ConceptNet Numberbatch's**](https://github.com/commonsense/conceptnet-numberbatch) multilingual word embeddings. This data contains semantic vectors from ConceptNet Numberbatch, by Luminoso Technologies, Inc. You may redistribute or modify the data under the terms of the CC-By-SA 4.0 license.
- Web-crawled data-set from Singapore-based newspapers: **The Straits Times** (en), **Lianhe Zaobao** (zh), **Berita Harian** (ms) and **Tamil Murasu** (ta).  

To run the notebook on the given* data-set, these requirements are necessary:
- 32GB RAM  
- \>32 GB Disk Space  
- GPU to run LASER docker container  
- A lot of patience with my horrible code  

\* Author's note: These are not currently given

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

### Necessary Data (to be added)

| Language      | Articles | Word vectors | Stopwords | Doc-vecs (BAA) | Doc-vecs (BAI) | Doc-vecs (LASER) | Doc-vecs (meta-LASER) |
| ------------- | -------- | ------------ | --------- | -------------- | -------------- | ---------------- | --------------------- |
| English (en)  | Link     | Link         | N.A.      | Link           | Link           | Link             | Link                  |
| Mandarin (zh) | Link     | Link         | Link      | Link           | Link           | Link             | Link                  |
| Malay (ms)    | Link     | Link         | Link      | Link           | Link           | Link             | Link                  |
| Tamil (ta)    | Link     | Link         | Link      | Link           | Link           | Link             | Link                  |

# Pre-processing

Included in our built-in methods are an implementation of Litschko et. al's Bilingual Word Embeddings (Aggregate Addition and Aggregate IDF-weighted Addition), and Artexte and Schwenke's Language Agnostic SEntence Representations (LASER) used on document representation and sentence (title) representation.

To install the docker container for LASER, please refer to their documentation on ([https://github.com/facebookresearch/LASER](https://github.com/facebookresearch/LASER)) and start the docker container on port 8050 (public): 80 (private).

Implemented in the above respository are 4 Multilingual IR models. They are labelled BWE-Agg-Add (`'baa'`), BWE-Agg-IDF (`'bai'`), LASER(`'lasers'`), LASER-meta(`'metalasers'`). A simple model using meta-data has also been included for diagnosis.

### Bilingual Word Embeddings (BWE)

For **BWE-Agg-Add (BAA)**, words are represented as vectors through word embeddings using FastText's FastVectors, a Bag-of-Concepts model is employed to represent a document as a sum of all individual vectors. Similarity is embedded in the 300-dimension vector space in the form of nearest neighbours.

Similarly for **BWE-Agg-IDF (BAI)**, employing the Bag-of-Concepts model with FastVector word embeddings allow us to calculate the document vector as a weighted sum of all token vectors, where the weight is determined by its [Term-Frequeuncy × Inverse-Document-Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (TF-IDF). Similarity is also calculated through cosine similarity (nearest neighbours) on vectors in a 300-dimension vector space.

### LASER

Using Facebook's **LASER** LSTM model, sentences are embedded using a 1024-dimension vector. By calculating the document as a stream of sentences, the model will be used to find the closest representation of the document through the sentence model. Similarity here is calculated through cosine similarity in a 1024-dimension vector space

For **LASER-meta**, the document title is processed as the sentence instead of the full document to reduce computational load.

# Search methodology

Refer to the interactive Jupyter Notebook to see the implementation of relevant searches.

### **Monolingual Queries**

| Query                     | Function                 | Input Arguments                                               |
| ------------------------- | ------------------------ | ------------------------------------------------------------- |
| Monolingual Query (annoy) | `monolingual_annoy_query`| `self:MulLingObject`, `q:str`, `model:str` `lang:str`, `k:int`|

**Specifications:**  
- `self` : Instanced MulLing Vector Object with loaded models  
- `q` : Input Query  
- `model` : Model to choose from: Bilingual Word Embeddings Aggregate Addition (`baa`) or TF-IDF (`bai`), LASER Document Representation (`laser`), LASER Title Representation (`metalaser`)
- `lang` : Language to search in, currently implemented in English (`en`), Simplified Chinese (`zh`), Bahasa Melayu (`ms`) and Tamil (`ta`)  
- `k` : Number of results to show  

**Returns:**  
- A list of `k` 3-tuples, sorted in decreasing cosine similarity. Each tuple is given as `(cosine_similarity, article_index, language)`

### **Multilingual Queries**

| Query                       | Function             | Input Arguments                              |
| --------------------------- | -------------------- | ---------------------------------------------|
| Multilingual Query (annoy)  | `mulling_annoy_query`| `self:MulLingObject`, `q:str`, `lang:str`, `k:int`, `L:int`, `normalize_top_L:bool`, `multilingual:bool` |
| Ensemble Query (brute-force)| `ensemble_query`     | `self:MulLingObject`, `ensemble:list`, `q:str`, `lang:str`, `k:int`, `L:int`, `normalize_top_L:bool`, `multilingual:bool` |

**Specifications:**  
- `ensemble` : Weights for each model (`baa`,`bai`,`laser`,`metalaser` respectively)  
- `L` : Number of maximum results from each language. Must be more than `k/langs.len()`  
- `normalize_top_L (default:True)` : Decides if results will be normalized by the average cosine similarity of the top L results for each language.  
- `multilingual (default:True)` : Decides if the query is in multiple languages  

**Returns:**  
- A list of `k` 3-tuples, sorted in decreasing cosine similarity. Each tuple is given as `(cosine_similarity, article_index, language)`

# Evaluation methodology

Using a non-parametric statistical evaluation of the models, an ensemble will be learnt from a data-set of given queries. There are two phases.

1. Comparative evaluation
2. Verification

For the comparative evaluation, no assumptions regarding the distribution of the corpora in the n-dimensional space are necessary. Using Friedman's non-parametric chi-squared test, the performance of the models can be evaluated comparatively. First, the models are tested for similar distributions i.e. statistically likely to rank the relevant articles similarly as given by the p-value (p>0.05 suggests they are similarly ranked which verifies the assumption that these articles are relevant in all models). 
 
If the models are shown to rank the relevant articles similarly, pairwise comparisons of the models are conducted using Nemenyi's tests which show how similar the rankings are. By taking the geometric mean of all Nemenyi test p-values of pairwise compared models for each model, this creates a reward-punishment model that effectively reward models that rank results similarly and punish those that rank them differently (Regression). 
 
Using these geometric means, an ensemble of values is generated for the given query that would be representative to how well each model has ranked the articles. The mean ensemble is taken as the new baseline model.

To verify the ensemble, we run the same comparison test, includin the ensemble model as a new model. If it is shown statistically that the new geometric means from the reward-punishment model prefer the ensemble, it will be verified as a effective model.

# Application

Provided in the repository is an [interactive Jupyter Notebook](Debug.ipynb) that will showcase some of the functionalities provided by the source code. These include but are not limited to:

1. Instancing the Model
2. Sample Queries
3. Evaluating the Model
4. Developing an Ensemble of Models after evaluation
5. Verifying the baseline ensemble model
6. Refining the model.
7. Testing the ensemble model.

# References

### Introduction
1. Klavans, J. and E. Hovy. (editor) 1999. [Multilingual Information Management](https://www.cs.cmu.edu/~ref/mlim/chapter2.html)  
2. Grefenstette, G. (editor) 1998. Cross-Language Information Retrieval.  

3. Sorg, P. and P. Cimiano. 2012. [Exploiting Wikipedia for cross-lingual and multilingual information retrieval](https://www.researchgate.net/publication/257026046_Exploiting_Wikipedia_for_cross-lingual_and_multilingual_information_retrieval)
4. Gabrilovich, E., S. Markovitch. 2007. Computing semantic relatedness using wikipedia-based explicit semantic analysis, Proceedings of the 20th International Joint Conference on Artificial Intelligence (IJCAI), pp. 1606–1611.

### Methods in Mulitilingual Information Retrieval

5. Bojanowski, P.\*, E. Grave\*, A. Joulin, T. Mikolov. 2017. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)  

6. Tsai, M.-F., H.-H. Chen, Y.-T., Wang. 2010. [Learning a merge model for multilingual information retrieval](https://www.researchgate.net/publication/220229367_Learning_a_merge_model_for_multilingual_information_retrieval)
7. Lin, W.-C. and H.-H. Chen. 2003. [Merging mechanisms in multilingual information retrieval. Lecture Notes in Computer Science, LNCS, 2785, 175–186](https://link.springer.com/chapter/10.1007/978-3-540-45237-9_14)
8. Rahimi, R., A. Shakery, I. King. 2015. [Multilingual information retrieval in the language
modeling framework](https://dl.acm.org/doi/10.1007/s10791-015-9255-1)

9. Litschko, R., G. Glavaš, S. Ponzetto, I. Vulić. 2018. [Unsupervised Cross-Lingual Information Retrieval
using Monolingual Data Only](https://arxiv.org/pdf/1805.00879.pdf)

10. Conneau*, A., G. Lample*, M. Ranzato, L. Denoyer, H. Jégou. 2018. [Word Translation without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf)

11. Artexte, M., H. Schwenk. 2019. [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/pdf/1812.10464.pdf)


### Performance Evaluation
12. Korra, R., P. Sujatha, S. Chetana, M. Kumar. 2011. [Performance Evaluation of Multilingual Information
Retrieval (MLIR) System over Information Retrieval
(IR) System](https://www.researchgate.net/publication/233916375_Performance_Evaluation_of_Multilingual_Information_Retrieval_MLIR_System_over_Information_Retrieval_IR_System)
13. Dasdan, A., K. Tsioutsiouliklis, E. Velipasaoglu. 2009. [Web Search Engine Metrics for Measuring User Satisfaction, Tutorial @ 18th International World Wide Web Conference.](http://www.dasdan.net/ali/www2009/web-search-metrics-tutorial-www09-parts0-5.pdf)


### Related Work:  
Manning, C., P. Raghavan, H. Schütze. 2009. [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)  
Speer R., J. Chin, C. Havasi. 2017. [ConceptNet 5.5: An Open Multilingual Graph of General Knowledge, in proceedings of AAAI 2017.](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972)


### What I used from references

Klavans (1999): 
- Highlights technical issues and expected bottlenecks for MLIR, raised by Grefensette (1998) and Klavans (1999).  

Sorg (2012):  
- Introduces the Bag-of-Concepts model in interlingual concept space as an expansion of traditional Bag-of-Words in Monolingual IR adapted from Gabrilovich and Markovitch's (2007) work on explicit semantic analysis.  
- Differentiates CLIR from MLIR, with the latter being significantly more complex due to the numerous target languages (compared to 1).  
- Qualifies two biases, here by known as Query Bias (a tendency to return documents in the query language) and Quantity Bias (a tendency to preferentially rank documents from the largest collection).  
- Exemplifies concerns of mistake propagation due to the multi-step procedure of IR in a multilingual setting

Bojanowski and Grave (2017):
 nil

Tsai, Chen and Wang (2010):
- Exemplifies and augments current methods of merging monolingual results lists into a multilingual results list. Here, we incorporate methods of normalizing by top-k proposed in Lin, Chen (2003) to implement our multilingual list, which is an improvement over raw score merging, round-robin merging (interleaving results by respective ranks) and top-1 normalizing (dividing by score of best result). This will be shown to reduce both query and quantity biases. In the normalizing by top-k method, the scores are divided by the average of the top-k ranking documents of each monolingual list to return a more representatively sorted list of documents. (essentially a top-2l thing)

Litschko et. al (2018):
- Provides the basis for Bilingual Word Embeddings as a form of vectorising query and documents in a vector space. BWE-Agg-Add and BWE-Agg-IDF as main modes of vectorising documents. Uses MAP to evaluate precision.

Conneau and Lample (2018):
- Denotes and shows how word embeddings in a multilingual vector space can be trained and provides implementation to train by oneself.

Artexte and Schwenk (2019):
- Uses sentence level embeddings which is implemented as an alternate document vectorizer.

Korra, Chetana and Kumar (2011):
- Provides Average Precision (AP) and Mean Average Precision (MAP) as competitive indicators to evaluate precision for a information retrieval task in a Multilingual Setting. 
