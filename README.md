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
- [**malaya**](https://github.com/huseinzol05/Malaya) for NLP toolkit in Bahasa Melayu (ms). If certain tensorflow modules are not loading due to compatibility issues (e.g. beam_search_ops in malaya/_utils/utils.py), known fix is to remove the incriminating line of code.
`pip install malaya`
- [**HindiTokenizer**](https://github.com/taranjeet/hindi-tokenizer) for NLP tooklit in Indic Languages (hi, ta). (installed)
- [**NLTK**](https://www.nltk.org/) for NLP Toolkit and Sentence Tokenization. `pip install nltk && python -i` 
>import nltk  
>nltk.download('punkt')
- [**googletrans**](https://pypi.org/project/googletrans/) as Machine Translation (MT) and Language detection toolkit.  
`pip install googletrans`
- [**scikitposthocs**](https://pypi.org/project/scikit-posthocs/) for Post-hoc Analysis of Friedman Test. `pip install scikit-posthocs`]
- [**Annoy**](https://github.com/spotify/annoy) for Nearest Neighbours Search and Indexing `pip install annoy`
- [**Flask and Components**](https://pypi.org/project/Flask/) for creating web-server to host web-app. `pip install Flask flask_cors flask-wtf`

Sample command to download requirements
```
$ pip install spacy jieba malaya nltk googletrans scikit-posthocs annoy Flask flask_cors flask-wtf && python -m spacy download en_core_web_md && python -i
>>> import nltk
>>> nltk.download('punkt')
^Z
$ cat anaconda3/lib/site-packages/malaya/_utils/utils.py
7 | from tqdm import tqdm
8 | ---remove deprecated tensorflow beam_search_ops lib import---
9 | from functools import wraps
```

The data provided is built with:
- [**ConceptNet Numberbatch's**](https://github.com/commonsense/conceptnet-numberbatch) multilingual word embeddings. This data contains semantic vectors from ConceptNet Numberbatch, by Luminoso Technologies, Inc. You may redistribute or modify the data under the terms of the CC-By-SA 4.0 license.
- Web-crawled data-set from Singapore-based newspapers: **The Straits Times** (en), **Lianhe Zaobao** (zh), **Berita Harian** (ms) and **Tamil Murasu** (ta).

To run the notebook on the given* data-set, these requirements are necessary:
- \>8GB RAM  
- \>32 GB Disk Space  
- GPU to run LASER docker container  
- A lot of patience with my horrible code  

\* Author's note: These are not currently given

# Get pre-calculated evaluation datasets and monolingual word embeddings

We obtain aligned multilingual word embeddings from Conceptnet's Numberbatch ([https://github.com/commonsense/conceptnet-numberbatch](https://github.com/commonsense/conceptnet-numberbatch)). These are sorted in a singular vector space. A crawler will be necessary to process the file to extract tokens of a particular language. You may use our crawler on either our dashboard's sandbox or the code below.

```
$ cd MulLing
$ cd dump
$ wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz
$ cd ..
$ python pkgs/FastText.py --lang en --data_dir dump
```

If you wish to train an aligned monolingual word embedding yourself, please refer to MUSE's github to learn how Procrustes Alignment may be used effectively. Since this is a multilingual task and not a crosslingual task, Procrustes Refinment should be avoided (i.e. `--n_refinement 0`) as both the source and target space are being edited.

Should you wish to evaluate the current methods, the corpora used are provided below. A pre-calculated version of the test set is also included should the computation be too demanding for your system.

### Data

To install the data, please access the Google Cloud Bucket as shown. If your root folder is the MulLing directory, the MulLing path will be '.'.

```
# Set-up Google Cloud SDK and GS Util and log-in to service account through JSON object
$ pip install gcloud gsutil
$ curl -L -o mulling.json "https://docs.google.com/uc?export=download&id=1NtxO4I0aGH7asIWou_VfauogEfq4EdOu"
$ gcloud auth activate-service-account --key-file mulling.json

# To use the pre-processed data
$ gsutil -m cp -r gs://mulling/030420/dump $MULLING_PATH$

# To use only the base word vectors and iso-stopwords lists
$ gsutil -m cp -r gs://mulling/200420 $MULLING_PATH$
$ mv 200420 dump
```

# Pre-processing

Included in our built-in methods are Litschko et. al's Bilingual Word Embeddings (Aggregate Addition and Aggregate IDF-weighted Addition), and Artexte and Schwenke's Language Agnostic SEntence Representations (LASER). See below for greater documentation.

To install the docker container for LASER, please refer to their documentation on ([https://github.com/facebookresearch/LASER](https://github.com/facebookresearch/LASER)) and start the docker container on port 8050 (public): 80 (private).

Sample build bash
```
# Install docker, python 3, pip installer
$ sudo apt-get install docker.io python3 python3-pip

# Setting up Docker without root
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
$ newgrp docker

$ git clone https://www.github.com/facebookresearch/LASER
$ cd LASER
$ docker build --tag=laser docker
$ docker run -it laser
$ docker system prune
$ docker run -p 8050:80 -it laser python app.py
```

This builds the LASER container on http://localhost:8050 (Linux/ WSL/ Mac OS/ Docker for Windows) or http://192.168.99.100:8050 (Docker Toolbox). It uses Faiss-CPU for Nearest Neighbours Search. To use CUDA toolkit and FAISS-GPU, see the following bash commands

```
# Get CUDA key for authenticating download
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Downlaod CUDA deb
$ wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/$ 7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda

# Download Anaconda/Miniconda
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && bash Anaconda3-2020.02-Linux-x86_64.sh

# Download PyTorch and Faiss-GPU
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
$ conda install faiss-gpu cudatoolkit=10.1 -c pytorch
```

## Models

Implemented in the respository are 7 Multilingual IR models.
1. BWE-Agg-Add (`'baa'`)
2. BWE-Agg-IDF (`'bai'`)
3. BWE-Agg-Add on meta-data/titles (`'meta'`)
4. BWE-Agg-IDF on individual sentences of articles (`'senbai'`)
5. LASER (`'laser'`)
6. LASER on meta-data/titles (`'metalasers'`)
7. LASER on individual sentences of articles (`'senlaser'`)

### Bilingual Word Embeddings (BWE)

For **BWE-Agg-Add (BAA)**, words are represented as vectors through word embeddings using FastText's FastVectors, a Bag-of-Concepts model is employed to represent a document as a sum of all individual vectors. Similarity is embedded in the 300-dimension vector space in the form of nearest neighbours.

Similarly for **BWE-Agg-IDF (BAI)**, employing the Bag-of-Concepts model with FastVector word embeddings allow us to calculate the document vector as a weighted sum of all token vectors, where the weight is determined by its [Term-Frequeuncy × Inverse-Document-Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (TF-IDF). Similarity is also calculated through angular distance (nearest neighbours) between vectors in a 300-dimension vector space.

### LASER

Using Facebook's **LASER** Bi-LSTM model, sentences/documents/phrases are embedded as a 1024-dimension vector. By calculating the document as a stream of sentences, the model will be used to find the closest representation of the document through the sentence model. Similarity here is calculated through angular distance between vectors in a 1024-dimension vector space.

# Training on your own data

To train on your own corpora, save each repository as an `articles.pkl` under the respective `language` folder with the respective Numberbatch wordvecs and iso-stopwords (except English).

Sample directory
```
MulLing
 L dump
    L en
       L articles.pkl
       L wordvecs.txt
    L zh
       L articles.pkl
       L wordvecs.txt
       L stopwords.txt
    L ms
       L ...
    L ta
       L ...
```
The articles pickle dump should be saved in the following format. The tuple can contain additional XML data but is optional and not currently used:
```
> import pickle
> with open('xxx') as f:
>   ...
> articles = [(title, text) for (title, text) in articles]
> pickle.dump(articles, open('./dump/xx/articles.pkl', 'wb'))
```
To train your data, start up a Jupyter Notebook or use the given .ipynb.

```
In [1]: from mulling import MulLingVectorsAnnoy
        print('Imported')

Out[1]: Imported!

In [2]: langs = ['en','zh','ms','ta']
        methods = ['baa','bai','meta','laser','metalaser','senlaser','senbai']
        my_object = MulLingVectorsAnnoy(methods=methods, langs=langs)

Out[2]: ...
```


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
- A list of `k` 3-tuples, sorted in decreasing cosine similarity. Each tuple is given as `(angular_distance, article_index, language)`

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
- A list of `k` 3-tuples, sorted in decreasing cosine similarity. Each tuple is given as `(angular_distance, article_index, language)`

# Evaluation methodology

## Statistical Method

Using a non-parametric statistical evaluation of the models, an ensemble will be learnt from a data-set of given queries. There are two phases.

1. Comparative evaluation
2. Verification

For the comparative evaluation, no assumptions regarding the distribution of the corpora in the n-dimensional space are necessary. We can also assume that all models generate decent results and return true precise results and that our base model to be excluded in final evaluation has good recall. Due to overfitting, we will also exclude models similar to base model.

The data we are using will be the top 500 most searched news article titles as unlabelled queries. After excluding several overly short queries, we have 503 queries (`:thinking:`).

Using Friedman's non-parametric chi-squared test, the performance of the models can be evaluated comparatively. First, the baseline model is used to extract generally relevant results for testing, `L` results per language and `k=4L` total results. Each model is then tasked to predict the similarity (angular distance) between the query vector and the document vector. Under the test assumptions of the Friedman's test, that

1. Measured by at least three methods
2. Random sample from population of relevant articles
3. Scoring is continuous
4. No need for normal distribution

The conditions of testing the 5 models (except for the `meta` model as baseline and `metalaser` which is too similar) are met and can be tested under the Friedman test for similar distributions i.e. statistically likely to rank the relevant articles similarly as given by the p-value (high p value suggests they are similarly ranked which verifies the assumption that these articles are relevant in all models). As the number of methods increase, the p value drops exponentially when different models rank them even slightly differently. As such, we will conduct a post hoc test to determine the pairwise p-value instead.
 
If the models are shown to rank the relevant articles similarly, pairwise comparisons of the models are conducted using Nemenyi's tests which show how similar the rankings are. By taking the geometric mean of all Nemenyi test p-values of pairwise compared models for each model, this creates a reward-punishment model that effectively reward models that rank results similarly and punish those that rank them differently (Regression). 
 
Using these geometric means, an ensemble of values is generated for the given query that would be representative to how well each model has ranked the articles. The mean ensemble is taken as the new baseline model. To verify the ensemble, we run the same comparison test, including the ensemble model as a new model. If it is shown statistically that the new geometric means from the reward-punishment model prefer the ensemble, it will be verified as a effective model.

### Results

**Parameters:**
- Base model: `meta`
- k: `24`, L: `6`
- Excluded models: `metalaser` due to similarity to baseline model
- Included models: `baa`, `bai`, `laser`, `senlaser`, `senbai`

| Model  | baa | bai | laser | senlaser | senbai |
|--------|-----|-----|-------|----------|--------|
|Ensemble|0.245|0.253| 0.220 | 0.031    | 0.252  |

### Conclusion

We make a few generalisations about the models here. Sentence LASER model performs poorly in this task and TF-IDF methods generally performs slightly better than vector addition methods. LASER methods perform marginally worse than BoW models. However, there are some concerns with this method of evaluation, namely that a meta-evaluation is not a full representation of true performance and serves to show the outlier models (i.e. senlaser). We can use several implications here in the next evaluation methodology.

## Evaluation with labelled data

Using the languages I know well, i.e. English and Chinese, I created a small dataset of manually tagged data for each of the language pairs, `en-en`, `en-zh`, `zh-en`, `zh-zh`. We can use this to hypertune parameters to improve the models subsequently and perhaps consider zero-shot cross lingual evaluation for Malay (`ms`) and Tamil (`ta`).

We test each model on its recall of the document tagged to the query for varying degrees of retrieved documents `k`. Repeating several times, this is averaged out for each query to determine average precision and across the set of all queries to receive the MAP score.

### Results
*The queries were tagged mainly as a summary of the article with huge reference to the article title especially for the same language pairs and is hence significantly overfitted.

**en-en**

| Model | K = 1 | K = 5 | K = 10 | K = 50 | K = 100 |
|-------|-------|-------|--------|--------|---------|
| baa | 0.19 | 0.24 | 0.27 | 0.31 | 0.34 | 
| meta* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 
| bai | 0.22 | 0.26 | 0.27 | 0.33 | 0.33 | 
| senbai | 0.15 | 0.18 | 0.23 | 0.33 | 0.35 | 
| laser | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 
| metalaser* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 
| senlaser | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 

**en-zh**

| Model | K = 1 | K = 5 | K = 10 | K = 50 | K = 100 |
|-------|-------|-------|--------|--------|---------|
| baa | 0.09 | 0.13 | 0.14 | 0.21 | 0.25 | 
| meta* | 0.39 | 0.46 | 0.47 | 0.48 | 0.51 | 
| bai | 0.11 | 0.13 | 0.17 | 0.25 | 0.28 | 
| senbai | 0.10 | 0.15 | 0.17 | 0.22 | 0.22 | 
| laser | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 
| metalaser* | 0.20 | 0.23 | 0.23 | 0.23 | 0.24 | 
| senlaser | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 


**zh-en**

| Model | K = 1 | K = 5 | K = 10 | K = 50 | K = 100 |
|-------|-------|-------|--------|--------|---------|
| baa | 0.10 | 0.15 | 0.18 | 0.29 | 0.32 | 
| meta* | 0.55 | 0.57 | 0.58 | 0.60 | 0.61 | 
| bai | 0.12 | 0.19 | 0.21 | 0.32 | 0.34 | 
| senbai | 0.03 | 0.12 | 0.14 | 0.20 | 0.24 | 
| laser | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 
| metalaser* | 0.18 | 0.20 | 0.20 | 0.20 | 0.20 | 
| senlaser | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 | 


**zh-zh**

| Model | K = 1 | K = 5 | K = 10 | K = 50 | K = 100 |
|-------|-------|-------|--------|--------|---------|
| baa | 0.31 | 0.33 | 0.37 | 0.42 | 0.43 | 
| meta* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 
| bai | 0.31 | 0.36 | 0.38 | 0.42 | 0.45 | 
| senbai | 0.29 | 0.34 | 0.41 | 0.46 | 0.49 | 
| laser | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 
| metalaser* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 
| senlaser | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 

**Graphs**

![by language](./dump/results1.png)
![by model](./dump/results2.png)

### Conclusion

The results from the first method are mostly corroborated. In the task of information retrieval, using a Bag of Words/Concepts model is preferable as semantics and pedantics are often lost between documents, sentences, titles and queries. As such, the Bi-LSTM models can seem ill-fitted for general IR tasks. However, in practice, it is able to search for exactly fitted sentences with a given threshold very well due to the sparse-ness of the vector space.

The Bag of Word models perform better when done across the whole document to preserve its general idea and meaning as well as the same keywords. Moreover, the TF-IDF models offer slightly better performance for filtering out common words and rare keywords.

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
