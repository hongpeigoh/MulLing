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

We obtain aligned multilingual word embeddings from Conceptnet's Numberbatch ([https://github.com/commonsense/conceptnet-numberbatch](https://github.com/commonsense/conceptnet-numberbatch)). These are sorted in a singular vector space. A crawler will be necessary to process the file to extract tokens of a particular language. An edited code sample by Facebook's Fasttext team shown below may be used.

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
            self.embed = np.zeros((_____, self.n_dim))
            k=0
            for i, line in enumerate(f):
                elems = line.rstrip('\n').split(' ')
                if elems[0][3:5] == "ta":
                    self.word2id[elems[0][6:]] = k
                    self.embed[k] = elems[1:self.n_dim+1]
                    self.id2word.append(elems[0][6:])
                    k+=1
        self.export(outpath)
    
    def export(self, outpath):
        fout = open(outpath, "w", encoding="utf8")
        fout.write(str(_____) + " " + str(self.n_dim) + "\n")
        for token in self.id2word:
            vector_components = ["%.6f" % number for number in self[token]]
            vector_as_string = " ".join(vector_components)

            out_line = token + " " + vector_as_string + "\n"
            fout.write(out_line)
        fout.close()

if __name__ == '__main__':
    lang_dictionary = FastVector(vector_file='numberbatch-19.08.txt.gz', outpath='wordvecs.txt')
```

If you wish to train an aligned monolingual word embedding yourself, please refer to MUSE's github to learn how Procrustes Alignment may be used effectively. Since this is a multilingual task and not a crosslingual task, Procrustes Refinment should be avoided (i.e. `--n_refinement 0`) as both the source and target space are being edited.

Should you wish to evaluate the current methods, the corpora used are provided below. A pre-calculated version of the test set is also included should the computation be too demanding for your system.

### Necessary Data

| Language           | Articles      | Word vectors  | Stopwords     |
| -------------      | ------------- | ------------- | ------------- |
| English (en)       | Link          | Link          | Link          |
| Mandarin (zh)      | Link          | Link          | Link          |
| Bahasa Melayu (ms) | Link          | Link          | Link          |
| Tamil (ta)         | Link          | Link          | Link          |

### Additional Data

| File                | Articles                                     |
| ----------------    | -------------------------------------------- |
| Document Vectors    | Link                                         |
| LASER Vectors       | Link                                         |
| IDFs                | Link                                         |
| Doc Vectors (Title) | Link                                         |

# Pre-processing

Included in our built-in methods are Litschko et. al's Bilingual Word Embeddings (Aggregate Addition and Aggregate IDF-weighted Addition), and Artexte and Schwenke's Language Agnostic SEntence Representations (LASER).

To install the docker container for LASER, please refer to their documentation on ([https://github.com/facebookresearch/LASER](https://github.com/facebookresearch/LASER)) and start the docker container on port 8050 (public): 80 (private).

# Evaluation methodology

# Application