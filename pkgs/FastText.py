import numpy as np
import gzip
import argparse
import os

class FastVector:
    def __init__(self, vector_file='', transform=None):
        self.word2id = {}
        self.id2word = []

        print('Reading word vectors from %s' % vector_file)
        with open(vector_file, 'r', encoding="utf8") as f:
            (self.n_words, self.n_dim) = \
                (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.embed = np.zeros((self.n_words, self.n_dim))
            for i, line in enumerate(f):
                elems = line.rstrip('\n').split(' ')
                self.word2id[elems[0]] = i
                self.embed[i] = elems[1:self.n_dim+1]
                self.id2word.append(elems[0])

        if transform is not None:
            print('Applying transformation to embedding')
            self.apply_transform(transform)

    def apply_transform(self, transform):
        transmat = np.loadtxt(transform) if isinstance(transform, str) else transform
        self.embed = np.matmul(self.embed, transmat)

    def export(self, outpath):

        fout = open(outpath, "w", encoding="utf8")

        fout.write(str(self.n_words) + " " + str(self.n_dim) + "\n")
        for token in self.id2word:
            vector_components = ["%.6f" % number for number in self[token]]
            vector_as_string = " ".join(vector_components)

            out_line = token + " " + vector_as_string + "\n"
            fout.write(out_line)

        fout.close()


    @classmethod
    def cosine_similarity(cls, vec_a, vec_b):
        """Compute cosine similarity between vec_a and vec_b"""
        return np.dot(vec_a, vec_b) / \
            (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def __contains__(self, key):
        return key in self.word2id

    def __getitem__(self, key):
        return self.embed[self.word2id[key]]

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

class FastVectorExport:
    def __init__(self, lang, outpath, vector_file=''):
        self.lang2vocabsize = {"fr":1388686,"la":855294,"es":651859,"de":594456,"it":557743,"en":516782,"ru":455325,"zh":307441,"fi":267307,"pt":262904,"ja":256648,"nl":190221,
            "bg":178508,"sv":167321,"pl":152949,"no":105689,"eo":96255,"th":95342,"sl":91134,"ms":90554,"cs":88613,"ca":87508,"ar":85325,"hu":74384,"se":67601,"sh":66746,
            "el":65905,"gl":59006,"da":57119,"fa":53984,"ro":51437,"tr":51308,"is":48639,"eu":44151,"ko":42106,"vi":39802,"ga":36988,"grc":36977,"uk":36851,"lv":36333,"he":33435,
            "mk":33370,"ka":32338,"hy":29844,"sk":29376,"lt":28826,"ast":28401,"mg":26865,"et":26525,"oc":26095,"fil":25088,"io":25004,"hsb":24852,"hi":23538,"te":22173,
            "be":22117,"fro":21249,"sq":20493,"mul":19376,"cy":18721,"xcl":18420,"az":17184,"kk":16979,"gd":16827,"af":16132,"fo":15973,"ang":15700,"ku":13804,"vo":12731,
            "ta":12690,"ur":12006,"sw":11150,"sa":11081,"nrf":10048,"non":8536,"gv":8425,"nv":8232,"rup":5107}
        assert lang in self.lang2vocabsize
        self.word2id = {}
        self.id2word = []

        print('Reading word vectors from %s' % vector_file)
        with gzip.open(vector_file, 'rt', encoding='utf-8') as f:
            (self.n_words, self.n_dim) = \
                (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.embed = np.zeros((self.lang2vocabsize[lang], self.n_dim))
            count = 0
            for line in f:
                if '/%s/'%lang in line[:7]:
                    elems = line.rstrip('\n').split(' ')
                    word = elems[0].partition('/%s/'%lang)[2]
                    self.word2id[word] = count
                    if count == self.lang2vocabsize:
                        np.append(self.embed, elems[1:self.n_dim+1])
                    else:
                        try:
                            self.embed[count] = elems[1:self.n_dim+1]
                        except:
                            print(elems)
                    self.id2word.append(word)
                    count += 1
            print('Expected Vocab Size: %i\nVocab Size: %i' % (self.lang2vocabsize[lang],count))
            self.export(outpath, lang)

    def export(self, outpath, lang):
        fout = open(outpath, "w", encoding="utf8")

        fout.write(str(self.lang2vocabsize[lang]) + " " + str(self.n_dim) + "\n")
        for token in self.id2word:
            vector_components = ["%.6f" % number for number in self[token]]
            vector_as_string = " ".join(vector_components)

            out_line = token + " " + vector_as_string + "\n"
            fout.write(out_line)

        fout.close()

    def __contains__(self, key):
        return key in self.word2id

    def __getitem__(self, key):
        return self.embed[self.word2id[key]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Export single language FastText word vectors for Numberbatch GZip file.')
    parser.add_argument(
        '--lang', '-L', type=str, default='en', help='Code of language to be exported')
    parser.add_argument(
        '--data_dir', type=str, default='dump',
        help='Base directory for Numberbatch word vectors and created files')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    if not os.path.exists('%s/%s' % (args.data_dir, args.lang)):
        os.mkdir('%s/%s' % (args.data_dir, args.lang))

    print('Exporting single language FastText word Vectors for %s' % args.lang)
    FastVectorExport(args.lang, '%s/%s/new_wordvecs.txt' % (args.data_dir, args.lang), '%s/numberbatch-19.08.txt.gz' % args.data_dir)
    print('Exported!')
    