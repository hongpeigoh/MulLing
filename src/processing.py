import numpy as np
import string
import codecs
import re
import spacy
import jieba as nlp_zh
import malaya as nlp_ms
from googletrans import Translator
from pkgs.HindiTokenizer import Tokenizer as nlp_ta
gtrans = Translator()
nlp_en = spacy.load("en_core_web_md")

# Load stopwords for NLP modules
stopwords = dict()
langs = ['en','zh','ms','ta']
for lang in langs:
    if lang == 'zh':
        f = codecs.open('dump/{}/stopwords.txt'.format(lang), encoding='utf-8')    # https://github.com/stopwords-iso/stopwords-zh
        stopwords[lang] = [line.rstrip('\r\n') for line in f][1:]
        f.close()
    elif lang == 'ms':
        f = codecs.open('dump/{}/stopwords.txt'.format(lang), encoding='utf-8')    # https://github.com/stopwords-iso/stopwords-ms
        stopwords[lang] = [repr(line)[1:-5] for line in f][1:]
        f.close()
    elif lang == 'ta':
        f = codecs.open('dump/{}/stopwords.txt'.format(lang), encoding='utf-8')    # https://github.com/AshokR/TamilNLP/blob/master/Resources/TamilStopWords.txt
        stopwords[lang] = [repr(line)[1:-5] for line in f][1:]
        f.close()

# Tokenizes Input
def tokenize(lang, input, include_stopwords=False):
    """
    Tokenizes input with given language
    --------------------
    lang: str
    Target input language, selected from one of langs
    --------------------
    input: str
    Target input phrase/query/document
    --------------------
    include_stopwords: bool
    Returns stopword tokens if True
    --------------------
    Returns:
    <generator> object of all tokens that can be embedding into a list for better manipulation
    """
    if lang == 'en':
        d = nlp_en(input.replace('\n', ' '))
        if not include_stopwords:
            d_tokens = filter(lambda x : (x.is_stop == False and x.is_punct == False and x.pos_!='NUM'), d)
        else:
            d_tokens = filter(lambda x : (x.is_punct == False and x.pos_!='NUM'), d)
        return map(lambda x: str(x).lower(), d_tokens)
    else:
        if lang == 'zh':
            d_tokens = nlp_zh.cut(input, cut_all=False)
        elif lang == 'ms':
            d_tokens = [str(x).lower() for x in nlp_ms.preprocessing.SocialTokenizer().tokenize(input)]
        elif lang == 'ta':
            d = nlp_ta(input)
            d.tokenize()
            d_tokens = d.tokens

        if not include_stopwords:
            return filter(lambda x: (x not in stopwords[lang] and x not in string.punctuation), d_tokens)
        else:
            return filter(lambda x: (x not in string.punctuation), d_tokens)

def vectorize(self, input_, include_stopwords=True):
    """
    Vectorizes input with NO given language
    --------------------
    input_: str
    Target input phrase/query/document
    --------------------
    include_stopwords: bool
    Returns stopword tokens if True
    --------------------
    Returns:
    300 dimension word vector
    """
    Q, tokens, token_vecs = dict(), dict(), list()
    # To tokenize and add spaces to Chinese text in multilingual query
    for word in input_.split():
        if word != ' ':
            lang = gtrans.detect(word).lang[:2].replace('id','ms')
            if lang in self.langs:
                if lang in Q:
                    Q[lang].append(word)
                else:
                    Q[lang] = [word]
            else:
                word_ = gtrans.translate(word, dest='en').text
                if self.langs[0] in Q:
                    Q[self.langs[0]].append(word_)
                else:
                    Q[self.langs[0]] = [word_]
    
    # To vectorise the query by using the language of each query token. Vectors are then added to form the query vector
    for lang in Q:
        tokens[lang] = list(tokenize(lang, ' '.join(Q[lang]), include_stopwords=include_stopwords))
        for token in tokens[lang]:
            try:
                token_vecs.append(self.vecs[lang][token])
            except:
                raise KeyError('%s cannot be found in %s dictionary' % (token, lang))
    return sum(np.array(vec) for vec in token_vecs)

def vectorize_lang(self, input_, lang, include_stopwords=True, raise_error=True):
    """
    Vectorizes input with given language
    --------------------
    input_: str
    Target input phrase/query/document
    --------------------
    lang: str
    Target input language, selected from one of langs
    --------------------
    include_stopwords: bool
    Returns stopword tokens if True
    --------------------
    raise_error: bool
    Raises Key Error if corpora contains words not in Fasttext word vectors dictionary.
    Preferably False for calculation and app version and True for debugging.
    --------------------
    Returns:
    300 dimension word vector
    """
    tokens = tokenize(lang, input_, include_stopwords=include_stopwords)
    tokens_vecs = []
    for token in list(tokens):
        try:
            tokens_vecs.append(self.vecs[lang][token])
        except:
            if raise_error:
                raise KeyError('%s cannot be found in dictionary' % (token))
            else:
                pass
    return sum(np.array(vec) for vec in tokens_vecs)

def zh_sent_tokenize(paragraph):
    """
    Simple module to tokenize sentences in Chinese.
    --------------------
    paragraph: str
    Paragraph in Chinese.
    --------------------
    Returns:
    <generator> object of sentences.
    """
    for sent in re.findall(r'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
        yield sent