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
        stopwords[lang] = [str(line)[0] for line in f][1:]
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
def tokenize(lang, input):
    if lang == 'en':
        d = nlp_en(input.replace('\n', ' '))
        d_tokens = filter(lambda x : (x.is_stop == False and x.is_punct == False and x.pos_!='NUM'), d)
        return map(lambda x: str(x).lower(), d_tokens)
    elif lang == 'zh':
        d = nlp_zh.cut(input, cut_all=False)
        return filter(lambda x: (x not in stopwords[lang] and x not in string.punctuation), d)
    elif lang == 'ms':
        d = nlp_ms.preprocessing.SocialTokenizer().tokenize(input)
        d_tokens = filter(lambda x: (x not in stopwords[lang] and x not in string.punctuation), d)
        return map(lambda x: str(x).lower(), d_tokens)
    elif lang == 'ta':
        d = nlp_ta(input)
        d.tokenize()
        return filter(lambda x: (x not in stopwords[lang] and x not in string.punctuation), d.tokens)

def vectorize(self, input_):
    Q, tokens, token_vecs = dict(), dict(), list()
    # To tokenize and add spaces to chinese text in multilingual query
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
        tokens[lang] = list(tokenize(lang,' '.join(Q[lang])))
        for token in tokens[lang]:
            try:
                token_vecs.append(self.vecs[lang][token])
            except:
                pass
                #raise KeyError('{} cannot be found in {} dictionary'.format(token, lang))
    return sum(np.array(vec) for vec in token_vecs)

def vectorize_lang(self, input_, lang):
    tokens = tokenize(lang, input_)
    tokens_vecs = []
    for token in list(tokens):
        try:
            tokens_vecs.append(self.vecs[lang][token])
        except:
            pass
            #raise KeyError('{} cannot be found in dictionary'.format(token))
    return sum(np.array(vec) for vec in tokens_vecs)

def zh_sent_tokenize(paragraph):
    for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
        yield sent