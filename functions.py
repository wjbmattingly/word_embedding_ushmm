import gensim
from gensim.utils import simple_preprocess
import glob
import string
from gensim.models import Word2Vec
import json

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

def remove_puncs(text):
    texts = []
    exclist = string.punctuation + string.digits

    exclist.replace("-", "")
    # remove punctuations and digits from oldtext
    table_ = str.maketrans('', '', exclist)
    newtext = ' '.join(text.translate(table_).split())
    newtext = newtext.split()

    return newtext

def gen_corpus(file_loc):
    texts = []
    files = glob.glob(file_loc)
    final_files = []
    for file in files:
        file = file.replace("\\", "/")
        with open (file, "r", encoding="utf-8") as f:
            text = f.read()
            segs =  text.replace("\n", " ").split(".")
            for seg in segs:
                original = seg.strip()+"."
                seg = remove_puncs(original)
                texts.append(seg)
                final_files.append((file, original))

    return (texts, final_files)



def similarity(model, word, n):
    res =  model.wv.similar_by_word(word, topn=n)
    return (res)


def make_rels_nums02(model, word, tiers=[10, 10, 10], removal_words=[]):
    #tier 0
    words = {word:[100, 0, word]}

    #tier 1
    res = similarity(model, word, tiers[0])
    x=1
    for item in res:
        if item[0] not in words and item[0] not in removal_words:
            words[item[0]] = [item[1], x, word]

    #other_tiers
    for tier in tiers[1:]:
        for item in list(words):
            data = words[item]
            if data[1] == x:
                res2 = similarity(model, item, tier)
                for item2 in res2:
                    if item2[0] not in words and item2[0] not in removal_words:
                        words[item2[0]] = [item2[1], x+1, item]
        x=x+1
    total_words = len(words)

    final = {}
    for item in words:
        data = words[item]
        val, tier, root = data
        if tier > 1:
            t2val = model.wv.similarity(word, root)
        else:
            t2val = 1
        val = val*t2val
        if item != word:
            final[item] = val*total_words
        else:
            final[item] = total_words*3
    return (final)



def calculate_similarity(model, word, tiers, removal_words=[]):
    data = make_rels_nums02(model, word, tiers, removal_words)
    return data

def text_value(word_vals, text, word, limited_words):
    hits = []
    total_val = 0
    found=False
    for item in limited_words:
        if item in text:
            found=True
    if found==True:
        for word in word_vals:
            quantity = text.count(word)
            val = quantity*word_vals[word]
            total_val = total_val+val
            if val > 0:
                hits.append((val, word, quantity, word_vals[word]))
    hits.sort()
    hits.reverse()
    text_val = [total_val, hits]
    return text_val

def run_algo(key_word, model, texts, limited_words, style_option, tiers=[10,10,10], removal_words=[]):
    word_vals = calculate_similarity(model, key_word, tiers, removal_words)
    res = {"word_vals": word_vals, "results": []}
    texts, files = texts
    if style_option == "Segment":
        x=0
        for text in texts:
            text_val = text_value(word_vals, text, key_word, limited_words)
            if text_val[0] > 0:
                res["results"].append((text_val[0], files[x], text_val[1]))
            x=x+1
        res["results"].sort()
        res["results"].reverse()
    elif style_option == "Document":
        doc_vals = {}
        x=0
        for text in texts:
            text_val = text_value(word_vals, text, key_word, limited_words)
            if text_val[0] > 0:

                if files[x][0] in doc_vals:

                    val = text_val[0]+doc_vals[files[x][0]]

                    doc_vals[files[x][0]] = val
                else:
                    doc_vals[files[x][0]] = text_val[0]
            x=x+1
        for val in doc_vals:
            res["results"].append((doc_vals[val], [val, "Too long to display..."], 0))
        res["results"].sort()
        res["results"].reverse()

    return (res)








#
