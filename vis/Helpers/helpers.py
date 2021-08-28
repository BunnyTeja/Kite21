#!/usr/bin/env python

# Imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
import sklearn


import spacy
from spacy.symbols import PUNCT, SYM, NUM, VERB
from spacy import displacy
import en_core_web_lg
nlp = spacy.load('en_core_web_lg')
nlp2 = spacy.load("en_core_web_sm") 
stopwords = nlp.Defaults.stop_words


import pickle
import requests
import re
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Comment
import pandas as pd
from gensim.models import Word2Vec
import random
import nltk
from nltk import bigrams, ngrams
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import matplotlib.cm as cm

from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

import os
path = os.path.dirname(os.path.abspath('__file__'))


# Link to a sample CNN article
link = 'http://lite.cnn.com/en/article/h_f68f79dede493cb0a82201f6ea0ce293'

# 4 categories of documents

computers = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x']
recr = ['rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey']
science = ['sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space']
politics = ['soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']


def save_data(data_object, path):
    pickling_on = open(path,'wb')
    pickle.dump(data_object, pickling_on)
    pickling_on.close()


# Calculate cosine similarity of 2 vectors
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

# Load pre-trained Glove embeddings
def loadGloveModel(File):
    # print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    # print(len(gloveModel)," words loaded!")
    return gloveModel
    

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


# Fetch an article given a URL (Name is misleading (get_cnn))
def get_cnn(url):
  x = requests.get(url)
  soup = BeautifulSoup(x.content, 'html.parser')
  text = soup.findAll(text=True)
  visible_texts = filter(tag_visible, text)  
  return u" ".join(t.strip() for t in visible_texts)
  

# Pre-process text to pass to the embeddings generator
def process(text):
    words = []
    verbs = set()
    for token in nlp(text):
        if token.pos != PUNCT and token.pos != NUM and token.pos != SYM  and str(token) not in stopwords and len(str(token)) > 1 and (str(token)).isalnum():
            word = str(token.lemma_).lower()
            words.append(word)
            if token.pos == VERB:
                verbs.add(word)
    return words, verbs

# Get RFP data to train document classifier
def get_rfp_docs(op = 0):
    df1 = pd.read_csv(path+'/vis/Data/programs_archive_info.csv')
    df2 = pd.read_csv(path+'/vis/Data/programs_active_info.csv')
    df = df1.append(df2)
    app = df['abstract'] + ' ' + df['description']
    df['all'] = app
    docs = list(set(df['all']))
    final_docs = []
    for doc in docs:
        try:
            x = len(doc)
            final_docs.append(doc)
        except:
            continue
    docs = final_docs
    new_docs = []
    for doc in docs:
        tokens = doc.split(' ')
        for i in range(len(tokens)//320):
            temp = ' '.join(tokens[i*320:(i+1)*320])
            new_docs.append(temp)
    random.seed(42)
    random.shuffle(new_docs)
    if op == 1:
        return new_docs[:2373]
    else:
        return docs


# Generate embeddings, given a corpus
def generate_embeddings(name, category = []):
    original_data = []
    if name == 'rfp':
        original_data = get_rfp_docs()
    else:
        data_train = fetch_20newsgroups(subset='train', categories = category, shuffle=True, random_state=42)
        original_data = data_train.data
    data = []
    verbs = set()
    print(len(original_data))
    for document in original_data:
        result = process(document)
        data.append(result[0])
        verbs = verbs.union(result[1])
    model = Word2Vec(data, min_count=1)
    print(model)
    model.save(path+'/../Models/'+name+'.bin')
    return verbs

# Train document classifier
def train_classifier():
    num_each = 2373
    
    train_politics = fetch_20newsgroups(subset='train', categories = politics, shuffle=True, random_state=42)
    data_politics = train_politics.data[:num_each]
    target_politics = [3 for i in range(num_each)]

    # print(len(data_politics))
    
    train_computers = fetch_20newsgroups(subset='train', categories = computers, shuffle=True, random_state=42)
    data_computers = train_computers.data[:num_each]
    target_computers = [0 for i in range(num_each)]
    
    # print(len(data_computers))
    
    train_science = fetch_20newsgroups(subset='train', categories = science, shuffle=True, random_state=42)
    data_science = train_science.data[:num_each]
    target_science = [2 for i in range(num_each)]
    
    # print(len(data_science))
    
    train_recr = fetch_20newsgroups(subset='train', categories = recr, shuffle=True, random_state=42)
    data_recr = train_politics.data[:num_each]
    target_recr = [1 for i in range(num_each)]
    
    # print(len(data_recr))
    
    data = data_politics + data_computers + data_science + data_recr
    target = target_politics + target_computers + target_science + target_recr

    data.extend(get_rfp_docs(1))
    target.extend([4 for i in range(len(get_rfp_docs(1)))])
    random.seed(42)
    c = list(zip(data, target))
    random.shuffle(c)
    data, target = zip(*c)
    target = np.array(target)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, target)
    return (count_vect, tfidf_transformer, clf)

(count, tfidf, clf) = train_classifier()


# Classify document theme into 1 of 5 categories
def classify(text):
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    docs_new = [text]
    X_new_counts = count.transform(docs_new)
    X_new_tfidf = tfidf.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    if predicted[0] == 0:
        return 'Computers'
    if predicted[0] == 1:
        return 'Recreational'
    if predicted[0] == 2:
        return 'Science'
    if predicted[0] == 3:
        return 'Politics'
    else:
        return 'RFP'


# Filter the verbs out of a set of words
def verb_filter(words, verbs_set):
    ans = set()
    for word in words:
        if word in verbs_set:
            ans.add(word)
        else:
            doc = nlp2(word + ' him')[0]
            if doc.pos == VERB:
                verbs_set.add(str(doc))
                ans.add(str(doc))
    return ans


# Compute verb vocabulary for all categories
def compute_verb_vocab():
    all_verbs = set()
    politics_verbs = (generate_embeddings('politics', politics)).intersection(glove)
    save_data(politics_verbs, path+'/vis/Data/politics_verbs.pickle')
    all_verbs = all_verbs.union(politics_verbs)
    
    computers_verbs = (generate_embeddings('computers', computers)).intersection(glove)
    save_data(computers_verbs, path+'/vis/Data/computers_verbs.pickle')
    all_verbs = all_verbs.union(computers_verbs)
    
    science_verbs = (generate_embeddings('science', science)).intersection(glove)
    save_data(science_verbs, path+'/vis/Data/science_verbs.pickle')
    all_verbs = all_verbs.union(science_verbs)
    
    recr_verbs = (generate_embeddings('recr', recr)).intersection(glove)
    save_data(recr_verbs, path+'/vis/Data/recr_verbs.pickle')
    all_verbs = all_verbs.union(recr_verbs)
    
    rfp_verbs = (generate_embeddings('rfp')).intersection(glove)
    save_data(rfp_verbs, path+'/vis/Data/rfp_verbs.pickle')
    all_verbs = all_verbs.union(rfp_verbs)
    
    save_data(all_verbs, path+'/vis/Data/all_verbs.pickle')

# Load verb vocabulary for all categories
def load_verb_vocab():
    politics_verbs = pickle.load(open(path+'/vis/Data/politics_verbs.pickle', 'rb'))
    computers_verbs = pickle.load(open(path+'/vis/Data/computers_verbs.pickle', 'rb'))
    science_verbs = pickle.load(open(path+'/vis/Data/science_verbs.pickle', 'rb'))
    recr_verbs = pickle.load(open(path+'/vis/Data/recr_verbs.pickle', 'rb'))
    rfp_verbs = pickle.load(open(path+'/vis/Data/rfp_verbs.pickle', 'rb'))
    all_verbs = pickle.load(open(path+'/vis/Data/all_verbs.pickle', 'rb'))
    return politics_verbs, computers_verbs, science_verbs, recr_verbs, rfp_verbs, all_verbs

# Load word embeddings for all categories
def get_embs(glove):
    model_politics = Word2Vec.load(path+'/vis/Models/politics.bin')
    model_computers = Word2Vec.load(path+'/vis/Models/computers.bin')
    model_science = Word2Vec.load(path+'/vis/Models/science.bin')
    model_recr = Word2Vec.load(path+'/vis/Models/recr.bin')
    model_rfp = Word2Vec.load(path+'/vis/Models/rfp.bin')
    politics_verbs, computers_verbs, science_verbs, recr_verbs, rfp_verbs, all_verbs = load_verb_vocab()
    politics_embs = []
    computers_embs = []
    science_embs = []
    recr_embs = []
    rfp_embs = []
    glove_embs = []
    for verb in politics_verbs:
        politics_embs.append(model_politics.wv[verb])
    for verb in science_verbs:
        science_embs.append(model_science.wv[verb])
    for verb in computers_verbs:
        computers_embs.append(model_computers.wv[verb])
    for verb in recr_verbs:
        recr_embs.append(model_recr.wv[verb])
    for verb in rfp_verbs:
        rfp_embs.append(model_rfp.wv[verb])
    for verb in all_verbs:
        glove_embs.append(glove[verb])
    return np.array(politics_embs), np.array(computers_embs), np.array(science_embs), np.array(recr_embs), np.array(rfp_embs), np.array(glove_embs)


# Helper function
def get_embs_map(glove):
    politics_verbs, computers_verbs, science_verbs, recr_verbs, rfp_verbs, all_verbs = load_verb_vocab()
    politics_embs, computers_embs, science_embs, recr_embs, rfp_embs, glove_embs = get_embs(glove)
    politics_map = {}
    science_map = {}
    computers_map = {}
    recr_map = {}
    rfp_map = {}
    glove_map = {}
    i = 0
    for verb in politics_verbs:
        politics_map[verb] = politics_embs[i]
        i += 1
    i = 0
    for verb in computers_verbs:
        computers_map[verb] = computers_embs[i]
        i += 1
    i = 0
    for verb in science_verbs:
        science_map[verb] = science_embs[i]
        i += 1
    i = 0
    for verb in recr_verbs:
        recr_map[verb] = recr_embs[i]
        i += 1
    i = 0
    for verb in rfp_verbs:
        rfp_map[verb] = rfp_embs[i]
        i += 1
    i = 0
    for verb in all_verbs:
        glove_map[verb] = glove_embs[i]
        i += 1
    return politics_map, computers_map, science_map, recr_map, rfp_map, glove_map 

# Solve the Nearest Neighbor problem with Euclidean distance as metric
def fit_neighbors(glove, n = 6):
    politics_embs, computers_embs, science_embs, recr_embs, rfp_embs, glove_embs = get_embs(glove)
    politics_neigh = NearestNeighbors(n_neighbors=n)
    politics_neigh.fit(politics_embs)
    computers_neigh = NearestNeighbors(n_neighbors=n)
    computers_neigh.fit(computers_embs)
    science_neigh = NearestNeighbors(n_neighbors=n)
    science_neigh.fit(science_embs)
    recr_neigh = NearestNeighbors(n_neighbors=n)
    recr_neigh.fit(recr_embs)
    rfp_neigh = NearestNeighbors(n_neighbors=n)
    rfp_neigh.fit(rfp_embs)
    glove_neigh = NearestNeighbors(n_neighbors=n)
    glove_neigh.fit(glove_embs)
    return politics_neigh, computers_neigh, science_neigh, recr_neigh, rfp_neigh, glove_neigh


# Helper function
def get_index_to_word():
    politics_verbs, computers_verbs, science_verbs, recr_verbs, rfp_verbs, all_verbs = load_verb_vocab()
    politics_map = {}
    science_map = {}
    computers_map = {}
    recr_map = {}
    rfp_map = {}
    glove_map = {}
    i = 0
    for verb in politics_verbs:
        politics_map[i] = verb
        i += 1
    i = 0
    for verb in computers_verbs:
        computers_map[i] = verb
        i += 1
    i = 0
    for verb in science_verbs:
        science_map[i] = verb
        i += 1
    i = 0
    for verb in recr_verbs:
        recr_map[i] = verb
        i += 1
    i = 0
    for verb in rfp_verbs:
        rfp_map[i] = verb
        i += 1
    i = 0
    for verb in all_verbs:
        glove_map[i] = verb
        i += 1
    return politics_map, computers_map, science_map, recr_map, rfp_map, glove_map