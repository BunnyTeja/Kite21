#!/usr/bin/env python


# includes
from .helpers import *

# Plotting
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 

# Pass custom comparator function to sort
from functools import cmp_to_key

# Generate Word Clouds
from wordcloud import WordCloud

# Sentiment Analysis
from textblob import TextBlob

import numpy as np

# Encode images
import base64
import io
from io import BytesIO

from collections import Counter

# Spacy
from spacy.symbols import nsubj, VERB, pobj, dobj, amod, NOUN, ADJ, PRON, PROPN, nsubjpass, ADV, PUNCT
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_lg
nlp = spacy.load('en_core_web_lg')

# Calculate Levenshtein distance
from fuzzywuzzy import fuzz

# Abbreviation to Meaning map
meaning = {'NORP':'Nationality / Religious or Political group', 'ORG': 'Organization', 'PERSON': 'Person', 'GPE': 'Geopolitical entity', 'FACILITY':'Facility', 'LOC':'Location', 'PRODUCT':'Product', 'EVENT':'Event', 'WORK_OF_ART':'Work of Art', 'LAW':'Law', 'LANGUAGE':'Language', 'FAC':'Buildings, Airports, Highways, Bridges'}

# Entity types to consider
todo = {'GPE', 'NORP', 'PERSON', 'ORG'}

# Entity types to not consider
nodo = {'DATE', 'TIME', 'MONEY', 'PERCENT', 'ORDINAL', 'CARDINAL', 'QUANTITY'}

# For HTML
import html

# Google Scholar API
from scholarly import scholarly

# Convert string to date
import datefinder

# Domains
categories = ['all', 'politics', 'computers', 'science', 'recr', 'rfp']

# Funding Agencies
agencies = {'nsf', 'nih', 'hhs', 'cdmrp', 'usda', 'darpa', 'army', 'navy', 'af', 'dhs', 'nasa', 'epa', 'ed', 'nea', 'neh', 'energy', 'hud', 'fbo', 'cfda'}


# Get average sentiment of a list of words
def get_sent(words):
    sent = 0
    sent_list = []
    if words != None:
        n = len(words)
        if n == 0:
            return 0
        for word in words:
            x = TextBlob(word).sentiment.polarity
            sent += x
            sent_list.append(x)
        sent /= n
    return sent, sent_list

# For printing purposes (Capitalise first letter)
def capital(text):
    try:
        temp = text.split(' ')
        return ' '.join([el[0].upper()+el[1:] for el in temp])
    except:
        return text


# Initialise trained embeddings and Nearest Neighbor models
def init_model(glove):
    model = {}
    model['politics_verbs'], model['computers_verbs'], model['science_verbs'], model['recr_verbs'], model['rfp_verbs'], model['glove_verbs'] = load_verb_vocab()
    model['politics_idx'], model['computers_idx'], model['science_idx'], model['recr_idx'], model['rfp_idx'], model['glove_idx'] = get_index_to_word()
    model['politics_embs'], model['computers_embs'], model['science_embs'], model['recr_embs'], model['rfp_embs'], model['glove_embs'] = get_embs(glove)
    model['politics_map'], model['computers_map'], model['science_map'], model['recr_map'], model['rfp_map'], model['glove_map'] = get_embs_map(glove)
    # model['politics_2d'], model['computers_2d'], model['science_2d'], model['recr_2d'], model['glove_2d'] = get_2d_embs()
    model['politics_neigh'], model['computers_neigh'], model['science_neigh'], model['recr_neigh'], model['rfp_neigh'], model['glove_neigh'] = fit_neighbors(glove)
    return model


# Given a verb and a category, return 5 closest verbs with Euclidean Distance as metric
def most_similar_words(word, category, metric, model, topn = 5):
    if category == 'all':
        category = 'glove'
    if metric == 'cosine':
        queries = list(model.get(category+'_verbs', []))
        if queries == []:
            return []
        by_similarity = sorted(queries, key=lambda w: get_cosine_similarity(model[category+'_map'][w], model[category+'_map'][word]), reverse=True)
        return [(w.lower()) for w in by_similarity[:topn+1] if w.lower() != word.lower()]
    else:
        if model[category+'_map'].get(word, []) == []:
            return []
        results = model[category+'_neigh'].kneighbors([model[category+'_map'][word]], return_distance=False)
        return [model[category+'_idx'][y] for y in list(results[0]) if model[category+'_idx'][y] != word]

# Used for Full Text Section
def pretty_print_long(text):
    soup = BeautifulSoup(displacy.render(nlp(text), page=True, style='ent'), 'html.parser')
    while True:
        temp = soup.find('body')
        if not temp:
            break
        temp.name = 'div'
    contents = BeautifulSoup(str(soup.div.div)[:-6]+'', 'html.parser')
    soup.div.div.replace_with(contents)
    return str(soup.div)

# Temporary function
def pretty_print_short(text):
    soup = BeautifulSoup(displacy.render(nlp(text), page=True, style='ent'), 'html.parser')
    while True:
        temp = soup.find('body')
        if not temp:
            break
        temp.name = 'div'
    contents = BeautifulSoup(str(soup.div.div)[:-6]+'<br><button type = "SUBMIT" class = "btn btn-primary" href="#more">View more</button>', 'html.parser')
    soup.div.div.replace_with(contents)
    return str(soup.div)

# Generate a trie from a map from words to codes
def get_trie(m1):
    trie = {}
    for word in list(m1.keys()):
        if word == '':
            continue
        temp = trie
        for letter in word:
            if letter not in temp.keys():
                temp[letter] = {}
            temp = temp[letter]
        temp['__end__'] = m1[word]
    return trie

# Return the closest string to the word in the trie
def find_(trie, word):
    c = ''
    temp = trie
    for letter in word:
        if letter not in temp.keys():
            dist, word = get_nearest(temp)
            return (dist, c+word)
        else:
            c += letter
            temp = temp[letter]
    if '__end__' not in temp.keys():
        dist, word = get_nearest(temp)
        return (dist, c+word)
    else:
        return (0, c)

# Helper function for find_
def get_nearest(trie):
    ans = 1000000
    word = ''
    if '__end__' in trie.keys():
        return (0, '')
    for key in list(trie.keys()):
        temp = get_nearest(trie[key])
        if ans > temp[0]:
            ans = temp[0]
            word = key+temp[1]
    return (ans+1, word)

# Reutrn Classification Codes, given a piece of text
def get_codes_class(text, term_to_code, code_to_term, triee, to_print = 0):
    try:
        # A map from a term to its count
        ans = {}

        # Tokenize text into words
        tokens = [str(word).lower() for word in word_tokenize(text) if str(word).isalpha()]
        
        # A map to check whether a word has already been included in the Classification
        done = {}

        # Generate N-Grams from 7 to 2
        for i in range(7,1,-1):
            text_temp_ = list(ngrams(tokens, i))
            for j, el in enumerate(text_temp_):
                term = ' '.join(el)
                dist, nearest = find_(triee, term)
                similarity = fuzz.ratio(nearest.strip(), term.strip())

                # Consider terms if similarity > 75 in the trie
                if similarity>75:
                    term = nearest
                    if term != 'data' and term != 'general' and term != '' and term != 'miscellaneous':
                        check = False
                        for k in range(len(el)):
                            if done.get(j+k, -1) == -1:
                                check = True
                                break
                        if check:
                            for k in range(len(el)):
                                done[j+k] = 1
                            if to_print == 1:
                                print(term, ',', nearest, ',', similarity, ',', j, ',', term_to_code[term])
                            ans[term] = ans.get(term, 0)+1
        
        # Rest of the code decides what to do in case a term belongs to 2 codes
        final = {}
        codes = {}
        for key in list(ans.keys()):
            if len(term_to_code[key]) == 1:
                final[(key, list(term_to_code[key])[0])] = ans[key]
                s = list(term_to_code[key])[0]
                codes[s] = codes.get(s, 0)+1
                if s.count('.') >= 1:
                    t = s[:s.find('.')]
                    codes[t] = codes.get(t, 0)+1
                if s.count('.') >= 2:
                    x = [m.start() for m in re.finditer('\.', s)]
                    t = s[:x[-1]]
                    codes[t] = codes.get(t, 0)+1
        for key in list(ans.keys()):
            if len(term_to_code[key]) == 1:
                continue
            lengths = []
            for el in term_to_code[key]:
                lengths.append(codes.get(el, 0))
            lengths.sort()
            if lengths[-1] != lengths[-2]:
                for el in term_to_code[key]:
                    if codes.get(el, 0) == lengths[-1]:
                        final[(key, el)] = ans[key]
                        break
            else:
                keys = []
                for el in term_to_code[key]:
                    if el.count('.') >= 1:
                        x = [m.start() for m in re.finditer('\.', el)]
                        keys.append(el[:x[-1]])
                    else:
                        keys.append(el)
                lengths = []
                for el in keys:
                    lengths.append(codes.get(el, 0))
                lengths.sort()
                if lengths[-1] != lengths[-2]:
                    for i, el in enumerate(keys):
                        if codes.get(el, 0) == lengths[-1]:
                            final[(key, list(term_to_code[key])[i])] = ans[key]
                            break
                else:
                    temp = []
                    for el in keys:
                        if el.count('.') >= 1:
                            x = [m.start() for m in re.finditer('\.', el)]
                            temp.append(el[:x[-1]])
                        else:
                            temp.append(el)
                    keys = temp
                    lengths = []
                    for el in keys:
                        lengths.append(codes.get(el, 0))
                    lengths.sort()
                    if lengths[-1] != lengths[-2]:
                        for i, el in enumerate(keys):
                            if codes.get(el, 0) == lengths[-1]:
                                final[(key, list(term_to_code[key])[i])] = ans[key]
                                break
                    else:
                        final[(key, list(term_to_code[key])[-1])] = ans[key]
        codes = {}
        title = {}
        for key in list(final.keys()):
            title[key[1][0]] = title.get(key[1][0], 0)+1
        letter = max(title, key= lambda x: title[x])
        title = code_to_term[letter]
        final_codes = sorted(list(final.keys()), key = lambda x: final[x], reverse=True)[:10]
        areas = {}
        codes = {}
        for i, code in enumerate(final_codes):
            codes[code[1]] = final[code]
            areas[code_to_term[code[1]]] = codes[code[1]]

        # Letter is the Code for the Letter with most codes (A-K)
        # codes is a map from Classification Code to Frequency (Top 10 codes)
        return (letter, codes)
    except:
        return ("", {})

# Return two arrays of dates and Classification Topics for previous NSF data stored
def get_time_series(agency, term_to_code, code_to_term):
    if agency.lower() != 'nsf':
        return ([], [])
    x = pickle.load(open(path+'/vis/Data/nsf_dates.pickle', 'rb'))
    y = pickle.load(open(path+'/vis/Data/nsf_topics.pickle', 'rb'))
    return (x, y)

# Return author id and a list of publications sorted in descending order of year
def get_google_scholar(person):
    people = [person]
    for person in people:
        try:
            # print(person)
            search_query = scholarly.search_author(person)
            author = scholarly.fill(next(search_query))
        except:
            return '', []
        author_id = author['scholar_id']
        pubs = author['publications']
        def remove_duplicates(pubs):
            titles = set()
            temp = []
            for pub in pubs:
                if pub['bib'].get('title', '') not in titles:
                    titles.add(pub['bib'].get('title', ''))
                    temp.append(pub)
            return temp

        pubs = remove_duplicates(pubs)
        pubs = sorted(pubs, key = lambda x:int(x['bib'].get('pub_year', -1)), reverse=True)
        pubs = [pub for pub in pubs if pub['bib'].get('pub_year', -1) != -1]
        return author_id, pubs

# Visualize top entities
def vis(model, text):
    # A map from ACM Classification Codes to their corresponding meaning
    code_to_term = pickle.load(open(path+'/vis/Data/code_to_term.pickle', 'rb'))

    # A map from ACM Classification Terms to their corresponding codes
    term_to_code = pickle.load(open(path+'/vis/Data/term_to_code.pickle', 'rb'))

    # Convert the map term_to_code to a trie
    trie = get_trie(term_to_code)

    # A map to pass on to the HTML page with information to display
    summary = {}

#     Initial set of entities
    ents = set()
    
#     Map from entity name to its type
    types = {}
    
#     Used for coreference resolution (Same entity reffered by different names. eg. -> John Daggett, Daggett)
    parent = {}
    
#     Get map from entity to number of occurences of all entities
    occur = {}
    
    # Get Classification codes for the text
    codes = get_codes_class(text, term_to_code, code_to_term, trie)

    # Use Spacy
    doc = nlp(text)

    # A list of sentences in the doc
    sents = [str(sent) for sent in doc.sents]

#     Keep track of all tokens in doc
    # A map from a token index to the token
    tokens = {}

    for token in doc:
        tokens[token.i] = token


#     Get relevant entities
    for ent in doc.ents:
        if ent.label_ not in nodo:
            ents.add(str(ent))
            parent[str(ent)] = str(ent)
            types[str(ent)] = ent.label_


    ents = list(ents)

    # Merge different references to same entity using Levenshtein Distance
    for i in range(len(ents)):
        for j in range(i+1, len(ents)):
            similarity = fuzz.ratio(ents[i].lower(), ents[j].lower())
            if similarity >75 or ((ents[i].lower() in ents[j].lower() or ents[j].lower() in ents[i].lower()) and types[ents[i]] == 'PERSON' and types[ents[j]] == 'PERSON'):
                if len(ents[i]) > len(ents[j]):
                    alpha = ents[i]
                    beta = ents[j]
                else:
                    alpha = ents[j]
                    beta = ents[i]
                while alpha != parent[alpha]:
                    alpha = parent[alpha]
                parent[beta] = alpha

    for key in list(parent.keys()):
        temp = key
        while temp != parent[temp]:
            temp = parent[temp]
        parent[key] = temp

#     Combine similar entities
    final_entities = set()
    for key in list(parent.keys()):
        final_entities.add(parent[key])

    # Final set of merged entities is stoed in final_entities

    # A map from entity to verbs (actions on) and their frequencies
    actions_on = {}

    # A map from entity to verbs (actions by) and their frequencies
    actions_by = {}

    # A map from entity to a set of all nearby words
    others = {}

    # A map from sentence number to a set of all entities present in it
    sentence_map = {}

    # A map from entity to a set of all sentences it is mentioned in
    ent_to_sentence = {}

    # A list of sentence numbers of all tokens. Length of list = Number of tokens in doc
    token_sent = [sent_id for sent_id, sent in enumerate(doc.sents) for token in sent]
    
    # process direct entity mentions
    for ent in doc.ents:
        if ent.label_ not in nodo:
        
            token_list = [token for token in ent]
            begin = token_list[0].i
            end = token_list[-1].i
            for j in range(max(0, begin-5), min(end+5, len(tokens)-1)):
                if tokens[j].pos != PUNCT and tokens[j].pos != VERB and str(tokens[j]).lower() not in STOP_WORDS and (j > end or j < begin) and str(tokens[j]) not in str(ent):
                    if parent[str(ent)] not in others:
                        others[parent[str(ent)]] = set()

                    # Add the nearby word to others
                    others[parent[str(ent)]].add(str(tokens[j]))

            # Maintain the occurence of the entity
            occur[parent[str(ent)]] = occur.get(parent[str(ent)], 0)+1

            for token in ent:
                
                if parent[str(ent)] not in ent_to_sentence:
                    ent_to_sentence[parent[str(ent)]] = set()
                ent_to_sentence[parent[str(ent)]].add(sents[token_sent[token.i]])

                if token_sent[int(token.i)] not in sentence_map:
                    sentence_map[token_sent[int(token.i)]] = set()
                sentence_map[token_sent[int(token.i)]].add(parent[str(ent)])
                
                if token.head.pos == VERB:
                    if token.dep != nsubj:
                        if parent[str(ent)] not in actions_on:
                            actions_on[parent[str(ent)]] = {}
                        actions_on[parent[str(ent)]][str(token.head.lemma_)] = actions_on[parent[str(ent)]].get(str(token.head.lemma_), 0) + 1
                
                    if token.dep == nsubj:
                        if parent[str(ent)] not in actions_by:
                            actions_by[parent[str(ent)]] = {}
                        actions_by[parent[str(ent)]][str(token.head.lemma_)] = actions_by[parent[str(ent)]].get(str(token.head.lemma_), 0) + 1

#     A list to store top entities sorted by their number of occurences
    sorted_result = []
#     Sort verbs first by sentiment and then by count
    for key in list(actions_on.keys()):
        temp = list(actions_on[key].keys())
        actions_on[key] = sorted(temp, key = lambda x: actions_on[key][x]+1000*np.abs(get_sent([x])[0]), reverse=True)
    
#     Sort verbs by count and sentiment
    for key in list(actions_by.keys()):
        temp = list(actions_by[key].keys())
        actions_by[key] = sorted(temp, key = lambda x: actions_by[key][x]+1000*np.abs(get_sent([x])[0]), reverse=True)
        
#     Get top entities
    for entity in final_entities:
        sorted_result.append([types[entity], entity, occur.get(entity, 0)])

    # This block is used to determine the funding agency
    ent_ag = sorted(list(occur.keys()), key= lambda x: occur[x], reverse = True)
    agency = ''
    for ag in ent_ag:
        if str(ag).lower() in agencies:
            agency = ag
            break

    # A comparator function to compare entities (first by occurence and then preference is given to PERSON)
    def comp_ents(e1, e2):
        if e1[2] != e2[2]:
            return e2[2] - e1[2]
        if e1[0] == 'PERSON' and e2[0] != 'PERSON':
            return -1
        if e2[0] == 'PERSON' and e1[0] != 'PERSON':
            return 1
        return 0

    sorted_result = sorted(sorted_result, key=cmp_to_key(comp_ents))

    # A list of names of top 5 entities
    sorted_names = [x[1] for x in sorted_result[:5]]
    

    final_entities = sorted_names

    #     Calculate entity co-relation
    # dist_mtr[i][j] is number of common sentences of entities i and j
    dist_mtr = np.zeros((len(final_entities), len(final_entities)))
    
    for i in range(len(final_entities)):
        for j in range(i+1, len(final_entities)):
            for key in list(sentence_map.keys()):
                if final_entities[i] in sentence_map[key] and final_entities[j] in sentence_map[key]:
                    dist_mtr[i][j] += 1
                    dist_mtr[j][i] += 1

    # A list to store related entities
    related = []

    # Total number of sentences
    sents_len = len([sent for sent in doc.sents])
    
    # Sentences are sorted by sentiment polarity
    sents = sorted(sents, key = lambda x: TextBlob(x).sentiment.polarity, reverse=True)
    
    # Top 5 positive sentences
    summary['positive_sent'] = [[sents[i], np.round(TextBlob(sents[i]).sentiment.polarity, 2)] for i in range(min(5, len(sents))) if TextBlob(sents[i]).sentiment.polarity > 0]
    sents.reverse()

    # Top 5 negative sentences
    summary['negative_sent'] = [[sents[i], np.round(TextBlob(sents[i]).sentiment.polarity, 2)] for i in range(min(5, len(sents))) if TextBlob(sents[i]).sentiment.polarity < 0]

    # Entities are related to each other if they co-occur in more than 5 % of total sentences
    for i in range(dist_mtr.shape[0]):
        for j in range(i+1, dist_mtr.shape[0]):
            if i != j and dist_mtr[i][j] >= max(0.05 * sents_len, 5) and final_entities[i] in sorted_names and final_entities[j] in sorted_names:
                related.append([final_entities[i], final_entities[j]])


    
    result = {}
    summary['text_type'] = classify(text) # Domain classification of text
    summary['related'] = related # Related entities list
    summary['pprint'] = pretty_print_long(text) # Full text section of HTML page
    summary['words'] = len([str(token) for token in doc if str(token).lower() not in STOP_WORDS and token.pos != PUNCT]) # Total number of words excluding stop words
    summary['sents'] = sents_len
    summary['sent_lens'] = [len([str(token) for token in sent if str(token).lower() not in STOP_WORDS]) for sent in doc.sents] # A list of lengths of each sentence
    summary['sent_sentiment'] = [TextBlob(str(sent)).sentiment.polarity for sent in doc.sents] # A list of sentiment polarity of each sentence
    
    try:
        summary['topic'] = capital(code_to_term[codes[0]]) + ' (' + codes[0] + ')' # Get the main topic from the Classification Codes
    except:
        summary['topic'] = ''
    try:
        summary['codes'] = codes[1] # Get a map of top 10 codes
    except:
        summary['codes'] = []

    summary['codes'] = [[code, capital(code_to_term[code]), summary['codes'][code]] for code in list(summary['codes'].keys())]
    summary['agency'] = agency # Funding agency, if RFP
    
    # The most important person by number of occurences
    person = ''
    for ent in sorted_result:
        if ent[0] == 'PERSON':
            person = str(ent[1])
            break

    # Boolean variables
    summary['is_person'] = 0
    summary['is_rfp'] = 0
    
    if agency != '' and summary['text_type'] == 'RFP':
        for ent in sorted_result[:5]:
            if ent[1] == agency and ent[1] != '':
                summary['is_rfp'] = 1
                break
            if ent[1] == person and ent[1] != '':
                break

    if person != '' and summary['text_type'] == 'RFP' and summary['is_rfp'] == 0:
        summary['text_type'] = 'Person'
        summary['is_person'] = 1
        summary['is_rfp'] = 0

    summary['person'] = person
    summary['time_series'] = ''
    summary['pie_share'] = ''
    summary['pubs'] = []

    # If text is RFP
    if summary['is_rfp'] == 1:

        # Get time series of dates and topics saved
        dates, topics = get_time_series(agency, term_to_code, code_to_term)
        topics = [capital(topic) + ' ('+list(term_to_code[topic])[0]+')' for topic in topics]
        
        # Plot Pie Chart
        fig = plt.figure()
        plt.title('Pie Share')
        counter = dict(Counter(topics))
        colors = list(iter(cm.rainbow(np.linspace(0, 1, len(counter.keys())))))
        topics_ = list(dict(Counter(topics)).keys())
        topic_to_color = {}
        for i in range(len(topics_)):
            topic_to_color[topics_[i]] = colors[i]
        patches, texts = plt.pie(list(dict(Counter(topics)).values()), colors = colors)
        plt.legend(patches, list(dict(Counter(topics)).keys()), loc='center left', bbox_to_anchor=(-1.2, 1.), fontsize=15)
        
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg', bbox_inches='tight')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
        summary['pie_share'] = my_base64_jpgData

        # Plot Time Series
        plt.figure()
        plt.xlabel('Years')
        plt.ylabel('Areas')
        plt.title('Time Series')
        plt.scatter(dates, topics, color = [topic_to_color[topic] for topic in topics])
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg', bbox_inches='tight')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
        summary['time_series'] = my_base64_jpgData

    # If text is Person
    if summary['is_person'] == 1:

        # Get information about author
        author_id, pubs = get_google_scholar(person)
        new_text = text

        # If author is searchable on Google Scholar
        if author_id != '':
            info = []
            all_pubs = []
            
            # Get all publication codes from titles of publications of the author
            for pub in pubs:
                year = pub['bib'].get('pub_year', -1)
                title = pub['bib'].get('title', '')
                pub_id = pub['author_pub_id']
                pub_link = 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user='+author_id+'&sortby=pubdate&citation_for_view='+pub_id
                new_text += '\n'+title
                topic, codes = get_codes_class(title, term_to_code, code_to_term, trie)

                if topic != '':
                    try:
                        yr = list(datefinder.find_dates(year + ' Jan'))[0]
                        info.append([yr, capital(code_to_term[topic]) + ' ('+topic + ')'])
                    except:
                        pass
                all_pubs.append([year, title, pub_link])

            # Top 10 publications to show
            all_pubs = all_pubs[:10]
            info.reverse()
            codes = {}

            # Get Classification codes incorporating publications also
            codes = get_codes_class(new_text, term_to_code, code_to_term, trie)
            summary['codes'] = codes[1]

            summary['codes'] = [[code, capital(code_to_term[code]), summary['codes'][code]] for code in list(summary['codes'].keys())]
            

            summary['pubs'] = all_pubs

            # Plot Pie Chart
            fig = plt.figure()
            plt.title('Pie Share')
            counter = dict(Counter([el[1] for el in info]))
            colors = list(iter(cm.rainbow(np.linspace(0, 1, len(counter.keys())))))
            topics = list(dict(Counter([el[1] for el in info])).keys())
            topic_to_color = {}
            for i in range(len(topics)):
                topic_to_color[topics[i]] = colors[i]
            patches, texts = plt.pie(list(dict(Counter([el[1] for el in info])).values()), colors = colors)
            plt.legend(patches, list(dict(Counter([el[1] for el in info])).keys()), loc='center left', bbox_to_anchor=(-1.2, 1.), fontsize=15)

            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='jpg', bbox_inches='tight')
            my_stringIObytes.seek(0)
            my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
            summary['pie_share'] = my_base64_jpgData

            # Plot Time Series
            plt.figure()
            plt.xlabel('Years')
            plt.ylabel('Areas')
            plt.title('Time Series')
            occurences = dict(Counter([str(el[0])+','+el[1] for el in info]))
            plt.scatter([el[0] for el in info], [el[1] for el in info], color = [topic_to_color[el[1]] for el in info], label = [occurences[str(el[0])+','+el[1]] for el in info])
            for i in range(len(info)):
                plt.annotate(occurences[str(info[i][0])+','+info[i][1]], (info[i][0],info[i][1]), textcoords="offset points", xytext=(0,10), ha='center')
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='jpg', bbox_inches='tight')
            my_stringIObytes.seek(0)
            my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
            summary['time_series'] = my_base64_jpgData
        
    # Plot Histogram of Sentence Sentiments
    plt.figure()
    n, bins, patches = plt.hist(x=summary['sent_sentiment'], bins='auto', range = [-1, 1], color='#0504aa', rwidth=0.9)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentence Sentiment')
    maxfreq = n.max()
    
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    summary['plot_sentiment'] = my_base64_jpgData

    # Plot Histogram of Sentence Lengths
    plt.figure()
    n, bins, patches = plt.hist(x=summary['sent_lens'], bins='auto', color='#0504aa', rwidth=0.9)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count of Sentences')
    maxfreq = n.max()
    
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    summary['plot_len'] = my_base64_jpgData
    plt.figure()

    # Plot top 5 entities with their number of mentions
    mentions = {}
    for ent in sorted_result[:5]:
        mentions[ent[1]] = ent[2]
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.bar(range(len(mentions)), list(mentions.values()), align='center')
    plt.xticks(range(len(mentions)), list(mentions.keys()), rotation=20)
    plt.xlabel('Entities')
    plt.ylabel('Number of Mentions')
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    summary['mentions'] = my_base64_jpgData
    plt.xticks([])
    plt.yticks([])
    plt.gcf().subplots_adjust(bottom=0)
    plt.xlabel('')
    plt.ylabel('')

    # Top 5 entities
    for ent in sorted_result[:5]:
        name = ent[1]
        result[name] = {}
        result[name]['name'] = name # Name of entitiy
        result[name]['count'] = ent[2] # Number of mentions
        result[name]['type'] = meaning[ent[0]] # Type (eg. NORP, PERSON, etc)
        sent = get_sent(others.get(ent[1])) # Sentiment of words associated
        if others.get(ent[1]) is not None:
            result[name]['sentiment'] = TextBlob(' '.join(others.get(ent[1]))).sentiment.polarity
        else:
            result[name]['sentiment'] = 0
        result[name]['sent_list'] = sent[1]


        sentence_list = list(ent_to_sentence[name]) # List of all sentences with mention of entity
        sentence_list = sorted(sentence_list, key = lambda x: TextBlob(x).sentiment.polarity, reverse=True)
        
        # Top 5 positive sentences with entity
        result[name]['positive_sent'] = [[sentence_list[i], np.round(TextBlob(sentence_list[i]).sentiment.polarity, 2)] for i in range(min(5, len(sentence_list))) if TextBlob(sentence_list[i]).sentiment.polarity > 0]
        sentence_list.reverse()

        # Top 5 negative sentences with entity
        result[name]['negative_sent'] = [[sentence_list[i], np.round(TextBlob(sentence_list[i]).sentiment.polarity, 2)] for i in range(min(5, len(sentence_list))) if TextBlob(sentence_list[i]).sentiment.polarity < 0]
        
        # Actions on entity
        result[name]['on'] = {}

        # Actions by entity
        result[name]['by'] = {}

        # Print Actions On
        i = 0
        for action in actions_on.get(ent[1], []):
            result[name]['on'][action] = []
            for category in categories:
                similar = most_similar_words(action, category, 'distance', model)
                if similar == []:
                    similar = ['-', '-', '-', '-', '-']
                result[name]['on'][action].append(similar)
            checker = [['-', '-', '-', '-', '-'] for i in range(len(categories))]
            if result[name]['on'][action] == checker:
                result[name]['on'].pop(action, None)
                continue
            i += 1
            if i == 3:
                break

#         Print Actions by
        i = 0
        for action in actions_by.get(ent[1], []):
            result[name]['by'][action] = []
            for category in categories:
                similar = most_similar_words(action, category, 'distance', model)
                if similar == []:
                    similar = ['-', '-', '-', '-', '-']
                result[name]['by'][action].append(similar)
            checker = [['-', '-', '-', '-', '-'] for i in range(len(categories))]
            if result[name]['by'][action] == checker:
                result[name]['by'].pop(action, None)
                continue
            i += 1
            if i == 3:
                break
            
        result[name]['image'] = ''

        # Generate Word cloud of associated words
        if len(others.get(ent[1], set())) != 0:
            try:
                wordcloud = WordCloud(max_font_size=40, width=600, height=400, max_words = 50, background_color='white', relative_scaling=1.0).generate_from_text(' '.join(others.get(ent[1], set())))
                plt.title(ent[1])
                plt.tight_layout(pad=20)
                plt.imshow(wordcloud, interpolation="bilinear")

                my_stringIObytes = io.BytesIO()
                plt.savefig(my_stringIObytes, format='jpg')
                my_stringIObytes.seek(0)
                my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
                result[name]['image'] = my_base64_jpgData
            except:
                result[name]['image'] = ''

    sents = [str(sent) for sent in doc.sents]

    # Map of frequency of each verb
    verb_ct = {}
    for token in doc:
        if token.pos == VERB:
            verb_ct[str(token.lemma_)] = verb_ct.get(str(token.lemma_), 0)+1

    # Code for handling events
    # Comparator function for comparing 2 events
    # First, the subject is checked if it is in top 5 entities, then object for the same
    # Then events with a location are given preference
    # Then events with a date are given preference
    # Then events with an object are given preference
    # Then events with higher verb frequency are given preference
    def compare(a, b):
        for name in sorted_names:
            if name in ' '.join(a[2]) and name not in ' '.join(b[2]):
                return -1
            if name in ' '.join(b[2]) and name not in ' '.join(a[2]):
                return 1
            if name in ' '.join(a[4]) and name not in ' '.join(b[4]):
                return -1
            if name in ' '.join(b[4]) and name not in ' '.join(a[4]):
                return 1
        if a[1] == '-' and b[1] != '-':
            return 1
        if a[1] != '-' and b[1] == '-':
            return -1
        if a[0] == '-' and b[0] != '-':
            return 1
        if a[0] != '-' and b[0] == '-':
            return -1
        if a[4] == '-' and b[4] != '-':
            return 1
        if a[4] != '-' and b[4] == '-':
            return -1
        if verb_ct[b[3].lemma_] > verb_ct[a[3].lemma_]:
            return 1
        if verb_ct[b[3].lemma_] < verb_ct[a[3].lemma_]:
            return -1
        return 0

    # List to store top 5 events
    events = []
    locs = filter(lambda e: e.label_=='GPE' or e.label_=='LOC',doc.ents)
    locs_final = []
    for loc in locs:
        for token in loc:
            if token.dep_ == 'pobj':
                locs_final.append(loc)
                break

    dates = [date for date in filter(lambda e: e.label_=='DATE',doc.ents)]
    for current in doc:
        if current.pos == VERB:
            location = '-'
            for loc in locs_final:
                temp = loc.root
                while(temp.dep_ != 'ROOT'):
                    temp = temp.head
                if temp == current:
                    location = str(loc)
                    break
            date = '-'
            for date_possible in dates:
                temp = date_possible.root
                while(temp.dep_ != 'ROOT'):
                    temp = temp.head
                if temp == current:
                    date = str(date_possible)
                    break
            subject = [' '+str(html.escape(dep_subtree(current,"nsubj")))+' ', ' '+str(html.escape(dep_subtree(current,"nsubjpass")))+' ']
            obj = " ".join(filter(None,[dep_subtree(current,"dobj")]))
            sentence = str(' '+html.escape(' '.join([str(token) for token in nlp(sents[token_sent[current.i]])])))
            for element in subject:
                if element != '':
                    sentence = sentence.replace(element, '<span style="color: #1e32e1">'+element+'</span>', 1)
            if obj == '':
                obj = '-'
            else:
                temp = str(html.escape(obj))
                sentence = sentence.replace(temp, '<span style="color: #5cbc43">'+temp+'</span>', 1)
            events.append([date, location, subject, current, obj, sentence])
    events = sorted(events, key=cmp_to_key(compare))[:5]
    summary['events'] = events
    # End of events code


    # Map of frequency of each token (required for word cloud)
    token_ct = {}
    for token in doc:
        if str(token).lower() not in STOP_WORDS and token.pos != PUNCT:
            token_ct[str(token)] = token_ct.get(str(token), 0)+1

    # Generate overall Word Cloud
    wordcloud = WordCloud(width=1000, height=500, max_words = 40, background_color='white', relative_scaling=1.0).generate_from_frequencies(token_ct)
    plt.title('Word Cloud')
    plt.tight_layout(pad=20)
    plt.imshow(wordcloud, interpolation="bilinear")
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode('ascii')
    summary['wordcloud'] = my_base64_jpgData

    return result, summary

# Helper function for events
def dep_subtree(token, dep):
    deps =[child.dep_ for child in token.children]
    child=next(filter(lambda c: c.dep_==dep, token.children), None)
    if child != None:
        return " ".join([c.text for c in child.subtree])
    else:
        return ""