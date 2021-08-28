from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from .Helpers.helpers2 import *


import re


class verb():
	def __init__(self, name, info):
		self.name = name
		self.info = info
class entity():
	def __init__(self, name, mentions, typee, sentiment, on, by, image, positive_sent, negative_sent):
		self.name = name
		self.mentions = mentions
		self.typee = typee
		self.sentiment = sentiment
		actions_on = []
		for word in list(on.keys()):
			actions_on.append(verb(word, on[word]))
		self.on = actions_on
		actions_by = []
		for word in list(by.keys()):
			actions_by.append(verb(word, by[word]))
		self.by = actions_by
		self.image = image
		self.positive_sent = positive_sent
		self.negative_sent = negative_sent
		self.actions_on_ct = len(actions_on)
		self.actions_by_ct = len(actions_by)
		self.positive_sent_ct = len(positive_sent)
		self.negative_sent_ct = len(negative_sent)

def index(request):
	return render(request, 'vis/home.html')


def display(request):
	if(request.method == 'POST'):
		text = request.POST.get('comment')
		islink = request.POST.get('islink')
		glove = loadGloveModel(path+'/vis/Data/glove_small.txt')
		model = init_model(glove)
		if islink == '1':
			text = get_cnn(text)
		result, summary = vis(model, text)
		entities = []
		context = {}
		for key in list(result.keys()):
			entities.append(entity(result[key]['name'], result[key]['count'], result[key]['type'], np.round(result[key]['sentiment'], 2), result[key]['on'], result[key]['by'], result[key]['image'], result[key]['positive_sent'], result[key]['negative_sent']))
		context['entities'] = entities
		context['summary'] = summary['pprint']
		context['events'] = summary['events']
		context['wordcloud'] = summary['wordcloud']
		context['mentions'] = summary['mentions']
		context['words'] = summary['words']
		context['sents'] = summary['sents']
		context['sent_lens'] = summary['sent_lens']
		context['sent_sentiment'] = summary['sent_sentiment']
		context['plot_sentiment'] = summary['plot_sentiment']
		context['plot_len'] = summary['plot_len']
		context['text_type'] = summary['text_type']
		context['related'] = summary['related']
		context['related_ct'] = len(summary['related'])
		context['positive_sent'] = summary['positive_sent']
		context['negative_sent'] = summary['negative_sent']
		context['positive_sent_ct'] = len(summary['positive_sent'])
		context['negative_sent_ct'] = len(summary['negative_sent'])
		context['sent_ct'] = len(summary['positive_sent']) + len(summary['negative_sent'])
		context['agency'] = summary['agency']
		context['is_rfp'] = summary['is_rfp']
		context['time_series'] = summary['time_series']
		context['pie_share'] = summary['pie_share']
		context['topic'] = summary['topic']
		context['codes'] = summary['codes']
		context['is_person'] = summary['is_person']
		context['person'] = summary['person']
		context['pubs'] = summary['pubs']
		return render(request, 'vis/display.html', context)
	else:
		return render(request, 'vis/home.html')