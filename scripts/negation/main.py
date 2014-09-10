#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import codecs
from client import client
import logging
import json

class Negation:

	def __init__(self):
		pass

	def get_dependencies(self,string,lang):
		deps = client({'sentence':string,'lang':lang})
		unpackedDeps = json.loads(json.loads(deps)['result'])
		#TODO:the lexical elements are to be decoded in utf-8
		return deps


def define_logging():

	logging.basicConfig(filename='../logs/negation.log',filemode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__=="__main__":

	define_logging()
	n = Negation()
	sent = u'我 爱 你 的 裤子'.encode('utf-8')
	n.get_dependencies(sent,'zh')



















# """extract dependencies from hypothesis"""
# def extract_deps(dict_reference,n_best,dict_indeces):

# 	rank=0

# 	best_rank=0
# 	"""Save the index of the hypothesis currently considered"""
# 	index=0
# 	"""Save the sentence so not to parse the same again"""
# 	hyp_sentence=list()

# 	"""Save the partial best score (starts from None cause 0 is > than None, so at least we get a sentence)"""
# 	partial_score=None
# 	"""Save the partial best sentence"""
# 	best_sentence=list()
	
# 	parser_file = '/mnt/buri4/federico/stanford-parser-2013-04-05/models/englishPCFG.ser.gz'
# 	sp = stanford.StanfordParser(parser_file)

# 	for line in n_best:
		
# 		total_dependencies=list()
# 		spl=line.split(' ||| ')
# 		number,sentence=int(spl[0]),spl[1]

# 		if sentence not in set(hyp_sentence):		

# 			if number!=index:
# 				"""Flush out the reranked sentence of the index"""
# 				#print
# 				print best_sentence[0].encode("ascii","ignore").replace('&apos;',"'")+' ||| '+str(partial_score)+' ||| '+str(best_rank)
# 				#print 
# 				"""Refresh all values for the new index"""
# 				index=number
# 				#print index
# 				hyp_sentence=[]
# 				partial_score=None
# 				best_sentence=[]
# 				rank=0
# 				best_rank=0


# 			ascii_sentence=sentence.encode("ascii","ignore").replace('&apos;',"'")
# 			hyp_sentence.append(sentence)
# 			try:
# 				dependencies,word_list=process_line(sp,ascii_sentence)
# 			except:
# 				pass

# 			"""Get the indeces"""
# 			ref_index=dict_indeces[index]


# 			"""Calculate the scores"""
# 			ref_deps=dict_reference[unicode(ref_index)]
# 			###{index:[ ( [[]], [[]] ) ]} --> [ ( [ [1] , [2] ], [ [1] , [2] ] ) ]

# 			max_score=list()


# 			for tuple_ in ref_deps:
# 				"""Controls those tuples that contains empty lists"""
# 				if tuple_[0]==[] and tuple_[1]==[]:max_score.append(1)
# 				###tuple_ : ( [ [1] , [2] ], [ [1] , [2] ] )
# 				else:
# 					score_deps_word = score_dependencies(tuple_,word_list,dependencies)
# 					max_score.append(score_deps_word)


# 			score=max(max_score)

# 			"""See whether the new score in higher than the previous score"""
# 			if score>partial_score:
# 				partial_score=score
# 				best_sentence=[sentence]
# 				best_rank=rank
			
# 			rank+=1
# 		else: rank+=1

# 	f.close()

# """Get the indices for the sentences and returns a dictionary with the correct indenes"""
# def get_indeces(filename):
# 	dict_indeces=pickle.load(open(filename,'r'))

# 	indeces=dict()
# 	v=0
	
# 	for k in range(0,len(dict_indeces.keys())):
# 		indeces.setdefault(k,list()).extend(range(v,v+dict_indeces[k]))
		
# 		v+=dict_indeces[k]

# 	indeces_reversed=dict()

# 	for item in indeces.items():
# 		list_reversed=map(lambda x: (x,item[0]), item[1])
# 		indeces_reversed.update(list_reversed)
		
# 	return indeces_reversed
		
# ####FREEZE FOR A MOMENT###############	

# """score the dependencies
# def score_dependencies(tuple_,word_list,hyp_deps):
	
# 	head_neg=tuple_[0]
# 	words=tuple_[1]

# 	tot_score = list()

# 	for i in range(0,len(head_neg)):

# 		print head_neg
# 		print hyp_deps
# 		print 
# 		if hyp_deps==[] or head_neg==[]:
# 			if hyp_deps!=[] or head_neg!=[]: 
# 				tot_score.append(0)
# 			else: 
# 				tot_score.append(1)
			
# 		else:
# 			wrap=list(hyp_deps)
# 			score_deps=len(set(wrap).intersection(set(head_neg)))
# 			print score_deps
# 			n_gram=score_ngrams(words[i],word_list)	
			
# 			print score_deps+n_gram
# 			tot_score.append(score_deps+n_gram)
			
# 	print tot_score	

# 	return max(tot_score)
# """

# def score_dependencies(tuple_,word_list,hyp_deps):
	
# 	"""Three possibilities:"""
# 	"""1-one of the two is an empty list and the other is not"""
# 	"""Both lists are empty, perfect match = score of 1"""
# 	"""We can easily cope with the other cases by just set intersection"""
# 	"""See the intersected dependency, go and look the corresponding node in the reference"""

# 	head_neg=tuple_[0]
# 	words=tuple_[1]

# 	tot_score = list()

	
# 	if hyp_deps==[] or head_neg==[]:
# 		if hyp_deps!=[] or head_neg!=[]: 
# 			tot_score.append(0)
# 		else: 
# 			tot_score.append(1)
			
# 	else:
# 		wrap=list(hyp_deps)
# 		score_deps=len(set(wrap).intersection(set(head_neg)))
# 		word_score=int()
# 		for item in list(set(wrap).intersection(set(head_neg))):
# 			index=head_neg.index(item)

# 			n_gram=score_ngrams(words[index],word_list)
# 			word_score+=n_gram	
			
# 		#print score_deps+word_score
# 		tot_score.append(score_deps+word_score)
			
# 	#print tot_score	

# 	return max(tot_score)

# def score_ngrams(ref_list,list_word):
	
# 	if list_word==None or ref_list==[]:
# 			return 0
# 	else:
# 		ngram_range=range(1,len(list_word)+1)
# 		score_weights=map(lambda x: round(x/reduce(lambda x,y:x+y,ngram_range),4),ngram_range)
	
# 		score_words=float()
# 		for i in ngram_range:
# 			hyp=nltk.util.ngrams(list_word,i)
# 			ref=nltk.util.ngrams(ref_list,i)
# 			"""Give a penalty according to the length difference"""
# 			if len(ref_list)==len(list_word): penalty=0
# 			else: penalty=1.0-(1.0/abs(len(ref_list)-len(list_word)))
# 			score_words+=(len(set(hyp).intersection(set(ref)))*score_weights[i-1])-penalty
		
# 	return score_words


# def process_line(sp,sentence):

# 	dependencies=stanford.get_dependencies_example(sp,sentence)
# 	word_list=stanford.parse_xml_path2s(sp,sentence)	
	
# 	return dependencies,word_list


# if __name__=="__main__":
# 	"""Insert the path to the reference file"""
# 	print "Reference loading..."
# 	dict_reference=pickle.load(open('dictionary_ref','r'))
# 	print "Reference loaded!"
	
# 	dict_indeces=get_indeces(sys.argv[1])
# 	n_best=codecs.open(sys.argv[2],'r',encoding='utf-8-sig')
# 	extract_deps(dict_reference,n_best,dict_indeces)
	