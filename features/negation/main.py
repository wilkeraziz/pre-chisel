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
		logging.info(unpackedDeps)
		return unpackedDeps

	def score_deps(self,sentDeps1,sentDeps2):		
		negDeps1 = filter(lambda x: x.keys()[0]=='neg',sentDeps1)
		negDeps2 = filter(lambda x: x.keys()[0]=='neg',sentDeps2)
		negCues1,negCues2 = self.extractCues(negDeps1,negDeps2)
		negEvents1,negEvents2 = self.extractEvents(negDeps1,negDeps2)

		cueScore = self.scoreCueOverlap(negCues1,negCues2)
		eventScore = self.scoreEventOverlap(negEvents1,negEvents2)
		scopeScore = self.scoreScopeOverlap(None,None)

	def extractCues(negDeps1,negDeps2):
		negCues1 = map(lambda x: x['dep'],negDeps1)
		negCues2 = map(lambda x: x['dep'],negDeps2)
		#plug HERE translation probabilties if langs are different

		return negCues1,negCues2

	def extractEvents(negDeps1,Deps2):
		negEvents1 = map(lambda x: x['gov'],negDeps1)
		negEvents2 = map(lambda x: x['gov'],negDeps2)
		#plug HERE translation probabilties if langs are different

		return negEvents1,negEvents2

	#???
	# def extractScope():
	# 	pass

	def scoreCueOverlap(cues1,cues2):
		return 0

	def scoreEventOverlap(events1,events2):
		return 0

	def scoreScopeOverlap(scope1,scope2):
		return 0
	



def define_logging():

	logging.basicConfig(filename='negation.log',filemode='w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__=="__main__":

	define_logging()
	n = Negation()
	sent = u'我 爱 你 的 裤子'.encode('utf-8')
	n.get_dependencies(sent,'zh')