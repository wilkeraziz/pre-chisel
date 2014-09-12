import logging
import cdec.sa
"""
This implements a simple LM feature compatible with grammar extraction in cdec.

The LM feature simulates something that Moses does. Basically, it incorporates LM probabilites for the phrases
so that only boundary-crossing ngrams need to be reweighted in forest rescoring.
This means that if no rescoring happens, the resulting proxy is a lot better than the forest without an LM component
(or with a unigram component).

@author waziz
"""

import kenlm
import re
import itertools

model = None
nt_pattern = re.compile('\[[^]]+\]')

@cdec.sa.configure
def configure(config):
    """here we load the language model"""
    global model
    logging.info('Loading ngam LM model: %s', config['NGramLMModel'])
    model = kenlm.LanguageModel(config['NGramLMModel'])

@cdec.sa.annotator
def annotate(sentence):
    """nothing to be done to the input sentence"""
    pass

@cdec.sa.feature
def NGramLM(ctxt):
    """here we assess the LM feature: each segment is scored independently"""
    return sum(model.score(' '.join(span), False, False) for span, _ in itertools.ifilter(lambda pair: pair[1], ctxt.ephrase.iterspans(nt_flag = False, to_str = True)))
    
