"""
@author waziz
"""
import ff
import kenlm

model = None

@ff.configure
def configure(config):
    global model
    model = kenlm.LanguageModel(config['KLanguageModel'])

@ff.feature
def KLanguageModel(hypothesis):
    return model.score(hypothesis.translation_)
