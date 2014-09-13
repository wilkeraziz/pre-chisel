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

@ff.features('LanguageModel', 'LanguageModel_OOV')
def KLanguageModel(hypothesis):
    total_prob = 0
    total_oov = 0
    for prob, length, oov in model.full_scores(hypothesis.translation_):
        total_prob += prob
        total_oov += oov
    return (total_prob, total_oov)

