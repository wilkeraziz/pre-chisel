"""
@author waziz
"""
import kenlm

#TODO: more flexible/elegant ff framework (like cdec-features)

def KLanguageModel(config, resources, sample_str):
    model = resources.get('KLanguageModel', None)
    if model is None:
        model = kenlm.LanguageModel(config['KLanguageModel'])
        resources['KLanguageModel'] = model
    return model.score(sample_str)

def compute_feature(config, resources, fname, sample_str):
    if fname == 'KLanguageModel':
        return KLanguageModel(config, resources, sample_str)
    #raise Exception('Unknown feature: %s' % fname)
    return 0

def FF(config, resources, sample_str):
    """
    config stores parameters you might need such as paths to models, etc.
    resources stores models (you can structure it as you like)
    samples_str is the translation (as a string)
    """
    pass
