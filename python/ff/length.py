"""
A feature function that captures the length of a complete hypothesis.

@author waziz
"""
import ff

@ff.features('FLength', 'ELength', 'LengthRatio', 'LengthDiff')
def ELength(hypothesis): 
    f, e = hypothesis.source_.split(), hypothesis.translation_.split()
    return (len(f), len(e), float(len(e))/len(f), len(e) - len(f))

