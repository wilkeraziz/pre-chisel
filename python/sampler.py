"""
@author waziz
"""
import collections
import logging
import os
import sys
import argparse
import cdec
import gzip
import numpy as np
import re
from io_utils import read_config, read_weights, SegmentMetaData, fmap2str, str2fmap
from features import compute_feature

def build_proxy(input_str, grammar_file, weights_file, scaling):
    
    decoder = cdec.Decoder(formalism='scfg', intersection_strategy='Full', add_pass_through_rules='true', feature_function='WordPenalty')

    logging.info('Loading weights: %s', weights_file)
    decoder.read_weights(weights_file, scaling)
    #logging.info('Weights: %s', dict(decoder.weights))
    
    logging.info('Loading grammar: %s', grammar_file)
    with gzip.open(grammar_file) as f:
        grammar = f.read()

    logging.info('Composing the forest')
    forest = decoder.translate(input_str, grammar = grammar)
    return forest

def sample(forest, n):
    sampledict = collections.defaultdict(list)
    for sample_str, sample_dot, sample_fmap in forest.sample_hypotheses(n):
        sampledict[sample_str.encode('utf8')].append((dict(sample_fmap), sample_dot))
    return sampledict

def map_dot(fmap, wmap):
    return sum(fmap.get(fname, 0) * fweight for fname, fweight in wmap.iteritems())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'MC sampler for hiero models')
    parser.add_argument("proxy", type=str, help="feature weights (proxy model)")
    parser.add_argument("target", type=str, help="feature weights (target model)")
    parser.add_argument("config", type=str, help="config file")
    parser.add_argument("--proxy-scaling", type=float, default = 1.0, help = "scaling parameter for the proxy model") 
    parser.add_argument("--samples", type=int, default = 100, help = "number of samples") 
    parser.add_argument("--input-format", type=str, default='chisel', help="chisel (tab-separated columns: grammar source), cdec (sgml), moses (|||-separated columns: grammar source)")
    options = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s %(message)s') 
    
    proxy_weights = read_weights(options.proxy)
    target_weights = read_weights(options.target)
    config = read_config(options.config)
    resources = {}
    extra_features = {k:v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.info('Extra features: %s', extra_features)

    for line in sys.stdin:
        # parses input format
        segment = SegmentMetaData.parse(line.strip(), options.input_format)
        # builds the proxy distribution
        forest = build_proxy(segment.src_, segment.grammar_, options.proxy, options.proxy_scaling)
        # samples from the proxy distribution
        samples = sample(forest, options.samples)
        # for now we do not have access to alignment  
        for sample_str, sample_info in sorted(samples.iteritems(), key = lambda pair : len(pair[1]), reverse = True):
            # computes additional features
            extraff = {}
            for fname, fweight in extra_features.iteritems():
                extraff[fname] = compute_feature(config, resources, fname, sample_str)
            # groups vectors associated with equivalent derivations
            counter = collections.Counter(frozenset(fmap.iteritems()) for fmap, _ in sample_info)
            # compute target vectors
            qdots, pdots = [], []
            for fpairs, count in counter.iteritems():
                # features that are reused from the proxy
                qmap = dict(fpairs)
                pmap = {fname:fvalue for fname, fvalue in fpairs if fname in target_weights}
                # additional features
                for fname, fvalue in extraff.iteritems():
                    pmap[fname] = fvalue
                # target score
                pdot = map_dot(pmap, target_weights)
                # proxy score
                qdot = map_dot(qmap, proxy_weights)
                # output info
                output = [str(count), 
                        sample_str,
                        fmap2str(fpairs),
                        str(qdot),
                        fmap2str(pmap.iteritems()),
                        str(pdot)]
                print ' ||| '.join(output)
                qdots.append(qdot)
                pdots.append(pdot)
