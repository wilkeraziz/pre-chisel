"""
@author waziz
"""
import logging
import collections
import os
import sys
import argparse
import re
import itertools
import math
from consensus import expected_bleu, expected_linear_bleu, consensus_bleu
from consensus import BLEUSufficientStatistics
from bleu import BLEU
from semiring import CountSemiring, MaxTimesSemiring, SumTimesSemiring

def argmax(samples, G):
    """
    @return (argmax, max)
    """
    h, gain = max(enumerate(G), key = lambda pair : pair[1])
    return (samples[h], gain)

def argmin(samples, R):
    """
    @return (argmin, min)
    """
    h, risk = max(enumerate(R), key = lambda pair : pair[1])
    return (samples[h], risk)

def sort(samples, scores, utility = True):
    """
    Sorts the samples on the basis of their utility/cost scores.
    @param samples list of samples
    @param scores list of scores
    @param utility = True means maximisation, otherwise it is a cost, thus minimisation
    @return sorted list of pairs (sample, score)
    """
    return tuple((samples[h], score) for h, score in sorted(enumerate(scores), key = lambda pair : pair[1], reverse = utility))
    
class FVector(object):
    """
    A feature vector (in reality it is more like a map).
    Each component (identified by an integer key) is associated with a vector of real-valued features.
    Every iterator produced by this class is sorted by key.

    Iterating with iteritems returns pairs (key, value) where the key is the integer
    identifier of the component and the value is a tuple of feature values.

    Iterating with __iter__ returns the feature values in order (as if this was a normal vector).
    """

    def __init__(self, vecstr):
        # disambiguate spaces (separate components by tabs)
        vecstr = re.sub(r' ([^ ]+=)', r'\t\1', vecstr)
        pairs = vecstr.split('\t')
        fmap = {}
        # parse each component's features
        for pair in pairs:
            strkey, strvalue = pair.split('=')
            strvalue = re.sub('[][()]', '', strvalue)
            fmap[int(strkey)] = tuple(float(v) for v in strvalue.split())
        self.fmap_ = fmap
        self.keys_ = tuple(sorted(fmap.iterkeys()))
        self.values_ = tuple(v for k, v in sorted(fmap.iteritems(), key = lambda pair : pair[0]))
        self.as_tuple_ = tuple(itertools.chain(*self.values_))

    @property
    def keys(self):
        return self.keys_

    @property
    def values(self):
        return self.values_

    @property
    def as_tuple(self):
        return self.as_tuple_

    def __getitem__(self, key):
        return self.fmap_[key]

    def __contains__(self, key):
        return key in self.fmap_

    def iteritems(self):
        return self.fmap_.iteritems()

    def iterkeys(self):
        return self.fmap_.iterkeys()

    def itervalues(self):
        return self.fmap_.itervalues()

    def n_components(self):
        return len(self.fmap_)

    def n_features(self):
        return sum(len(vec) for vec in itervalues(self))

    def __str__(self):
        return ' '.join(('%s=%s' % (k, str(v[0]))) if len(v) == 1 else ('%s=(%s)' % (k, ' '.join(str(x) for x in v))) for k, v in self.iteritems())


class Derivation(object):
    """
    A tree-like structure that represents a sequence of translation rules.
    """
    
    def __init__(self, derivationstr):
        """
        Parses a Moses-style string, e.g. "a b |0-0| c |3-4| d |2-2|"
        """
        strsegments = re.findall(r'[|][0-9]+-[0-9]+[|]', derivationstr)
        alignment_pattern = re.compile(r'[|][0-9]+-[0-9]+[|]')
        offset, tgt, src = None, None, None
        leaves = []
        derivation = []
        for span in re.split(r' *([|][0-9]+-[0-9]+[|]) *', derivationstr):
            if span.strip() == '':
                continue
            if alignment_pattern.match(span) is not None:
                strpair = re.sub(r'[|]', '', span.strip()).split('-')
                src = (int(strpair[0]), int(strpair[1]))
            else:
                offset = len(leaves)
                tgt = tuple(span.split())
                leaves.extend(tgt)
            if offset is not None and tgt is not None and src is not None:
                derivation.append((offset, tgt, src))
                offset, tgt, src = None, None, None

        self.str_ = derivationstr
        self.tree_ = tuple(derivation)
        self.leaves_ = tuple(leaves)

    @property
    def tree(self): # TODO: structure it like a tree.
        """
        @return tuple of bi-spans
        a bi-span is itself a tuple (target offset, target words, source span)
        'source span' is a pair (from, to)
        """
        return self.tree_
    
    @property
    def leaves(self):
        """
        @return tuple of words
        """
        return self.leaves_

    @property
    def projection(self):
        return ' '.join(self.leaves_)

    @property
    def bracketed_projection(self):
        return self.str_

    def __hash__(self):
        return hash(self.str_)

    def __eq__(self, other):
        return self.str_ == other.str_

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return self.str_


class Solution(object):
    """
    A scored translation derivation (sampled a number of times)
    """
    
    def __init__(self, derivation, vector, score, count):
        self.derivation_ = derivation
        self.vector_ = vector
        self.score_ = score
        self.count_ = count

    @property
    def derivation(self):
        return self.derivation_
    
    @property
    def vector(self):
        return self.vector_

    @property
    def score(self):
        return self.score_

    @property
    def count(self):
        return self.count_

    def __hash__(self):
        return hash(self.derivation_)

    def __eq__(self, other):
        return self.derivation_ == other.derivation_

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return '%s\t%s\t%s\t%s' % (self.score_, self.count_, self.derivation_, self.vector_)

class Group(object):
    """
    A group is a list of solutions where every solutions has the same "key"
    """

    def __init__(self, solutions, key):
        """
        @param solutions
        @param key is the group's unique key
        """
        if len(solutions) == 0:
            raise Exception('A group cannot be empty')
        # get the key that indexes the group
        self.solutions_ = []
        self.mykey_ = key
        # add solutions
        self.solutions_.extend(solutions)

    @property
    def key(self):
        return self.mykey_

    def __len__(self):
        return len(self.solutions_)

    def __iter__(self):
        return iter(self.solutions_)

    def leaves(self):
        assert len(self.solutions_) > 0, "A group can never be empty"
        return self.solutions_[0].derivation.leaves

    def count(self, op = CountSemiring.sum):
        """
        @param op represents sum in a given semiring (e.g. CountSemiring.sum)
        @return total count according to op
        """
        return reduce(op, (sol.count for sol in self.solutions_))

    def score(self, op = SumTimesSemiring.sum):
        """
        @param op represent sum in a given semiring (e.g. SumTimesSemiring.sum)
        @return total score according to op
        """
        return reduce(op, (sol.score for sol in self.solutions_))

class Sample(object):

    def __init__(self, leaves, count, norm_count, score, norm_score):
        self.leaves_ = leaves
        self.projection_ = ' '.join(leaves)
        self.count_ = count
        self.normalised_count_ = norm_count
        self.score_ = score
        self.normalised_score_ = norm_score

    @property
    def leaves(self):
        return self.leaves_

    @property
    def count(self):
        return self.count_

    @property
    def score(self):
        return self.score_

    @property
    def normcount(self):
        return self.normalised_count_

    @property
    def normscore(self):
        return self.normalised_score_

    def __hash__(self):
        return hash(self.leaves_)

    def __eq__(self, other):
        return self.leaves_ == other.leaves_

    def __ne__(self, other):
        return self.leaves_ != other.leaves_

    def __str__(self):
        return ' '.join(self.leaves_)

def read_solutions(fi):
    """
    Parse a file containing samples. The file is structured as a table (tab-separated columns).
    The first line contains the column names.
    We expect at least a fixed set of columns (e.g. derivation, vector, score, count).
    The table must be grouped by derivation.
    @return list of solutions
    """
    logging.info('reading from %s', fi)
    # get the column names
    raw = next(fi)
    if not raw.startswith('#'):
        raise Exception('missing header')
    colnames = [colname.replace('#', '') for colname in raw.strip().split('\t')]
    needed = frozenset('derivation vector score count'.split())
    # sanity check
    if not (needed <= frozenset(colnames)):
        raise Exception('missing columns: %s' % ', '.join(needed - frozenset(colnames)))
    logging.info('%d columns: %s', len(colnames), colnames)
    # parse rows
    S = []
    for row in (raw.strip().split('\t') for raw in fi) :
        k2v = {key:value for key, value in zip(colnames, row)}
        sol = Solution(derivation = Derivation(k2v['derivation']),
                vector = FVector(k2v['vector']),
                score = float(k2v['score']),
                count = int(k2v['count']))
        S.append(sol)
    logging.info('%d rows', len(S))
    return S

def make_samples(groups, C, S):
    """
    Creates a list of Sample objects, one per group.
    @param groups is a list of Group objects
    @param C is the count semiring
    @parma S is the score semiring
    """
    # reduce counts and scores within a group
    sums = [(group.count(C.sum),group.score(S.sum)) for group in groups]
    # computes the normalisation contants (for count N, and scores Z)
    N, Z = reduce(lambda x, y: (C.sum(x[0], y[0]), S.sum(x[1], y[1])), sums)
    # create samples
    samples = []
    for group, pair in itertools.izip(groups, sums):
        count, score = pair
        sample = Sample(group.leaves(), 
                count, 
                C.division(count, N),
                score,
                S.division(score, Z))
        samples.append(sample)
    return samples

def groupby(solutions, key):
    key2group = collections.defaultdict(list)
    [key2group[key(sol)].append(sol) for sol in solutions]
    return [Group(group, k) for k, group in key2group.iteritems()]

def singletons(solutions):
    return [Group([sol], k) for k, sol in enumerate(solutions)]

def MAP(samples):
    return tuple(sample.normcount for sample in samples)

def viterbi(samples):
    return tuple(sample.score for sample in samples)

def print_nbest(samples, scores, score_type, nbest, out = sys.stdout):
    for sample, score in sort(samples, scores)[:nbest]:
        print >> out, 'prob=%s\t%s=%s\t%s' % (sample.normcount, score_type, score, sample)

def derivation_max_times(solutions, options):
    """
    Decision rules in MaxTimes.
    """
    groups = singletons(solutions)
    logging.info('%d (not necessarily unique) derivations', len(groups))
    samples = make_samples(groups, CountSemiring, MaxTimesSemiring)
    scores = viterbi(samples)
    print 'Viterbi: derivation (MaxTimes)'
    print_nbest(samples, scores, 'score', options.nbest)

def derivation_sum_times(solutions, options):
    groups = groupby(solutions, lambda sol : sol.derivation.bracketed_projection) 
    logging.info('%d unique derivations', len(groups))
    samples = make_samples(groups, CountSemiring, SumTimesSemiring)

    #print 'MAP: derivation (SumTimes)'
    #map_probs = MAP(samples)
    #print_nbest(samples, map_probs, 'prob', options.nbest)

    if options.viterbi:
        print 'Viterbi: derivation (SumTimes)'
        viterbi_scores = viterbi(samples)
        print_nbest(samples, viterbi_scores, 'score', options.nbest)

def string_sum_times(solutions, options):
    groups = groupby(solutions, lambda sol : sol.derivation.projection)
    logging.info('%d unique strings', len(groups))
    samples = make_samples(groups, CountSemiring, SumTimesSemiring) 
    
    if options.map:
        print 'MAP: string'
        map_probs = MAP(samples)
        print_nbest(samples, map_probs, 'prob', options.nbest)

    if options.mbrbleu or options.mbrlbleu or options.conbleu:
        bleusuff = BLEUSufficientStatistics(samples)

        if options.conbleu:
            print 'Consensus: exact BLEU'
            cb_gains = consensus_bleu(samples, bleusuff, BLEU.get(options.metric))
            print_nbest(samples, cb_gains, 'bleu', options.nbest)

        if options.mbrlbleu:
            print 'MBR: linear bleu'
            lb_gains = expected_linear_bleu(samples, bleusuff, options.T, options.p, options.r)
            print_nbest(samples, lb_gains, 'gain', options.nbest)

        if options.mbrbleu:
            print 'MBR: exact BLEU'
            eb_gains = expected_bleu(samples, bleusuff, BLEU.get(options.metric))
            print_nbest(samples, eb_gains, 'gain', options.nbest)
    
if __name__ == '__main__':
    
    # TODO: 
    # * Consensus string (DeNero)?

    parser = argparse.ArgumentParser(description = 'Applies a decision rule to a sample.')
    parser.add_argument("--viterbi", action='store_true', help = "Viterbi (best derivation)")
    parser.add_argument("--map", action='store_true', help = "MAP solution")
    parser.add_argument("--mbrbleu", action='store_true', help = "MBR (exact BLEU)")
    parser.add_argument("--mbrlbleu", action='store_true', help = "MBR (linear BLEU) see -T, -p and -r")
    parser.add_argument("--conbleu", action='store_true', help = "Consensus (exact BLEU)")
    parser.add_argument("--metric", type=str, default = 'ibm_bleu', help = "similarity function, one of {ibm_bleu, bleu_p1, unsmoothed_bleu}")
    parser.add_argument("--nbest", type=int, default = 1, help = "number of solutions") 
    parser.add_argument("-T", type=float, default = 1.0, help = "average number of unigram tokens (linear BLEU). Note: the MBR solution is insensitive to T") 
    parser.add_argument("-p", type=float, default = 0.85, help = "average unigram precision (linear BLEU)") 
    parser.add_argument("-r", type=float, default = 0.7, help = "average decay ratio (linear BLEU)") 
    options = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s %(message)s') 

    solutions = read_solutions(sys.stdin)

    #derivation_max_times(solutions, options)
    if options.viterbi:
        derivation_sum_times(solutions, options)
    string_sum_times(solutions, options)

