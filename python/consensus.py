"""
@author waziz
"""
import itertools
import collections
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import math
from bleu import BLEU

class NGramStats(object):
    """
    Represents the statistics associated with a certain ngram
    """

    def __init__(self, d):
        """
        @param d dimensionality of the vector of counts (size of the evidence set)
        """
        self.counts = [0] * d
        self.posterior = 0

class CountStats(object):
    """
    Represents the statistics associated with each class of ngram.
    """

    def __init__(self, n):
        """
        @param n maximum ngram order
        """
        # cn[0] is the length
        # cn[k] k > 0 is the number of kgrams of order n
        self.cn = [0] * (n + 1)

class InefficientClippedCounts(object):
    """
    Pre-computes the clipped counts between every two sentences.
    O(|E|^2*|N|) where E is the evidence set and N is the set of kgrams (1<=k<=n) occurring in E.
    """

    def __init__(self, ngramstats, countstats, n = 4):
        # size of the evidence set
        M = len(countstats)

        self.pn_ = collections.defaultdict(lambda : [0] * (n + 1))
        for i in xrange(M):
            for j in xrange(i + 1):
                for ngram, stats in ngramstats.iteritems():
                    count = min(stats.counts[i], stats.counts[j]) 
                    self.pn_[(i,j)][len(ngram)] += count # clipped counts

    def counts(self, i, j):
        """
        Returns a vector C such that C[k], k=1..n, is the clipped counts for kgrams between samples i and j
        """
        return self.pn_[(i, j)] if j < i else self.pn_[(j, i)]
    
class EfficientClippedCounts(object):
    """
    Pre-computes the clipped counts between every two sentences.
    O(|E|^2*|I|) where E is the evidence set and I is the average hypothesis length.
    In comparison to InefficientClippedCounts this is faster because there is a direct map from samples to ngrams.
    """

    def __init__(self, snt2ngrams, n = 4):
        """
        Constructs the clipped counts from a list of dictionaries.
        Each element of the list corresponds to a sample i.
        The dictionary store key-value pairs of the type (ngram, occurrences in i).
        """
        # size of the evidence set
        M = len(snt2ngrams)

        self.cc_ = []
        for i in xrange(M):
            # this is a row with i+1 columns, each column has n+1 cells 
            # each row represents a sample
            # each column represents another
            # each cell represents the clipped count for a certain ngram order
            self.cc_.append( [[0] * (n+1) for _ in xrange(i+1)] ) 
            # row
            cci = self.cc_[-1]  
            # count for columns (just the lower half of the matrix)
            for j in xrange(i + 1):
                # d1 is the smaller set, d2 is the larger one
                (d1, d2) = (snt2ngrams[i], snt2ngrams[j]) if len(snt2ngrams[i]) <= len(snt2ngrams[j]) else (snt2ngrams[j], snt2ngrams[i])
                # clip counts
                for w, c1 in d1.iteritems():
                    cci[j][len(w)] += min(c1, d2.get(w, 0))

    def counts(self, i, j):
        """
        Returns a vector C such that C[k] is the clipped counts for kgrams between samples i and j
        """
        return self.cc_[i][j] if j < i else self.cc_[j][i]

class BLEUSufficientStatistics(object):

    def __init__(self, samples, n = 4):
        """
        @param samples is an evidence set (where we gather ngram counts from) encoded as a list of Sample objects
        @param maximum ngram order (for standard BLEU this is 4)

        This computes:

        1) ngramstats: a dictionary such that each key is a kgram (k in [1,4]) and the value is an NGramStats object.
        Each NGramStats object contains a vector of M=len(samples) counts representing the number of occurrences 
        of the ngram in each sample of the evidence set. It also contains the n-gram posterior probability computed
        as a function of normalised counts or scores.

        2) counstats: a list such that each element is a CountStats object.
        Each object summarises the counts for a certain kgram class (1 <= k <= n determines the class).

        3) snt2ngrams: a list of dictionaries, each dictionary is associated with a sample, its key-value pairs represent
        an ngram and how many times it occurs in the sample.
        This is somewhat redundant with (1) however it speeds up the computation of clipped counts (for exact BLEU).

        4) clippedcounts: an object of the type ClippedCounts which provides clipped counts (computed from (1) or (3) above).
        This object is lazily constructed.
        """

        M = len(samples)
        self.maxorder_ = n
        self.ngramstats = collections.defaultdict(lambda : NGramStats(M))
        self.countstats = [CountStats(n) for _ in samples]
        self.snt2ngrams = [collections.defaultdict(int) for _ in samples]
        self.clippedcounts = None
        ngramstats, countstats, snt2ngrams = self.ngramstats, self.countstats, self.snt2ngrams

        # map ngrams into integers
        self.nid_ = 0
        def new_id():
            self.nid_ += 1
            return self.nid_
        self.ngram2int = collections.defaultdict(new_id)

        # for each sample in the evidence set
        for i, sample in enumerate(samples):
            # first count position is the length
            # the kth postion (1 <= k <= n) is associated with ngrams of length k
            countstats[i].cn[0] = len(sample.leaves)
            # count kgrams for k=1..n
            context = collections.deque()
            for w in sample.leaves:
                # left trim the context
                if len(context) == n:
                    context.popleft()
                # count ngrams ending in w
                ngram = collections.deque([w])
                # process w
                # nid = self.ngram2int[tuple(ngram)]
                ngramstats[tuple(ngram)].counts[i] += 1
                countstats[i].cn[1] += 1
                snt2ngrams[i][tuple(ngram)] += 1
                # process ngrams larger than w
                for h in context:
                    ngram.appendleft(h)
                    # nid = self.ngram2int[tuple(ngram)]
                    ngramstats[tuple(ngram)].counts[i] += 1
                    countstats[i].cn[len(ngram)] += 1
                    snt2ngrams[i][tuple(ngram)] += 1
                context.append(w)

        # pre-computes ngram posteriors
        for ngram, stats in ngramstats.iteritems():
            # TODO: generalise to semiring.sum and value(sample)
            stats.posterior = sum(sample.normcount for i, sample in enumerate(samples) if stats.counts[i] > 0)

    @property
    def maxorder(self):
        return self.maxorder_
        
    def length(self, h):
        return self.countstats[h].cn[0] # candidate length
    
    def tc(self, h):
        """
        Total counts in h for all orders
        """
        return self.countstats[h].cn
                
    def tcn(self, h, n):
        """
        Total counts of order n in h
        """
        return self.countstats[h].cn[n]
   
    def compute_clipped_counts(self, efficient = True):
        if efficient:
            self.clippedcounts = EfficientClippedCounts(self.snt2ngrams, self.maxorder_)
        else:
            self.clippedcounts = InefficientClippedCounts(self.ngramstats, self.countstats, self.maxorder_)
        return self.clippedcounts

    def cc(self, h, r):
        """
        Return clipped counts between h and r for all orders.
        This assumes compute_clipped_counts was invoked.
        @return C such that C[k] is the clipped count for k-grams
        """
        return self.clippedcounts.counts(h, r)

    def ccn(self, h, r, n):
        """
        Clipped counts of order n between h and r
        This assumes compute_clipped_counts was invoked.
        @return clipped count
        """
        return self.clippedcounts.counts(h, r)[n]

    def ngrams(self, h):
        return self.snt2ngrams[h]

def expected_bleu(samples, bleusuff, bleu = BLEU.ibm_bleu, importance = lambda sample : 1.0, efficient = True):
    """
    Computes the expected (exact) BLEU of each candidate.
    @param samples is the candidates (also the evidence set)
    @param ngramstats, countstats (see count_ngrams)
    @param n max ngram order
    @return a list of pairs of the kind (sample, expected bleu) sorted from best to worst
    """
    # size of the evidence set
    M = len(samples)
    G = [0.0] * M
    n = bleusuff.maxorder
    # compute the exact clipped counts (intersection between the candidate and each evidence)
    bleusuff.compute_clipped_counts(efficient)
    for h, d in enumerate(samples):
        for r in xrange(M):
            # compute BLEU
            b = bleu(r = bleusuff.length(r), 
                    c = bleusuff.length(h), 
                    cc = bleusuff.cc(h, r), 
                    tc = bleusuff.tc(h),
                    n = bleusuff.maxorder)
            # accumulate gain
            G[h] += b * samples[r].normcount * importance(samples[r])
    return G

def expected_linear_bleu(samples, bleusuff, T = 1, p = 0.85, r = 0.7):
    """
    Computes the expected linear BLEU (see Tromble et al 2008) of each candidate.
    @param samples is the candidates (also the evidence set)
    @param bleusuff (see BLEUSufficientStatistics)
    @param T, p, r are language-pair-specific parameters (see Tromble et al 2008) informed by devsets
    @param n max ngram order
    @return a list of pairs of the kind (sample, expected linear bleu) sorted from best to worst
    """

    # From (Tromble et al, 2008)
    # maximise the expected gain
    # d* = argmax_{d' \in H} { \theta_0 |d'| + \sum_{w \in N} \theta_w count_w(d') p(w|E)}
    # d' is a hypothesis (a candidate)
    # H is the hypothesis space
    # E is the evidence set
    # N is the set of n-grams (1 <= n <= 4) in D_e
    # p(w|E) = Z(E_w)/Z(E) where E_w is the set of hypotheses in E which contain w
    # count_w(d') is the number of occurrences of w in d'
    # \theta_0 and \theta_w(cn) are the Taylor coefficients
    # taylor coefficients

    theta_0 = -1.0/T
    cn = [0] * (bleusuff.maxorder + 1)
    for k in range(1, bleusuff.maxorder + 1):
        cn[k] = T * p * math.pow(r, k - 1)
    theta_w = lambda ngram : 1.0/(4 * cn[len(ngram)])

    G = [0] * len(samples)
    ngramstats = bleusuff.ngramstats
    for h, d in enumerate(samples):
        gain = theta_0 * len(d.leaves) + sum(theta_w(w) * stats.counts[h] * stats.posterior for w, stats in ngramstats.iteritems())
        G[h] = gain
    return G

def consensus_bleu(samples, bleusuff, bleu = BLEU.ibm_bleu):
    ngramstats = bleusuff.ngramstats
    # TODO: wrap in a class that prepares the expectations
    N = [w for w in ngramstats.iterkeys()]
    C = csc_matrix([stats.counts for stats in ngramstats.itervalues()])
    L = np.matrix([bleusuff.length(h) for h in xrange(len(samples))])
    P = np.matrix([sample.normcount for sample in samples]).transpose()
    Ec = C * P
    El = L * P
    N2E = {w:Ec[i] for i, w in enumerate(N)}


    S = [0] * len(samples)
    for h, sample in enumerate(samples):
    
        # clip to the expected counts
        cc = [0] * (bleusuff.maxorder + 1)
        for w, c in bleusuff.ngrams(h).iteritems():
            cc[len(w)] += min(c, N2E.get(w, 0))

        # compute BLEU
        S[h] = bleu(r = El[0,0], 
                c = bleusuff.length(h), 
                cc = cc,
                tc = bleusuff.tc(h),
                n = bleusuff.maxorder)

    return S


def cobleu(reference, samples, bleusuff, bleu = BLEU.ibm_bleu):
    """
    See Pauls et at 2009
    it is basically BLEU where the candidate is represented by a vector of expected counts.

    If BLEU(c, r) represents the modified ngram precision between a candidate c  and a reference r, then:
        * in CoBLEU training, r is the reference and c is represented by expected counts
            we then maximise theta (the parameters of the model)

        * in Consensus decoding, c is a hypothesis and r is represented by expected counts
    """
    pass
