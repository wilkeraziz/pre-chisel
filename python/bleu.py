"""
@author waziz
"""

import math

class BLEU(object):
    
    @classmethod
    def eval(cls, r, c, cc, tc, n, smoothing):
        """
        @param r reference length
        @param c candidate length
        @param cc (clipped counts) is a vector of clipped counts such that cc[k] is the count for k-grams
        @param tc (total counts) is a vector of the total ngram counts (for the candidate), tc[k] is the count for k-grams
        @param n max ngram order
        @param smoothing computes smoothed precisions from cc and tc (both adjusted to exactly n positions)
        @return bleu
        """
        bp = 1.0 if c > r else math.exp(1-float(r)/c)
        return bp * math.exp(1.0/n * sum(math.log(pn) for pn in smoothing(cc[1:n+1], tc[1:n+1])))
    
    @classmethod
    def no_smoothing(cls, cc, tn):
        """
        Unsmoothed precisions.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        if any(c == 0 for c in cc):
            raise ValueError, 'At least one of the clipped counts is zero: %s' % str(cc)
        for c, t in zip(cc, tn):
            yield float(c) / t
    
    @classmethod
    def p1_smoothing(cls, cc, tn):
        """
        Sum 1 to numerator and denorminator for all orders.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        for c, t in zip(cc, tn):
            yield float(c + 1)/(t + 1)
    
    @classmethod
    def ibm_smoothing(cls, cc, tn):
        """
        IBM smoothing. Assigns a precision of 1/2^k for null counts, where k = 1 for the first n
        whose counts are null.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        k = 0
        for c, t in zip(cc, tn):
            if c > 0:
                yield float(c)/t
            else:
                k += 1
                yield 1.0/math.pow(2, k)

    @classmethod
    def unsmoothed_bleu(cls, r, c, cc, tc, n = 4):
        return BLEU.eval(r, c, cc, tc, n, BLEU.no_smoothing)
    
    @classmethod
    def bleu_p1(cls, r, c, cc, tc, n = 4):
        return BLEU.eval(r, c, cc, tc, n, BLEU.p1_smoothing)
    
    @classmethod
    def ibm_bleu(cls, r, c, cc, tc, n = 4):
        return BLEU.eval(r, c, cc, tc, n, BLEU.ibm_smoothing)
    
    @classmethod
    def get(cls, name):
        if name == 'unsmoothed_bleu':
            return BLEU.unsmoothed_bleu
        if name == 'bleu_p1':
            return BLEU.bleu_p1
        if name == 'ibm_bleu':
            return BLEU.ibm_bleu
        raise Exception, 'Unknown implementation BLEU.%s' % name

