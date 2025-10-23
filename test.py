import train as tr
import math as math


def wordProb(word, classBoW, totalWords):
    # Laplace smoothing
    return (classBoW.get(word, 0) + 1) / (classBoW["TOTAL"] + classBoW["VOCABTOTAL"])


def classifyReview(review):
    tk_review = tr.tokenize(review)
    # prob_pos = math.log(tr.posReviewsBoW['TOTAL']/(tr.posReviewsBoW['TOTAL']+tr.negReviewsBoW['TOTAL']))
    # prob_neg = math.log(tr.negReviewsBoW['TOTAL']/(tr.posReviewsBoW['TOTAL']+tr.negReviewsBoW['TOTAL']))
    prob_neg = tr.prob_negReview
    prob_pos = tr.prob_posReview
    for word in tk_review:
        prob_pos += math.log(wordProb(word, tr.posReviewsBoW, tr.total_negpos))
        prob_neg += math.log(wordProb(word, tr.negReviewsBoW, tr.total_negpos))

    # using log sum exp trick, to convert log probabilities back into meaningfull percentages,
    # to be displayed as certainty.
    max_log = max(prob_pos, prob_neg)
    prob_pos_exp = math.exp(prob_pos - max_log)
    prob_neg_exp = math.exp(prob_neg - max_log)
    total = prob_pos_exp + prob_neg_exp

    pos_percentage = (prob_pos_exp / total) * 100
    neg_percentage = (prob_neg_exp / total) * 100

    if prob_pos > prob_neg:
        return f"Positive ({pos_percentage:.1f}% certain)"
    else:
        return f"Negative ({neg_percentage:.1f}% certain)"
