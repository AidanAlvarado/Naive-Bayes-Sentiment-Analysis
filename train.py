import pandas as pd
import re as re
import math as math
import random


# function to create bags of words, which are dictionaries with words as keys and frequencies as values.
def createBoW(
    tokens,
):  # creates a dictionary with the bag of words (frequency of all the words in the txt)
    BoW = {}
    BoW["TOTAL"] = 0
    BoW["VOCABTOTAL"] = 0
    for i in tokens:
        BoW["TOTAL"] += 1
        if i in BoW:
            BoW[i] += 1
        else:
            BoW["VOCABTOTAL"] += 1
            BoW[i] = 1
    return BoW


# tokenize tokenizes input text for training.
def tokenize(
    txt,
):  # turns the input text into a list of all the words. takes out capital letters and punc.

    negation_words = [
        "no",
        "not",
        "never",
        "none",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "isn't",
        "arent",
        "wasnt",
        "werent",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "wouldnt",
        "cant",
        "couldnt",
        "shouldnt",
        "mustnt",
        "hasnt",
        "havent",
        "hadnt",
        "aint",
        "isnt",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "can't",
        "couldn't",
        "shouldn't",
        "mustn't",
        "hasn't",
        "haven't",
        "hadn't",
        "ain't",
    ]
    retval = []
    if not isinstance(txt, str):
        return []
    negating = False
    retval = []
    # tokens = re.findall(r"\w+|[.!?]", txt.lower())
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|[.!?]", txt.lower())

    for token in tokens:
        if token in negation_words:
            negating = True
        elif token in [".", "!", "?"]:
            negating = False
            continue
        elif negating:
            retval.append("NOT_" + token)
        else:
            retval.append(token)
    return retval

    # main runs all the training

    """Parsing the training data and extracting the columns that are needed"""


df = pd.read_csv("mr.csv")
column_names = df.columns
reviews = df["reviewText"]
ratings = df["overall"]

""" Sorting all of the reviews into negative or positive."""
posReviews = []
negReviews = []

for i in range(len(reviews)):
    if ratings[i] > 3:
        posReviews.append(reviews[i])
    if ratings[i] < 3:
        negReviews.append(reviews[i])

n = min(len(posReviews), len(negReviews))

posReviews = random.sample(posReviews, n)
negReviews = random.sample(negReviews, n)

prob_posReview = math.log((len(posReviews) / (len(posReviews) + len(negReviews))))
prob_negReview = math.log((len(negReviews) / (len(posReviews) + len(negReviews))))

""" Tokenizing the negative and positive reviews """
tk_posReviews = []
tk_negReviews = []
for i in posReviews:
    tk_posReviews.extend(tokenize(i))
for i in negReviews:
    tk_negReviews.extend(tokenize(i))

    """ Making a bag of words for the negative and positive reviews, in order to get word frequencies"""
posReviewsBoW = createBoW(tk_posReviews)
negReviewsBoW = createBoW(tk_negReviews)

total_negpos = posReviewsBoW["TOTAL"] + negReviewsBoW["TOTAL"]
