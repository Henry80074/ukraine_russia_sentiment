from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import os
# import nltk
# import pycountry
# import re
# import string
from passwords import access_token, access_token_secret, bearer, api_key, api_secret
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# Authentication
project_api_key = api_key
project_api_secret = api_secret
project_access_token = access_token
project_access_secret = access_token_secret
project_bearer = bearer

auth = tweepy.OAuthHandler(project_api_key, project_api_secret)
auth.set_access_token(project_access_token, project_access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# Sentiment Analysis
def percentage(part, whole):
    return 100 * float(part) / float(whole)

# Moscow
geocode = "55.7558 , 37.6173, 300km"
keyword = "putin"
no_of_tweet = 100
tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(no_of_tweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
for tweet in tweets:

    # print(tweet.text)
    tweet_list.append(tweet.text)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(tweet.text)
        negative += 1
    elif pos > neg:
        positive_list.append(tweet.text)
        positive += 1

    elif pos == neg:
        neutral_list.append(tweet.text)
        neutral += 1
        positive = percentage(positive, no_of_tweet)
        negative = percentage(negative, no_of_tweet)
        neutral = percentage(neutral, no_of_tweet)
        polarity = percentage(polarity, no_of_tweet)
        # positive = format(positive, .1f)
        # negative = format(negative, .1f)
        # neutral = format(neutral, .1f)

#Creating PieCart
labels = ["Positive ["+str(positive)+"%]" , "Neutral ["+str(neutral)+"%]","Negative ["+str(negative)+"%]"]
sizes = [positive, neutral, negative]
colors = ["yellowgreen", "blue","red"]
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use("default")
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= "+keyword+"")
plt.axis("equal")
plt.show()