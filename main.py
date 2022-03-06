import datetime
import json
import sqlite3
import pandas as pd
from textblob import TextBlob
import tweepy
from passwords import access_token, access_token_secret, bearer, api_key, api_secret, deep_ai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import deepl
nltk.download('vader_lexicon')
# Authentication
project_api_key = api_key
project_api_secret = api_secret
project_access_token = access_token
project_access_secret = access_token_secret
project_bearer = bearer
deepl_api_key = deep_ai

auth = tweepy.OAuthHandler(project_api_key, project_api_secret)
auth.set_access_token(project_access_token, project_access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
translator = deepl.Translator(deepl_api_key)

# Sentiment Analysis
def percentage(part, whole):
    return 100 * float(part) / float(whole)


def get_tweets(geo, keyword, number, start_date, end_date):
    date_string = f"since:{start_date} until:{end_date}"
    tweets = tweepy.Cursor(api.search_tweets, q=("#" + keyword + " " + date_string), lang="ru", geocode=geo, tweet_mode='extended').items(number)
    return tweets


def get_sentiment(tweets, date):
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []
    for tweet in tweets:
        print(tweet.full_text)
        text = str(translator.translate_text(tweet.full_text, target_lang='en'))
        if 'RT @' not in text:
            tweet_list.append(text)
            analysis = TextBlob(text)
            score = SentimentIntensityAnalyzer().polarity_scores(text)
            neg = score["neg"]
            neu = score["neu"]
            pos = score["pos"]
            comp = score["compound"]
            polarity += analysis.sentiment.polarity
            # increment counters based on sentiment
            if neg > pos:
                negative_list.append(text)
                negative += 1
            elif pos > neg:
                positive_list.append(text)
                positive += 1
            elif pos == neg:
                neutral_list.append(text)
                neutral += 1
    no_of_tweet = len(tweet_list)
    positive = format(percentage(positive, no_of_tweet), ".1f")
    negative = format(percentage(negative, no_of_tweet), ".1f")
    neutral = format(percentage(neutral, no_of_tweet), ".1f")
    polarity = format(percentage(polarity, no_of_tweet), ".1f")

    data = {"date": date, "positive": positive, "negative": negative,
            "neutral": neutral, "polarity": polarity, "neutral_list": json.dumps(neutral_list),
            "positive_list": json.dumps(positive_list), "negative_list": json.dumps(negative_list)}
    return data


def compile_data():
    geo = "55.7558,37.6173,300km"
    keyword = "putin OR Путин"
    no_of_tweets = 100
    today = datetime.datetime.today()# - datetime.timedelta(days=4)
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    tweets = get_tweets(geo=geo, keyword=keyword, number=no_of_tweets, start_date=start_date, end_date=end_date)
    data = get_sentiment(tweets, end_date)
    df = pd.DataFrame(data, index=[0])
    conn = sqlite3.connect('sentiment.db')
    df.to_sql('sentiment_table', conn, if_exists='append', index=False)


class ModelData:

    def __init__(self, geo, keyword, number, start_date, end_date):
        self.geo = geo
        self.keyword = keyword
        self.number = number
        self.start_date = start_date
        self.end_date = end_date
        self.tweets = None

    def get_data(self):
        self.tweets = get_tweets(self.geo, self.keyword, self.number, self.start_date, self.end_date)

    def create_csv(self):
        english_tweet_list = []
        russian_tweet_list = []
        for tweet in self.tweets:
            russian_tweet = tweet.full_text
            english_tweet = str(translator.translate_text(russian_tweet, target_lang='en'))
            if 'RT @' not in english_tweet:
                english_tweet_list.append(english_tweet)
                russian_tweet_list.append(tweet.full_text)

        data = {"tweet_id": [x for x in range(len(english_tweet_list))], "russian": [x for x in russian_tweet_list], "english": [x for x in english_tweet_list], "sentiment": ["unassigned" for i in range(len(english_tweet_list))]}
        df = pd.DataFrame(data)
        print(df)
        df.to_csv('manual_classification_data.csv', mode='a', index=False)

class SentimentModel:

    def __init__(self, name):
        self.name = name

#compile_data()
# today = datetime.datetime.today()
# model_data = ModelData(geo="55.7558,37.6173,300km", keyword="putin OR Путин", number=100,
#                  start_date=(today - datetime.timedelta(days=6)).strftime('%Y-%m-%d'), end_date=today.strftime('%Y-%m-%d'))
#
#
# model_data.get_data()
# model_data.create_csv()