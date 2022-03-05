import datetime
import json
import sqlite3
import pandas as pd
from textblob import TextBlob
import tweepy
import matplotlib.pyplot as plt
from passwords import access_token, access_token_secret, bearer, api_key, api_secret
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google_trans_new import google_translator
nltk.download('vader_lexicon')
# Authentication
project_api_key = api_key
project_api_secret = api_secret
project_access_token = access_token
project_access_secret = access_token_secret
project_bearer = bearer

auth = tweepy.OAuthHandler(project_api_key, project_api_secret)
auth.set_access_token(project_access_token, project_access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


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
        translator = google_translator()
        print(tweet.full_text)
        text = str(translator.translate(tweet.full_text, lang_tgt='en'))
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


def plot_pie(keyword, **kwargs):
    labels = []
    sizes = []
    for key, value in kwargs:
        labels.append(str(key) + " [" + str(value) + "%]")
        sizes.append(value)
    # Creating PieCart
    colors = ["yellowgreen", "blue", "red"]
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use("default")
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for keyword= " + keyword + "")
    plt.axis("equal")
    plt.show()

compile_data()