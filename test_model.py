import datetime
from model import SentimentModel
import pandas as pd
import os
from main import ModelData

current_dir = os.getcwd()
model_dir = current_dir + "/model"

# gathers data and makes predictions
today = datetime.datetime.today()
database = ModelData(geo="55.7558,37.6173,300km", keyword="putin OR Путин", number=10,
                 start_date=(today - datetime.timedelta(days=6)).strftime('%Y-%m-%d'), end_date=(today - datetime.timedelta(days=5)).strftime('%Y-%m-%d'))

database.get_data()
russian_tweet_list = []
for tweet in database.tweets:
    russian_tweet = tweet.full_text
    if 'RT @' not in russian_tweet and russian_tweet not in russian_tweet_list:
        russian_tweet_list.append(tweet.full_text)

data = {"russian": [x for x in russian_tweet_list]}
df = pd.DataFrame(data)

sentiment_model = SentimentModel("russian_sentiment", df)
sentiment_model.preprocess_prediction_data()
sentiment_model.load_model()
sentiment_model.predict_sentiment()

# trains model
# #dataframe = pd.read_csv("./manual_classification_data.csv")
# sentiment_model = SentimentModel("russian_sentiment", dataframe)
# sentiment_model.preprocess_training_data()
# sentiment_model.build_model()
# sentiment_model.fit_data()
# sentiment_model.plot_metrics()