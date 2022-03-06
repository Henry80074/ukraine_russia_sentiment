import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from keras.layers import Embedding
import matplotlib.pyplot as plt
dataframe = pd.read_csv("./manual_classification_data.csv")
current_dir = os.getcwd()
print(dataframe["sentiment"].value_counts())


class SentimentModel:

    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.model = None
        self.tokenizer = None
        self.padded_sequence = None
        self.sentiment_label = None
        self.history = None

    def build_model(self):
        if self.tokenizer:
            embedding_vector_length = 32
            model = Sequential()
            model.add(Embedding(len(self.tokenizer.word_index) + 1, embedding_vector_length, input_length=350))
            model.add(SpatialDropout1D(0.25))
            model.add(LSTM(15, dropout=0.5, recurrent_dropout=0.5))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='sigmoid'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = model
            print(model.summary())
        else:
            print("Preprocess data first!")

    def preprocess_data(self):
        self.sentiment_label = self.df.sentiment.factorize()
        tweets = self.df['russian'].values
        # convert tweets to strings
        tweets = [str(x) for x in tweets]
        # converts words to vector of numbers, 5000 is max number of different words
        self.tokenizer = Tokenizer(num_words=5000)
        # creates an association between the words and the assigned numbers
        self.tokenizer.fit_on_texts(tweets)
        # replace the words with their assigned numbers
        encoded_docs = self.tokenizer.texts_to_sequences(tweets)
        # makes all sentences same size
        self.padded_sequence = pad_sequences(encoded_docs, maxlen=350)
        print("data processed")
        print(self.sentiment_label[0])
        print(self.tokenizer)

    def fit_data(self):
        if self.model:
            self.history = self.model.fit(self.padded_sequence, self.sentiment_label[0], validation_split=0.2, epochs=5, batch_size=1)
            self.model.save(current_dir)
        else:
            print("No model found")

    def load_model(self):
        self.model = keras.models.load_model(current_dir)
        return self.model

    def plot_metrics(self):
        try:
            plt.plot(self.history.history['accuracy'], label='acc')
            plt.plot(self.history.history['val_accuracy'], label='val_acc')
            plt.legend()
            plt.show()
            plt.savefig("Accuracy plot.jpg")
            plt.plot(self.history.history['loss'], label='loss')
            plt.plot(self.history.history['val_loss'], label='val_loss')
            plt.legend()
            plt.show()
            plt.savefig("Loss plt.jpg")
        except(TypeError):
            print("history not found")

SentimentModel = SentimentModel("russian_sentiment", dataframe)
SentimentModel.preprocess_data()
SentimentModel.build_model()
SentimentModel.fit_data()
SentimentModel.plot_metrics()