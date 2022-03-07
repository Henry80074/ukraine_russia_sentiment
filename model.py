import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle
from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

current_dir = os.getcwd()
model_dir = current_dir + "/model"


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
            model.add(Embedding(len(self.tokenizer.word_index) + 1, embedding_vector_length, input_length=60))
            model.add(SpatialDropout1D(0.25))
            model.add(LSTM(15, dropout=0.5, recurrent_dropout=0.5))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='sigmoid'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = model
            print(model.summary())
        else:
            print("Preprocess data first!")

    def preprocess_training_data(self):

        self.sentiment_label = self.df.sentiment.factorize()
        tweets = self.df['russian'].values
        # convert tweets to strings
        tweets = [str(x) for x in tweets]
        # converts words to vector of numbers, 5000 is max number of different words
        self.tokenizer = Tokenizer(num_words=5000)
        # creates an association between the words and the assigned numbers
        self.tokenizer.fit_on_texts(tweets)
        with open(model_dir + "/tokenizer.pickle", 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(model_dir + "/sentiment_label_text.pickle", 'wb') as handle:
            pickle.dump(self.sentiment_label[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # replace the words with their assigned numbers
        encoded_docs = self.tokenizer.texts_to_sequences(tweets)
        # makes all sentences same size
        self.padded_sequence = pad_sequences(encoded_docs, maxlen=60)
        return self.padded_sequence

    def preprocess_prediction_data(self):
        # get tweets
        tweets = self.df['russian'].values
        # convert tweets to strings
        tweets = [str(x) for x in tweets]
        # get tokenizer from training
        with open(model_dir + "/tokenizer.pickle", 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        # get text index from training
        with open(model_dir + "/sentiment_label_text.pickle", 'rb') as sent:
             self.sentiment_label = pickle.load(sent)
        # replace the words with their assigned numbers
        encoded_docs = self.tokenizer.texts_to_sequences(tweets)
        # makes all sentences same size
        self.padded_sequence = pad_sequences(encoded_docs, maxlen=60)

    def fit_data(self):
        # Separate the test data
        x, x_test, y, y_test = train_test_split(self.padded_sequence, self.sentiment_label[0], test_size=0.15, shuffle=True)
        # Split the remaining data to train and validation
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, shuffle=True)
        # Training the Keras model
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_dir,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        if self.model:
            self.history = self.model.fit(x=x_train, y=y_train, batch_size=2, epochs=20, validation_data=(x_val, y_val), callbacks=[model_checkpoint_callback])
            self.model.save(model_dir)
        else:
            print("No model found")

    def load_model(self):
        self.model = keras.models.load_model(model_dir)
        return self.model

    def plot_metrics(self):
        try:
            plt.plot(self.history.history['accuracy'], label='acc')
            plt.plot(self.history.history['val_accuracy'], label='val_acc')
            plt.legend()
            plt.show()
            plt.savefig(model_dir + "/Accuracy_plot.jpg")
            plt.plot(self.history.history['loss'], label='loss')
            plt.plot(self.history.history['val_loss'], label='val_loss')
            plt.legend()
            plt.show()
            plt.savefig(model_dir + "/Loss_plt.jpg")
        except TypeError:
            print("history not found")

    def predict_sentiment(self):
        for tweet in self.df['russian'].values:
            text = tweet
            tw = self.tokenizer.texts_to_sequences([text])
            tw = pad_sequences(tw, maxlen=60)
            prediction = self.model.predict(tw)
            max_index = np.argmax(prediction[0])
            prediction = self.sentiment_label[max_index]
            print(tweet)
            print(prediction)

