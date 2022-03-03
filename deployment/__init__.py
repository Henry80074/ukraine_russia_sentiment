from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
app.config.from_object("deployment.config.Config")
db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(ROOT_DIR, 'sentiment.db')



class SentimentTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String())
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)
    polarity = db.Column(db.Integer)
    neutral_list = db.Column(db.String())
    positive_list = db.Column(db.String())
    negative_list = db.Column(db.String())


from deployment import routes
