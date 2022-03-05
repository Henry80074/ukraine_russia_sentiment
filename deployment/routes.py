import datetime
import sqlite3

import pandas
from flask import render_template, request
import numpy as np
import pandas as pd
import pickle
import json
import plotly
import plotly.express as px
import os
import plotly.graph_objects as go
from deployment import app
pd.options.plotting.backend = "plotly"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route('/', methods=['GET'])
def dashboard():
    conn = sqlite3.connect('sentiment.db')
    # get data, sort by date
    df = pandas.read_sql_query("SELECT * FROM sentiment_table", conn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")
    # filter data for pie chart
    filtered_df = df.filter(items=['negative', 'positive', 'neutral'])
    sentiment_list = filtered_df.iloc[[-1]].values[0]
    # plot line chart
    line_fig = px.line(df, x="date", y="negative", width=800, height=450, title="Longitudinal Negative Sentiment")
    # plot pie chart
    pie_fig = go.Figure(data=[go.Pie(labels=['negative', 'positive', 'neutral'], values=sentiment_list)])
    pie_fig.update_layout(title_text="Today's Sentiment",)
    # for render
    graph_json = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
    graph_json2 = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("prediction.html", graphJSON=graph_json, graphJSON2=graph_json2)