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
    sentiment_list_day = filtered_df.iloc[[-1]].values[0]
    week_df = filtered_df.iloc[-7:]
    negative_week = week_df['negative'].sum()
    positive_week = week_df['positive'].sum()
    neutral_week = week_df['neutral'].sum()
    sentiment_list_week = [negative_week, positive_week, neutral_week]
    # plot line chart
    line_fig = px.line(df, x="date", y="negative",title="Longitudinal Negative Sentiment")
    # plot pie chart
    pie_fig = go.Figure(data=[go.Pie(labels=['negative', 'positive', 'neutral'], values=sentiment_list_day)])
    pie_fig.update_layout(title_text="Today's Sentiment")
    # plot week pie chart
    pie_week_fig = go.Figure(data=[go.Pie(labels=['negative', 'positive', 'neutral'], values=sentiment_list_week)])
    pie_week_fig.update_layout(title_text="This Week's Sentiment", )
    # for render
    line_json = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
    pie_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)
    pie_week_json = json.dumps(pie_week_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("prediction.html", line_fig=line_json, pie_fig=pie_json, week_pie_fig=pie_week_json)
