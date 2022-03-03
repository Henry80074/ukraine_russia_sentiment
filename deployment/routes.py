import datetime
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route('/', methods=['GET'])
def dashboard():
    today = datetime.datetime.today()
    look_back = 45
    fortune_teller = pd.DataFrame(
        data={'predictions': [col[0] for col in results], 'actual': [col[0] for col in actual]},
        columns=["predictions", "actual"])
    # predictions from certain day in time to X days into the future
    fig = px.line(fortune_teller, x=[i for i in range(len(fortune_teller))], y=fortune_teller.columns, width=800, height=400,
                  title="projected price from %s days ago to %s days ago" % (days, days - future_predict))
    # bitcoin price chart
    fig2 = px.line(df, x="date", y="prices", width=800, height=400, title="bitcoin price")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("prediction.html", graphJSON=graphJSON, graphJSON2=graphJSON2)