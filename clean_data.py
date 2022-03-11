import pandas as pd
from main import clean_text
df = pd.read_csv("test_data.csv")
df.columns = ["ind", "russian", "english", "sentiment"]
for i in range(len(df)):
    russian_text = df.iloc[i]['russian']
    df.at[i, 'russian']= clean_text(russian_text, True)

df.to_csv("cleaned.csv")