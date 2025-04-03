import pandas as pd
from sklearn.model_selection import train_test_split
from shaphypetune.scorecard.category_encoder import ProbSmoothingStr2RspTransformer, OrderStr2RspTransformer, TfidfStr2RspTransfomer


df = pd.DataFrame([
    { 'length_in': 55, 'large_gauge': 1, 'color': 'orange', 'completed': 1 },
    { 'length_in': 55, 'large_gauge': 0, 'color': 'orange', 'completed': 1 },
    { 'length_in': 55, 'large_gauge': 0, 'color': 'brown', 'completed': 1 },
    { 'length_in': 60, 'large_gauge': 0, 'color': 'brown', 'completed': 1 },
    { 'length_in': 60, 'large_gauge': 0, 'color': 'grey', 'completed': 0 },
    { 'length_in': 70, 'large_gauge': 0, 'color': 'grey', 'completed': 1 },
    { 'length_in': 70, 'large_gauge': 0, 'color': 'orange', 'completed': 0 },
    { 'length_in': 82, 'large_gauge': 1, 'color': 'grey', 'completed': 1 },
    { 'length_in': 82, 'large_gauge': 0, 'color': 'brown', 'completed': 0 },
    { 'length_in': 82, 'large_gauge': 0, 'color': 'orange', 'completed': 0 },
    { 'length_in': 82, 'large_gauge': 1, 'color': 'brown', 'completed': 0 },
])

train, valid, _, _ = train_test_split(df, df["completed"], random_state=0)
train["set"] = "train"
valid["set"] = "valid"
df = pd.concat([train, valid], axis=0)



def test_oderstr2rsptransformer():
    cols = ['length_in', 'large_gauge', 'color']
    encoder = OrderStr2RspTransformer(col_x=cols, data_parts="set", data_parts_use=["train"])
    encoder.fit(df, df["completed"])
    result = encoder.transform(df)
    

def test_probsmoothingstr2ssptransformer():
    cols = ['length_in', 'large_gauge', 'color']
    encoder = ProbSmoothingStr2RspTransformer(col_x=cols, data_parts="set", data_parts_use=["train"])
    encoder.fit(df, df["completed"])
    result = encoder.transform(df)
    
    
def test_tfidfstr2rsptransfomer():
    cols = ['length_in', 'large_gauge', 'color']
    encoder = TfidfStr2RspTransfomer(col_x=cols, data_parts="set", data_parts_use=["train"])
    encoder.fit(df, df["completed"])
    result = encoder.transform(df)

