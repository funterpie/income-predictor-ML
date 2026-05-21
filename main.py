from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="Income Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = joblib.load("model.pkl")

class UserInput(BaseModel):
    age:             int
    workclass:       str
    educational_num: int
    marital_status:  str
    occupation:      str
    relationship:    str
    race:            str
    gender:          str
    capital_gain:    float
    capital_loss:    float
    hours_per_week:  float
    native_country:  str

@app.get("/")
def root():
    return {"status": "Income Predictor API by funter pie is live ✓"}

@app.post("/predict")
def predict(data: UserInput):
    input_df = pd.DataFrame([{
        'age'             : data.age,
        'workclass'       : data.workclass,
        'educational-num' : data.educational_num,
        'marital-status'  : data.marital_status,
        'occupation'      : data.occupation,
        'relationship'    : data.relationship,
        'race'            : data.race,
        'gender'          : data.gender,
        'capital-gain'    : data.capital_gain,
        'capital-loss'    : data.capital_loss,
        'hours-per-week'  : data.hours_per_week,
        'native-country'  : data.native_country
    }])

    pred   = model.predict(input_df)[0]
    proba  = model.predict_proba(input_df)[0]

    return {
        "prediction"  : int(pred == '>50K') if isinstance(pred, str) else int(pred),
        "income"      : str(pred),
        "confidence"  : f"{max(proba)*100:.1f}%",
        "prob_below"  : f"{proba[0]*100:.1f}%",
        "prob_above"  : f"{proba[1]*100:.1f}%"
    }