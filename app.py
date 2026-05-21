import streamlit as st
import requests

st.set_page_config(page_title="Income Predictor By Funter Pie AKA Muhammad Taha Sattar", page_icon="💰", layout="centered")

API_URL = "http://localhost:8000/predict"  # local test
# API_URL = "https://your-app.onrender.com/predict"  # deploy ke baad change karna

EDUCATION_MAP = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3,
    '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7,
    '12th': 8, 'HS-grad': 9, 'Some-college': 10,
    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13,
    'Masters': 14, 'Prof-school': 15, 'Doctorate': 16
}

st.title("💰 Income Predictor By Funter Pie AKA Muhammad Taha Sattar")
st.write("Predict whether a person earns **>50K** or **≤50K**")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age            = st.slider("Age", 17, 90, 35)
    education      = st.selectbox("Education", list(EDUCATION_MAP.keys()))
    capital_gain   = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss   = st.number_input("Capital Loss", 0, 4356, 0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)

with col2:
    gender         = st.selectbox("Gender", ["Male", "Female"])
    workclass      = st.selectbox("Work Class", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc',
        'Federal-gov', 'Local-gov', 'State-gov',
        'Without-pay', 'Never-worked'
    ])
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ])
    occupation     = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    relationship   = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband',
        'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    race           = st.selectbox("Race", [
        'White', 'Asian-Pac-Islander',
        'Amer-Indian-Eskimo', 'Other', 'Black'
    ])
    native_country = st.selectbox("Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
        'India', 'England', 'Cuba', 'Jamaica', 'South', 'China',
        'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan',
        'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal',
        'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland',
        'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Thailand',
        'Yugoslavia', 'El-Salvador', 'Hungary', 'Honduras', 'Scotland',
        'Outlying-US(Guam-USVI-etc)', 'Holand-Netherlands'
    ])

st.divider()
btn = st.button("🔍 Predict Income", use_container_width=True)

# ── Result yahan aata hai ──────────────────────────────────────
if btn:
    payload = {
        "age"            : age,
        "workclass"      : workclass,
        "educational_num": EDUCATION_MAP[education],
        "marital_status" : marital_status,
        "occupation"     : occupation,
        "relationship"   : relationship,
        "race"           : race,
        "gender"         : gender,
        "capital_gain"   : capital_gain,
        "capital_loss"   : capital_loss,
        "hours_per_week" : hours_per_week,
        "native_country" : native_country
    }

    with st.spinner("Predicting..."):
        try:
            res    = requests.post(API_URL, json=payload, timeout=10)
            result = res.json()

            # ── RESULT DISPLAY ──────────────────────────────
            st.divider()
            if result["prediction"] == 1:
                st.success(f"✅ Predicted Income: **{result['income']}**")
            else:
                st.info(f"📊 Predicted Income: **{result['income']}**")

            # Confidence
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Confidence",  result["confidence"])
            col_b.metric("Prob >50K",   result["prob_above"])
            col_c.metric("Prob ≤50K",   result["prob_below"])

            # Progress bars
            st.progress(
                float(result['prob_above'].replace('%','')) / 100,
                text=f">50K → {result['prob_above']}"
            )
            st.progress(
                float(result['prob_below'].replace('%','')) / 100,
                text=f"≤50K → {result['prob_below']}"
            )

        except Exception as e:
            st.error(f"❌ API Error: {e}")

st.divider()
st.caption("Made with ❤️ by [Funter Pie](https://github.com/funter-pie) (Muhammad Taha Sattar) | [GitHub](https://github.com/funter-pie)")  
st.caption("Model trained on UCI Adult Dataset | API built with FastAPI | Frontend built with Streamlit")   
st.caption("visit tahatradz.online for more fun projects and blogs!")