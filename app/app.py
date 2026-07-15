import streamlit as st
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("final_model_rf.pkl")
SCALER_PATH = Path("final_scaler.pkl")
FEATURES_FULL = ["Elo_Diff", "P1_Elo", "P2_Elo", "WinRate_Diff", "P1_Streak", "P2_Streak"]

st.set_page_config(page_title="Ping Pong Matchmaking", layout="centered")


@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()

st.title("Ping Pong Matchmaking Predictor")
st.write(
    "Estimate the probability that a matchup will be an exciting match "
    "(Neck and Neck or Comeback) before it's played, using pre-match Elo, "
    "win rate, and streak data."
)

if model is None or scaler is None:
    st.error(
        "Model files not found. This app expects final_model_rf.pkl "
        "and final_scaler.pkl in the same folder as app.py.\n\n"
        "To generate them, run 03_model_tuning.ipynb from top to bottom. "
        "The final cell saves both files via joblib.dump(). Then copy them "
        "into this app's folder and refresh."
    )
    st.stop()

st.divider()

st.subheader("Player 1")
col1, col2 = st.columns(2)
with col1:
    p1_elo = st.number_input("P1 Elo rating", min_value=0, max_value=3000, value=1000, step=1)
    p1_streak = st.number_input("P1 current win streak", min_value=0, max_value=50, value=0, step=1)
with col2:
    p1_winrate = st.slider("P1 win rate", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

st.subheader("Player 2")
col3, col4 = st.columns(2)
with col3:
    p2_elo = st.number_input("P2 Elo rating", min_value=0, max_value=3000, value=1000, step=1)
    p2_streak = st.number_input("P2 current win streak", min_value=0, max_value=50, value=0, step=1)
with col4:
    p2_winrate = st.slider("P2 win rate", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

st.divider()

if st.button("Predict match excitement", type="primary"):
    elo_diff = p1_elo - p2_elo
    winrate_diff = p1_winrate - p2_winrate

    X = np.array([[elo_diff, p1_elo, p2_elo, winrate_diff, p1_streak, p2_streak]])
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]

    st.metric("Predicted probability of an exciting match", f"{proba:.1%}")

    if pred == 1:
        st.success("This matchup is predicted to be exciting -- recommend it.")
    else:
        st.info("This matchup is predicted to be not exciting.")

    with st.expander("How should I read this?"):
        st.write(
            "- The model prioritizes recall over precision: it's tuned to "
            "catch as many genuinely exciting matchups as possible, even if "
            "that means some matches predicted as 'exciting' turn out not to be.\n"
            "- Only about 5.5% of matches in the training data were exciting, "
            "so precision on this class is inherently low (~10%). Treat this "
            "as a recommendation signal, not a guarantee.\n"
            "- This model was trained partly on synthetic data (50,000 of "
            "57,851 rows) designed to simulate community growth. Real-world "
            "validation is still needed before treating these predictions as "
            "fully reliable."
        )

st.divider()
st.caption(
    "Model: Tuned Random Forest (GridSearchCV). "
    "Features: Elo gap, individual Elo, win rate gap, win streaks. "
    "See project README for full methodology and limitations."
)
