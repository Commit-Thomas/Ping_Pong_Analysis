# Ping Pong Community Analytics

## Project Purpose
This project investigates whether match competitiveness drives player 
engagement in a recreational ping pong community, and whether we can 
predict match competitiveness in advance to actively improve it.

The core finding that motivates the model: players who experience 
exciting matches (Neck & Neck or Comeback) return the following month 
at an 88.6% rate compared to 72.5% for players who did not — a 16 
percentage point difference. The model attempts to predict which 
matchups will produce exciting matches before they are played, enabling 
Elo-based matchmaking recommendations.

## Current Status (as of July 15)
**Sprint 1 — Lock the Model: Complete.**
**Sprint 2 — Build the Artifact: Complete.**
- Data cleaning and feature engineering refactored out of notebooks into 
  reusable scripts (`src/data_pipeline.py`, `src/features.py`)
- `requirements.txt` added, listing exact package versions used
- Streamlit matchmaking app built and running locally (`app/app.py`), 
  loading the saved final model and scaler to predict match excitement 
  probability from player Elo, win rate, and streak

**Next up — Sprint 3 (due July 24):** AI-use log, slide deck, Yoodli, 
and final presentation.

## Stakeholder
Recreational ping pong players and club organisers who want a structured 
way to track player progression, measure match quality, and grow an 
engaged community — without the infrastructure of a professional 
organisation.

## Analytical Question
Do more competitive matches lead to higher player engagement, and can 
we predict match competitiveness in advance to actively improve it 
through smarter matchmaking?

## Data Source
- **Original dataset:** 7,851 real tournament match records from 
  June–July 2022 (565 players, 29 consecutive days, ~271 matches/day)
- **Synthetic dataset:** 50,000 additional rows generated to simulate 
  8 months of community growth, modelled using Elo-driven match 
  outcomes, player progression archetypes, and an engagement model
- **Total:** 57,851 rows, 839 unique players, June 2022 – March 2023

> **Important limitation:** The synthetic data was designed to behave 
> like a healthy growing community. The 16pp retention finding is a 
> pattern built into the generation model and should be validated 
> against real data before drawing empirical conclusions. The engagement 
> model baked in higher return rates for players who won, which 
> correlates with exciting matches — this is a known confound.

## Repository Navigation

| Folder/File | Purpose |
|---|---|
| `notebooks/` | EDA, feature engineering, and modelling notebooks in order |
| `src/` | Reusable functions for cleaning (`data_pipeline.py`) and feature engineering (`features.py`) |
| `docs/` | Pipeline diagram, stakeholder map, AI-use log *(Sprint 3)* |
| `app/` | Streamlit matchmaking recommendation tool, including the saved model and scaler it loads |
| `figures/` | Exported visualisations |
| `outputs/` | Saved models and metrics tables |
| `data/raw/` | Original unmodified source data |
| `data/processed/` | Cleaned and feature-engineered datasets |

## Engagement Definition
Engagement is defined as **monthly retention** — whether a player 
who was active in a given month was also active the following month. 
A monthly threshold was chosen because the tournament structure means 
daily and weekly return rates reflect scheduling continuity rather than 
genuine re-engagement decisions. This threshold should be validated 
against real community data.

## Engineered Features

| Feature | Description | Justification |
|---|---|---|
| `Elo_Diff` | Skill gap between players (P1 Elo − P2 Elo) | EDA showed larger gap = more Sweeps, smaller gap = more exciting matches |
| `P1_Elo` / `P2_Elo` | Individual Elo rating at match time | Absolute skill context beyond the gap alone |
| `WinRate_Diff` | Win rate gap between players (P1 − P2) | Captures form-based imbalance beyond Elo |
| `P1_Streak` / `P2_Streak` | Current win streak going into the match | Momentum signal — may predict performance above current Elo |
| `Series_Type` | Categorical match narrative — Sweep, Secure, Recovery, Neck & Neck, Comeback | Engineered from set-by-set sequence — source of the target variable |
| `Is_Exciting` | 1 if Neck & Neck or Comeback, 0 otherwise | Binary target variable derived from Series_Type — directly tied to the EDA retention finding |

All features calculated using only data prior to each match — no leakage.

**Note:** `Series_Type` and `Is_Exciting` are only knowable after a match 
is played. They are engineered as target variables, not predictors. 
The model uses pre-match features (Elo, win rate, streak) to predict 
`Is_Exciting` before the match occurs.

## Model Progression

| Stage | Model | Test ROC-AUC | Recall (Exciting) | Status |
|---|---|---|---|---|
| Baseline | DummyClassifier | 0.500 | 0.000 | ✅ Done |
| Simple | Logistic Regression (Elo_Diff only) | 0.800 | 0.665 | ✅ Done |
| Tuned | Logistic Regression (full features, GridSearchCV) | 0.802 | 0.706 | ✅ Done |
| **Tuned — Final** | **Random Forest (full features, GridSearchCV)** | **0.794** | **0.765** | ✅ **Selected** |

**Target variable:** `Is_Exciting` — predicting whether a matchup will 
produce a Neck & Neck or Comeback series before it is played.

**Why Random Forest was selected:** Recall on the exciting class was 
prioritized over precision, since for a matchmaking tool, missing a 
genuinely exciting matchup (false negative) is more costly than 
over-recommending one (false positive). The tuned Random Forest achieved 
the highest recall (76.5%) while remaining well-generalized (small train/test 
gap), thanks to `max_depth` and `min_samples_leaf` constraints found via 
`GridSearchCV`. Precision remains low (~10%) across all models due to 
severe class imbalance (5.5% exciting rate) — an acknowledged tradeoff 
given the stakeholder's priority.

## Running the App

```
cd app
pip install -r ../requirements.txt
streamlit run app.py
```

The app loads `final_model_rf.pkl` and `final_scaler.pkl` (saved in the 
`app/` folder alongside `app.py`) and predicts the probability that a 
given matchup will be an exciting match, based on both players' Elo, 
win rate, and win streak.

## Weekly Workflow

| Week | Focus | Status |
|---|---|---|
| Week 1 | Problem framing, pipeline design, data audit | ✅ Done |
| Week 2 | Cleaning, EDA, feature engineering, statistical testing | ✅ Done |
| Week 3 | Baseline and simple model | ✅ Done |
| Week 4 | Model tuning and optimisation | ✅ Done |
| Week 5 | Refactoring into .py scripts, Streamlit deployment | ✅ Done |
| Week 6 | Final polish and presentation | 🔜 In progress |

## Leakage Prevention
All engineered features are calculated using a sequential pass through 
chronologically sorted data. Each feature records the value before the 
current match is processed — no future information influences any feature 
value. The train/test split is applied after feature engineering on the 
full sorted dataset, with no transformations fit on test data.

The target variable `Is_Exciting` is derived from `Series_Type` which 
is only knowable after a match is played. It is used as a prediction 
target, never as a predictor.

During model tuning, leakage was additionally controlled by:
- Using `TimeSeriesSplit` instead of standard k-fold cross-validation, 
  so validation folds never precede training folds in time
- Fitting `GridSearchCV` only on training data — the test set was 
  untouched until final model evaluation
- Fitting the scaler (`StandardScaler`) only on training data 
  (`fit_transform` on train, `transform` on test)

## Limitations
- Dataset is partially synthetic — real-world validation is required
- The engagement model used in data generation baked in higher return 
  rates for winners, which correlates with exciting matches — this is 
  a known confound in the retention finding
- Monthly retention threshold is pragmatic rather than theoretically 
  optimal — weekly windows may be more meaningful in a real app context
- Cold
