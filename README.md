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
| `src/` | Reusable functions for cleaning, features, and modelling |
| `docs/` | Pipeline diagram, stakeholder map, AI-use log |
| `app/` | Streamlit matchmaking recommendation tool |
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
| `Elo_Diff` | Absolute skill gap between players | EDA showed larger gap = more Sweeps, smaller gap = more exciting matches |
| `P1_Elo` / `P2_Elo` | Individual Elo rating at match time | Absolute skill context beyond the gap alone |
| `WinRate_Diff` | Absolute win rate gap between players | Captures form-based imbalance beyond Elo |
| `P1_Streak` / `P2_Streak` | Current win streak going into the match | Momentum signal — may predict performance above current Elo |
| `Is_Exciting` | 1 if Neck & Neck or Comeback, 0 otherwise | Target variable — derived from the EDA retention finding |

All features calculated using only data prior to each match — no leakage.

## Model Progression

| Stage | Model | Purpose |
|---|---|---|
| Baseline | DummyClassifier | Establishes minimum benchmark |
| Simple | Logistic Regression (Elo_Diff only) | Tests strongest single feature |
| Tuned | Random Forest / XGBoost + GridSearchCV | Optimised full feature set |

**Target variable:** `Is_Exciting` — predicting whether a matchup will 
produce a Neck & Neck or Comeback series before it is played.

## Weekly Workflow

| Week | Focus |
|---|---|
| Week 1 | Problem framing, pipeline design, data audit |
| Week 2 | Cleaning, EDA, feature engineering, statistical testing |
| Week 3 | Baseline and simple model |
| Week 4 | Model tuning and optimisation |
| Week 5 | Refactoring into .py scripts, Streamlit deployment |
| Week 6 | Final polish and presentation |

## Leakage Prevention
All engineered features are calculated using a sequential pass through 
chronologically sorted data. Each feature records the value before the 
current match is processed — no future information influences any feature 
value. The train/test split is applied after feature engineering on the 
full sorted dataset, with no transformations fit on test data.

The target variable `Is_Exciting` is derived from `Series_Type` which 
is only knowable after a match is played. It is used as a prediction 
target, never as a predictor.

## Limitations
- Dataset is partially synthetic — real-world validation is required
- The engagement model used in data generation baked in higher return 
  rates for winners, which correlates with exciting matches — this is 
  a known confound in the retention finding
- Monthly retention threshold is pragmatic rather than theoretically 
  optimal — weekly windows may be more meaningful in a real app context
- Cold start problem: players with fewer than 20 matches have 
  unreliable Elo and win rate estimates
- The model predicts match excitement but cannot guarantee it — 
  matchmaking recommendations are probabilistic, not deterministic

## Future Improvements
- Collect real match data from a local club to validate findings
- Add head-to-head history as a feature
- Test weekly retention as an alternative engagement threshold
- Expand Streamlit app into a full community-facing product
- Model player progression explicitly to personalise matchmaking 
  over time
