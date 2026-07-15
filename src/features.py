"""
Ping Pong Community Analytics — Feature Engineering
------------------------------------------------------
Reusable feature-engineering functions, refactored from
notebooks/01_eda_feature_engineering.ipynb (Sections 4-5).

Builds, in order:
    - P1_Elo, P2_Elo, Elo_Diff       (pre-match skill ratings)
    - P1_WinRate, P2_WinRate         (career win rate before this match)
    - WinRate_Diff                   (signed win-rate gap, P1 - P2)
    - P1_Streak, P2_Streak           (current win streak before this match)
    - Series_Type, Is_Exciting       (target variables, post-match only)

All features are computed via a single chronological pass through the
data, so no feature ever uses information from a match that hasn't
been played yet.

Usage:
    from features import engineer_features

    df = engineer_features(df)   # df must already be sorted by DateTime
"""

import pandas as pd

ELO_BASE = 1000
ELO_K = 24


def compute_elo(df):
    """
    Compute pre-match Elo ratings for both players, updating after
    each match in chronological order. Adds P1_Elo, P2_Elo, Elo_Diff.
    """
    elo = {}

    def get_elo(p):
        return elo.get(p, ELO_BASE)

    def exp_score(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_elo(w, l):
        rw, rl = get_elo(w), get_elo(l)
        ew = exp_score(rw, rl)
        elo[w] = rw + ELO_K * (1 - ew)
        elo[l] = rl + ELO_K * (0 - (1 - ew))

    p1_elo_list, p2_elo_list = [], []
    for _, row in df.iterrows():
        p1, p2 = row['Player1'], row['Player2']
        p1_elo_list.append(round(get_elo(p1), 1))
        p2_elo_list.append(round(get_elo(p2), 1))
        winner = p1 if row['HomeWinner'] == 1 else p2
        loser = p2 if row['HomeWinner'] == 1 else p1
        update_elo(winner, loser)

    df = df.copy()
    df['P1_Elo'] = p1_elo_list
    df['P2_Elo'] = p2_elo_list
    df['Elo_Diff'] = df['P1_Elo'] - df['P2_Elo']
    return df


def compute_winrate_and_streak(df):
    """
    Compute each player's career win rate and current win streak,
    both evaluated *before* the match in that row is played.
    Adds P1_WinRate, P2_WinRate, WinRate_Diff, P1_Streak, P2_Streak.
    """
    df = df.copy()

    # ── Win rate ──────────────────────────────────────────────
    player_wins = {}
    player_matches = {}
    p1_wr, p2_wr = [], []

    for _, row in df.iterrows():
        p1, p2 = row['Player1'], row['Player2']
        p1_won = row['HomeWinner'] == 1
        for player, rates in [(p1, p1_wr), (p2, p2_wr)]:
            m = player_matches.get(player, 0)
            w = player_wins.get(player, 0)
            rates.append(round(w / m, 4) if m > 0 else None)
        player_matches[p1] = player_matches.get(p1, 0) + 1
        player_matches[p2] = player_matches.get(p2, 0) + 1
        player_wins[p1] = player_wins.get(p1, 0) + (1 if p1_won else 0)
        player_wins[p2] = player_wins.get(p2, 0) + (0 if p1_won else 1)

    df['P1_WinRate'] = p1_wr
    df['P2_WinRate'] = p2_wr
    df['WinRate_Diff'] = df['P1_WinRate'] - df['P2_WinRate']

    # ── Win streak ────────────────────────────────────────────
    streak = {}

    def get_streak(player):
        return streak.get(player, 0)

    def update_streak(player, won):
        current = get_streak(player)
        streak[player] = current + 1 if won else 0

    p1_streak, p2_streak = [], []
    for _, row in df.iterrows():
        p1, p2 = row['Player1'], row['Player2']
        p1_won = row['HomeWinner'] == 1
        p1_streak.append(get_streak(p1))
        p2_streak.append(get_streak(p2))
        update_streak(p1, p1_won)
        update_streak(p2, not p1_won)

    df['P1_Streak'] = p1_streak
    df['P2_Streak'] = p2_streak

    return df


def categorize_series(row):
    """
    Classify a match series into one of five narrative types.
    Uses only the set sequence — no future information.
    """
    p1, p2 = row['Sets_P1'], row['Sets_P2']
    sets = [
        1 if row[f'P1_G{g}'] > row[f'P2_G{g}'] else 2
        for g in range(1, p1 + p2 + 1)
    ]
    winner, loser = (1, 2) if p1 == 3 else (2, 1)

    if (p1 == 3 and p2 == 0) or (p1 == 0 and p2 == 3):
        return 'Sweep'
    if sets[0] == loser and sets[1] == loser:
        return 'Comeback'
    if all(sets[i] != sets[i + 1] for i in range(len(sets) - 1)):
        return 'Neck & Neck'

    p1s = p2s = 0
    won_lead = lost_lead = False
    for s in sets:
        if s == 1:
            p1s += 1
        else:
            p2s += 1
        if winner == 1 and p1s > p2s:
            won_lead = True
        elif winner == 2 and p2s > p1s:
            won_lead = True
        if won_lead:
            if winner == 1 and p2s > p1s:
                lost_lead = True
            elif winner == 2 and p1s > p2s:
                lost_lead = True
    if won_lead and lost_lead:
        return 'Recovery'
    return 'Secure'


def compute_target(df):
    """
    Add Series_Type (categorical match narrative) and Is_Exciting
    (binary target: 1 if Neck & Neck or Comeback). These are only
    knowable after a match is played — target variables, not predictors.
    """
    df = df.copy()
    df['Series_Type'] = df.apply(categorize_series, axis=1)
    df['Is_Exciting'] = df['Series_Type'].isin(['Neck & Neck', 'Comeback']).astype(int)
    return df


def engineer_features(df):
    """
    Run the full feature-engineering pipeline in the correct order:
    Elo -> win rate/streak -> target variables.

    df must already be chronologically sorted (see data_pipeline.parse_datetime).
    """
    df = compute_elo(df)
    df = compute_winrate_and_streak(df)
    df = compute_target(df)
    return df
