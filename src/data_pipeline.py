"""
Ping Pong Community Analytics — Data Pipeline
-----------------------------------------------
Reusable loading and cleaning functions, refactored from
notebooks/01_eda_feature_engineering.ipynb (Section 1-2).

Usage:
    from data_pipeline import load_and_clean_data

    df = load_and_clean_data('data/raw/ping_pong_data.csv')
"""

import pandas as pd


def parse_datetime(df):
    """
    Parse Date and Time columns into a single DateTime column.
    Sorts chronologically — required for leakage-free feature engineering.
    """
    df = df.copy()
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
    )
    df = df.sort_values('DateTime').reset_index(drop=True)
    return df


def validate_scores(df):
    """
    Check all played games have valid ping pong scores.
    Rules: first to 11 points, win by at least 2, no point cap.
    Returns list of row indices with invalid scores.
    """
    issues = []
    for idx, row in df.iterrows():
        games = int(row['Sets_P1']) + int(row['Sets_P2'])
        for g in range(1, games + 1):
            s1, s2 = row[f'P1_G{g}'], row[f'P2_G{g}']
            if not (max(s1, s2) >= 11 and abs(s1 - s2) >= 2):
                issues.append(idx)
    return issues


def validate_series(df):
    """
    Check all series have valid best-of-5 outcomes.
    Valid outcomes: 3-0, 3-1, 3-2 (or reversed).
    Returns list of row indices with invalid series.
    """
    valid = {(3, 0), (3, 1), (3, 2), (0, 3), (1, 3), (2, 3)}
    mask = df.apply(
        lambda r: (r['Sets_P1'], r['Sets_P2']) not in valid, axis=1
    )
    return df[mask].index.tolist()


def check_nulls(df):
    """
    Report null counts across all columns.
    G4/G5 nulls are expected — flags anything else.
    """
    expected_null_cols = ['P1_G4', 'P2_G4', 'P1_G5', 'P2_G5']
    nulls = df.isnull().sum()
    unexpected = nulls[
        ~nulls.index.isin(expected_null_cols) & (nulls > 0)
    ]
    return nulls, unexpected


def check_duplicates(df):
    """Return count of fully duplicated rows."""
    return df.duplicated().sum()


def flag_out_of_hours(df, start_hour=8, end_hour=22):
    """
    Flag matches outside 8am-10pm playing hours.
    Real data pre-dates this constraint so out-of-hours real rows are expected.
    """
    df = df.copy()
    hour = df['DateTime'].dt.hour
    df['OutOfHours'] = (hour < start_hour) | (hour >= end_hour)
    return df


def validate_g4_g5_nulls(df):
    """
    Confirm G4/G5 are null only when the series ended before that game.
    Returns count of rows where nulls appear incorrectly.
    """
    games_played = df['Sets_P1'] + df['Sets_P2']
    g4_wrong = (games_played >= 4) & df['P1_G4'].isnull()
    g5_wrong = (games_played == 5) & df['P1_G5'].isnull()
    return g4_wrong.sum(), g5_wrong.sum()


def run_quality_checks(df, verbose=True):
    """
    Run the full quality-check suite used in notebook 01, Section 2.
    Returns a dict summarizing the results. Prints a report if verbose=True.
    """
    dupes = check_duplicates(df)
    bad_series = validate_series(df)
    bad_scores = validate_scores(df.sample(min(5000, len(df)), random_state=42))
    nulls, unexpected = check_nulls(df)
    g4_wrong, g5_wrong = validate_g4_g5_nulls(df)
    df_flagged = flag_out_of_hours(df)
    real_ooh = df_flagged[(df_flagged['OutOfHours']) & (df_flagged['Source'] == 'Real')]
    synth_ooh = df_flagged[(df_flagged['OutOfHours']) & (df_flagged['Source'] == 'Synthetic')]

    results = {
        'duplicate_rows': dupes,
        'invalid_series': len(bad_series),
        'invalid_scores_sampled': len(bad_scores),
        'unexpected_null_cols': len(unexpected),
        'g4_null_when_should_exist': g4_wrong,
        'g5_null_when_should_exist': g5_wrong,
        'out_of_hours_real': len(real_ooh),
        'out_of_hours_synthetic': len(synth_ooh),
    }

    if verbose:
        print('=' * 50)
        print('QUALITY CHECK REPORT')
        print('=' * 50)
        for key, val in results.items():
            print(f'{key:30s}: {val}')

    return results


def load_and_clean_data(path):
    """
    Load the raw ping pong CSV, parse/sort by datetime, tag the data
    source (Real vs Synthetic), and run quality checks.

    This dataset was synthetically generated under strict rules, so
    cleaning here is verification rather than correction — see notebook
    01 for the reasoning behind that choice.
    """
    df = pd.read_csv(path)
    df = parse_datetime(df)

    # Tag data source for reference in EDA (first 7,851 rows are real)
    df['Source'] = df['X'].apply(lambda x: 'Real' if x <= 7851 else 'Synthetic')

    run_quality_checks(df, verbose=True)

    return df
