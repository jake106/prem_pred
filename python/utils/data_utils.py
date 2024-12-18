import numpy as np
import pandas as pd
import csv
import datetime as dt
import os
from scipy.stats import skellam, poisson

import plotting


def read_data(filename: str)->pd.DataFrame:
    '''Simple function to read data into a pandas dataframe. File extension added in the func.'''
    df = pd.read_csv(f'./data/{filename}.csv')
    return df


def create_datasets(split_date: dt.date)->(pd.DataFrame, pd.DataFrame):
    '''
    Splits downloaded data by date into fitting data and prediction data.
    '''
    df = read_data('data_prem_all')
    df, team_map = assign_team_index(df)
    df.Date = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in df.Date]
    # Define training and testing dates
    fitset = df[df.Date <= split_date]
    simset = df[(df.Date > split_date) & (df.Date <= dt.date(2025, 12, 23))]
    return fitset, simset, team_map 


def home_advantage_mean(df: pd.DataFrame)->float:
    '''
    Calculate mean home advantage (HA) for a dataset:
    HA = (goals by home team - goals by away team) / games played
    '''
    home_goals = df.FTHG.sum()
    home_conceded = df.FTAG.sum()
    total_games = len(df.index)
    return (home_goals - home_conceded) / total_games


def home_advantage_match(df: pd.DataFrame)->np.ndarray:
    '''
    Calculate per-match home advantage for a sample.
    HA = goals by home team - goals by away team
    '''
    return np.array(df.FTHG - df.FTAG)


def add_ha_totg(df:pd.DataFrame)->pd.DataFrame:
    '''Add home advantage and total goals to dataframe'''
    df['HA'] = home_advantage_match(df)
    df['TotG'] = df['FTHG']+df['FTAG']
    return df


def create_model_dataset(df: pd.DataFrame)->(np.ndarray, np.ndarray):
    '''Creates dataset for model training and inference as (FTHG, FTAG)'''
    return (df['FTHG'].to_numpy(), df['FTAG'].to_numpy())


def create_model_dataset_extended(df: pd.DataFrame)->(np.ndarray, np.ndarray, np.ndarray):
    '''Creates dataset for model training and inference as (FTHG, FTAG)'''
    return (df['FTHG'].to_numpy(), df['FTAG'].to_numpy(), df['Season_prop'].to_numpy())


def single_season_prop(df: pd.DataFrame, season: [dt.date, dt.date])->pd.DataFrame:
    '''Calculates the proportion a given match day is through a single season.'''
    days_since_start = df.Date - season[0]
    total_days = season[1] - season[0]
    long_season = days_since_start / total_days
    pd.options.mode.chained_assignment = None
    df.loc[:, 'Season_prop'] = long_season.values
    return df


def add_season_prop(df: pd.DataFrame)->pd.DataFrame:
    '''
    Adds proportion of way through the season a given matchday is.
    Since season lengths vary, we define 'long season' as between the 1st August and 31st July,
    to capture all variations in season length (even though this may have a minor impact on the
    prediction accuracy).
    '''
    # TODO: normalise over season range, set end of season manually for incomplete seasons
    start_date = df.Date.min().year
    end_date = df.Date.max().year
    if start_date == end_date:
        season = calc_season_bounds(df, start_date)
        return single_season_prop(df, season)
    df_out = pd.DataFrame({})
    for j in range(start_date, end_date):
        season = [dt.date(j, 8, 1), dt.date(j+1, 7, 31)]
        this_df = df[(df.Date > season[0]) & (df.Date < season[1])]
        this_df = single_season_prop(this_df, season)
        df_out = pd.concat([df_out, this_df])
    return df_out


def assign_team_index(fitset: pd.DataFrame)->(pd.DataFrame, dict):
    '''
    Assign per-team index to data, for use in sample model.
    Using zero-indexing to make it easier to use with python.
    '''
    all_teams = fitset.HomeTeam.unique()
    team_map = {t: i for i, t in enumerate(all_teams)}
    fitset['home_idx'] = fitset.HomeTeam.map(team_map.get)
    fitset['away_idx'] = fitset.AwayTeam.map(team_map.get)
    return fitset, team_map


def extract_team_index(simset: pd.DataFrame)->(np.ndarray, np.ndarray):
    '''
    Extracts team indices such that they can be used for model training and inference.
    Just retuns (home_indices, away_indices)
    '''
    home_idx = simset.home_idx
    away_idx = simset.away_idx
    return (home_idx.to_numpy(), away_idx.to_numpy())


def get_global_params(fitset):
    '''
    Extract Poisson model global parameter starting values from training data.
    '''
    fitset = fitset.dropna(inplace=False)
    # If there are only nan values we are predicting so this will be overwritten anyway
    if not len(fitset.index):
        return 1, 1
    tg = fitset['TotG'].mean()
    ha = fitset['HA'].mean()
    gamma = np.log(np.sqrt(tg**2 - ha**2) / 2)
    eta = np.log((tg + ha) / ((tg - ha)))
    # Mean values from dataset
    lamb_exp = (tg+ha)/2
    mu_exp = (tg-ha)/2
    print(f'Global coefficients gamma and eta = {gamma:.04f}, {eta:.04f}')
    return gamma, eta


def calc_rmse(predicted: np.ndarray, true:np.ndarray)->np.float64:
    '''
    Calculates the square root of the mean-squared error between model predictions
    and observations.
    '''
    N = len(predicted)
    difference = (predicted - true)**2
    return np.sqrt(np.sum(difference) / N)


def calc_rae(predicted: np.ndarray, true:np.ndarray)->np.float64:
    '''
    Calculates the square root of the absolute error between model predictions
    and observations.
    '''
    N = len(predicted)
    difference = abs(predicted - true)
    return np.sqrt(np.sum(difference) / N)


def calc_bs(
        home_win_prob: np.ndarray,
        draw_prob: np.ndarray,
        away_win_prob: np.ndarray,
        true_result: np.ndarray)->np.float64:
    '''
    Calculates Brier score for a given model with three possible outcomes:
    home win, draw, away win
    Note: this is not the commonly used binary Brier score, but the original multi-class variant.
    Since there are three classes, the output is divided by 3 to equate to the popular
    binary Brier score.
    '''
    N = len(true_result)
    home_obs = np.ones(N)*[true_result == 'H']
    draw_obs = np.ones(N)*[true_result == 'D']
    away_obs = np.ones(N)*[true_result == 'A']
    bs_unnorm = np.sum([(home_win_prob - home_obs)**2,
                        (draw_prob - draw_obs)**2,
                        (away_win_prob - away_obs)**2])
    return bs_unnorm/N/3


def skellam_dist(lamb: np.ndarray, mu: np.ndarray)->(np.ndarray, np.ndarray, np.ndarray):
    '''
    Skellam distribution for scoring rates lambda and mu.
    Takes arrays or floats as input detailing scoring rates and returns probability
    that the scores are equal, or that one is above another.
    Returns probabilities for the three possible match outcomes as (home win, draw, away win).
    '''
    rv = skellam(lamb, mu)
    prob_draw = rv.pmf(0)
    # rv.cdf is inclusive of k, rv.sf is not
    prob_away_win = rv.cdf(-1)
    prob_home_win = rv.sf(0)
    assert np.all(prob_draw + prob_home_win + prob_away_win) == 1, 'Probabilities should sum to 1.'
    return prob_home_win, prob_draw, prob_away_win


def calc_exp_points(
        home_win_prob: np.ndarray,
        draw_prob: np.ndarray,
        away_win_prob: np.ndarray)->(np.ndarray, np.ndarray):
    '''
    Calculates expected points for home and away teams from a given match using:
    expected home points = 3*P(home win) + P(Draw)
    expected away points = 3*P(away_win) + P(Draw)
    returns two arrays of expected (home, away) points.
    '''
    home_pts = 3 * home_win_prob + draw_prob
    away_pts = 3 * away_win_prob + draw_prob
    return home_pts, away_pts


def calc_season_bounds(df: pd.DataFrame, start_date: int)->[dt.date, dt.date]:
    '''
    Calculates beginnning and end of a given season
    '''
    if df.Date.max().month < 6:
        # Current season started last year
        season = [dt.date(start_date - 1, 8, 1), dt.date(start_date, 7, 31)]
    else:
        # Current season ends next year
        season = [dt.date(start_date, 8, 1), dt.date(start_date + 1, 7, 31)]
    return season


def get_true_points(df: pd.DataFrame)->(np.ndarray, np.ndarray):
    '''
    Calculates actual points for the most recent season in a historical dataframe
    '''
    pt_map = pd.DataFrame({'FTR': ['H', 'A', 'D'], 'home_pts': [3, 0, 1], 'away_pts': [0, 3, 1]})
    df = df.merge(pt_map, on = 'FTR', how='left')
    return df.home_pts.to_numpy(), df.away_pts.to_numpy()


def get_historic_pts(historical: pd.DataFrame)->pd.DataFrame:
    pd.options.mode.chained_assignment = None
    start_date = historical.Date.max().year
    season = calc_season_bounds(historical, start_date)
    historical = historical[historical.Date > season[0]]
    historical['Home_pts'], historical['Away_pts'] = get_true_points(historical)
    return historical


def winner_from_goals(df: pd.DataFrame)->pd.DataFrame:
    '''
    Fills in 'FTR' column of dataframe from simulated match results
    '''
    df['FTR'] = df['FTR'].mask(df['HG'] == df['AG'], 'D')
    df['FTR'] = df['FTR'].mask(df['HG'] > df['AG'], 'H')
    df['FTR'] = df['FTR'].mask(df['HG'] < df['AG'], 'A')
    return df


def simulate_match_scores(
        home_score_rate: np.ndarray,
        away_score_rate: np.ndarray)->(np.ndarray, np.ndarray):
    '''
    Simulates the results of matches using independent Poisson distributions
    with provided home and away scoring rates.
    '''
    home_scores = poisson.rvs(home_score_rate)
    away_scores = poisson.rvs(away_score_rate)
    return home_scores, away_scores


