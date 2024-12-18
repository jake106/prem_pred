import numpy as np
import pandas as pd

import models
import utils.data_utils as dutils
import fitting


def import_model(
        data: pd.DataFrame,
        modeltype: str, 
        team_map: dict)->models.PoissonModel:
    '''
    Imports the relevant model for either forecast or simulation. 
    '''
    if not modeltype in ['simple', 'seasonal']:
        raise ValueError('Model type must be either "simple" or "seasonal".')

    if modeltype == 'simple':
        model = models.SimplePoisson(data, team_map).from_saved()
    elif modeltype == 'seasonal':
        model = models.ExtendedSeasonalPoisson(data, team_map).from_saved()
    return model


def compute_win_loss_draw_prob(
        data: pd.DataFrame,
        modeltype: str, 
        team_map: dict)->pd.DataFrame:
    '''
    Computes the win, loss, and draw probabilities for matches in 'data' using the
    skellam distribution

    Params:
    data - matches to calculate for
    modeltype - model type to use (either 'simple' or 'seasonal')

    Returns:
    Modifies and returns input dataframe with probabilities included
    '''
    model = import_model(data, modeltype, team_map)

    data['Home_rate'], data['Away_rate'] = model.scoring_rates()
    data['Home_prob'], data['Draw_prob'], data['Away_prob'] = dutils.skellam_dist(data.Home_rate,
                                                                data.Away_rate)
    return data


def build_league_table(data: pd.DataFrame, historical: pd.DataFrame)->pd.DataFrame:
    '''
    Build league table from match results.
    '''
    cols = ['Home_pts', 'Away_pts', 'HomeTeam', 'AwayTeam']
    data = pd.concat([historical[cols], data[cols]])
    data_home_pts = data.groupby(['HomeTeam'])['Home_pts'].sum()
    data_away_pts = data.groupby(['AwayTeam'])['Away_pts'].sum()
    data_proj_total_pts = data_home_pts + data_away_pts
    league_table = data_proj_total_pts.sort_values(ascending = False)
    return league_table


def predict_league_table(data: pd.DataFrame, historical: pd.DataFrame)->pd.Series:
    '''
    Predict the league table for the latest season.

    Params:
    data - match data to calculate league table for - can be combination of matches that have occured
    and those in the future. Must have model predictions.
    historical - historical data to fill in entire league table if used halfway through season.
    '''
    # Get only current league from histroical data
    historical = dutils.get_historic_pts(historical)
    data['Home_pts'], data['Away_pts'] = dutils.calc_exp_points(data.Home_prob,
                                           data.Draw_prob,
                                           data.Away_prob)
    # Merge prediction data with historical
    proj_league_table = build_league_table(data, historical)
    return proj_league_table


def simulate_league(
        data: pd.DataFrame,
        historical: pd.DataFrame,
        modeltype: str,
        team_map: dict,
        N_sim: int)->list:
    '''
    Simulated N_sim number of leagues based on combining historical data from current league with
    future fixture schedules.
    '''
    model = import_model(data, modeltype, team_map)

    historical = dutils.get_historic_pts(historical)
    data['Home_rate'], data['Away_rate'] = model.scoring_rates()
    simulated_tables = []
    for n in range(N_sim):
        home_rate = data.Home_rate.to_numpy()
        away_rate = data.Away_rate.to_numpy()
        data['HG'], data['AG'] = dutils.simulate_match_scores(home_rate, away_rate)
        data = dutils.winner_from_goals(data)
        data['Home_pts'], data['Away_pts'] = dutils.get_true_points(data)
        sim_league_table = build_league_table(data, historical)
        simulated_tables += [sim_league_table]
    return simulated_tables


def evaluate_pred(data: pd.DataFrame):
    '''
    Evaluate the predictions given by a certain model.

    Params:
    data - Match data with model predictions included as well as actual results to evaluate
           model accuracy.
    '''
    observed_totg = data['TotG'].to_numpy()
    expected_totg = data['Home_rate'].to_numpy() + data['Away_rate'].to_numpy()
    rmse = dutils.calc_rmse(expected_totg, observed_totg)
    rae = dutils.calc_rae(expected_totg, observed_totg)
    print('### Total goals prediction error ###')
    print(f'RMSE for total goals = {rmse:.04f}')
    print(f'RAE for total goals = {rae:.04f}')

    # Brier score measures accuracy of predictions in a probablistic manner
    # Best score is 0, worst score is 1
    observed_res = data.FTR.to_numpy()
    brier_score = dutils.calc_bs(data.Home_prob.to_numpy(),
                    data.Draw_prob.to_numpy(),
                    data.Away_prob.to_numpy(),
                    observed_res)
    print('### Predicted results score ###')
    print(f'Brier score for model win probabilities = {brier_score:.04f}')
    print('Brier score of below 0.25 is considered useful, best possible Brier score is 0')


