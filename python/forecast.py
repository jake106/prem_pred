import numpy as np
import pandas as pd

import models
import utils.data_utils as dutils
import fitting


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
    if not modeltype in ['simple', 'seasonal']:
        raise ValueError('Model type must be either "simple" or "seasonal".')

    if modeltype == 'simple':
        model = models.SimplePoisson(data, team_map).from_saved()
    elif modeltype == 'seasonal':
        model = models.ExtendedSeasonalPoisson(data, team_map).from_saved()

    data['Home_rate'], data['Away_rate'] = model.scoring_rates()
    data['Home_prob'], data['Draw_prob'], data['Away_prob'] = dutils.skellam_dist(data.Home_rate,
                                                                data.Away_rate)
    return data


def predict_league_table(data: pd.DataFrame):
    '''
    Predict the league table for the latest season.

    Params:
    data - match data to calculate league table for - can be combination of matches that have occured
    and those in the future. Must have model predictions.

    TODO - add ability to incorperate recent match results into model to improve prediction as
           matches occur.
    '''
    data['Home_pts'], data['Away_pts'] = dutils.calc_exp_points(data.Home_prob,
                                           data.Draw_prob,
                                           data.Away_prob)
    data_home_pts = data.groupby(['HomeTeam'])['Home_pts'].sum()
    data_away_pts = data.groupby(['AwayTeam'])['Away_pts'].sum()
    data_proj_total_pts = data_home_pts + data_away_pts
    proj_league_table = data_proj_total_pts.sort_values(ascending = False)
    print('Projected league table:')
    print(proj_league_table)


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


