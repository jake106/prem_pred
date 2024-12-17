import pandas as pd
import numpy as np
import fitting
import utils.test_utils as tutils
import utils.data_utils as dutils
import models


def check_model_order(
        alpha: (np.ndarray, np.ndarray),
        gamma: float,
        eta: float,
        team_map: dict,
        fitset: pd.DataFrame):
    '''
    Ensure the order of the predicted average goals from the model and actual
    average goals in data are at least similar.

    Params:
    alpha, gamma, eta - Model parameters. Alpha contains both attacking and defensive strengths
                        for historical reasons.
    team_map - Dictionary mapping team names to indices.
    fitset - Dataset model trained on.
    '''
    model = models.PoissonModel(fitset, team_map)
    av_pred_scores_home = model.score_rate(alpha[0], alpha[1].mean(), gamma, eta)
    team_vals_home = {t: hv for t, hv in zip(team_map.keys(), av_pred_scores_home)}
    fitset_team_home = fitset.groupby('HomeTeam')['FTHG'].mean().sort_values()

    av_pred_scores_away = model.score_rate(alpha[0], alpha[1].mean(), gamma, -eta)
    team_vals_away = {t: hv for t, hv in zip(team_map.keys(), av_pred_scores_away)}
    fitset_team_away = fitset.groupby('AwayTeam')['FTAG'].mean().sort_values()

    print('## The following should be similarly ordered: ##')
    print('# HomeTeam team scoring rates: #')
    print('Predicted home team scoring rates:')
    print(pd.Series(team_vals_home).sort_values())
    print('Actual home team scoring rates')
    print(fitset_team_home)
    print('# AwayTeam team scoring rates: #')
    print('Predicted away team scoring rates:')
    print(pd.Series(team_vals_away).sort_values())
    print('Actual away team scoring rates')
    print(fitset_team_away)


def check_simulated_data():
    '''
    Generate simulated data wih known scoring rates and compare model fit to actual data.
    If model works correctly these should be very similar, and this test should highlight any
    issues with the model. 

    TODO: create comparible check for seasonal model.
    '''
    sim_df = tutils.generate_simulated_data()
    gamma, eta = dutils.get_global_params(sim_df)
    sim_df, team_map = dutils.assign_team_index(sim_df)

    model = models.SimplePoisson(sim_df, team_map)
    alpha, gamma, eta, nll = model.fit(save = False)
    print(f'''Team results:
        {[(index, name, a, b) for name, index, a, b in zip(team_map.keys(), team_map.values(), alpha[0], alpha[1])]}''')
    check_model_order(alpha, gamma, eta, team_map, sim_df)



