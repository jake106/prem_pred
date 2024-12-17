import pandas as pd
import numpy as np

import models
import utils.data_utils as dutils
import model_checks


def simple_model(fitset: pd.DataFrame, team_map: dict):
    '''
    Fit and assess a simple Poisson model. Goals for each team are modelled with Poisson
    distributions, with the scoring rate for each distribution the log of the product of
    parameters describing each teams attacking (alpha) and defending (beta) strength, as well
    as two global dataset values describing total goals (gamma) and home advantage (eta).

    Params:
    fitset - Pandas dataframe to fit the model on.
    team_map - dictionary of team names and indices, must be same for training and inference.
    '''
    # Create training data and train model
    simple_model = models.SimplePoisson(fitset, team_map)
    alpha, gamma, eta, nll = simple_model.fit(save = True)
    print(f'''Team results:
        {[(index, name, a, b) for name, index, a, b in zip(team_map.keys(), team_map.values(), alpha[0], alpha[1])]}''')
    print(f'Variance in team performance = {np.var(alpha)}')

    # Sanity check - the order of these should be roughly the same
    model_checks.check_model_order(alpha, gamma, eta, team_map, fitset)
    model_checks.check_simulated_data()
    # Add 2 to AIC for floating eta and gamma
    AIC = len(team_map.values()) + 2 + nll
    print(f'Model AIC = {AIC}')


def extended_model(fitset: pd.DataFrame, team_map: dict)->np.ndarray:
    '''
    Fit and assess a Poisson model with a value of gamma that varies seasonally. Goals for each
    team are modelled with Poisson distributions, with the scoring rate for each distribution
    the log of the product of parameters describing each teams attacking (alpha) and
    defending (beta) strength, as well as two global dataset values describing total
    goals (gamma), modelled as linear in time through each season a given match takes place,
    and home advantage (eta).

    Params:
    fitset - Pandas dataframe to fit the model on.
    team_map - dictionary of team names and indices, must be same for training and inference.
    '''
    # Add prop. of way through season to calculate seasonal trends
    extended_model = models.ExtendedSeasonalPoisson(fitset, team_map)
    alpha, prop, eta, nll = extended_model.fit(save = True)
    print(f'''Team results:
        {[(index, name, a, b) for name, index, a, b in zip(team_map.keys(), team_map.values(), alpha[0], alpha[1])]}''')

    # TODO: Write sanity check for extended model similar to the one used for the simple model
    # Add x to nll for fitted gamma and floating eta
    AIC = len(team_map.values()) + 3 + nll
    print(f'Model AIC = {AIC}')


