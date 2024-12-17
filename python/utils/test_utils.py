import numpy as np
import pandas as pd
from scipy.stats import poisson
import itertools

import utils.data_utils as dutils
import fitting
import models


def generate_simulated_data():
    '''
    Generate some simple synthetic data for testing purposes - not suitable for
    testing models with seasonal components.
    '''
    gamma = 0.25
    eta = 0.4
    teams = {'Windows': [-0.8, -0.8, 0],
             'Linux': [0.1, 0.1, 1],
             'MacOS': [0.7, 0.7, 2],
             'Literally a Potato': [0.0, 0.0, 3]} # Clearly Windows is the worst team
    match_dict = {'HomeTeam': [], 'AwayTeam': [], 'FTHG': [], 'FTAG': [], 'TotG': [], 'HA': []}
    match_df = pd.DataFrame(match_dict)
    n_matches = 5000

    for matchup in list(itertools.combinations(teams.keys(), 2)):
        # Generate and add (home, away) = (team1, team2)
        home_idx = teams[matchup[0]][2]
        away_idx = teams[matchup[1]][2]
        data = pd.DataFrame({'home_idx': [home_idx],
                             'away_idx': [away_idx]})
        model = models.PoissonModel(data, {t: teams[t][-1] for t in teams.keys()}, test = True)
        home_rate = model.score_rate(teams[matchup[0]][0], teams[matchup[1]][1], gamma, eta)
        away_rate = model.score_rate(teams[matchup[1]][0], teams[matchup[0]][1], gamma, -eta)
        FTHG = poisson.rvs(home_rate, size = n_matches)
        FTAG = poisson.rvs(away_rate, size = n_matches)
        match_df = pd.concat([match_df,
                              pd.DataFrame({'HomeTeam': [matchup[0]] * n_matches,
                                            'AwayTeam': [matchup[1]] * n_matches,
                                            'home_idx': [home_idx] * n_matches,
                                            'away_idx': [away_idx] * n_matches,
                                            'FTHG': FTHG,
                                            'FTAG': FTAG,
                                            'TotG': FTHG + FTAG,
                                            'HA': FTHG - FTAG})], ignore_index = True)
        # Generate and add (home, away) = (team2, team1)
        home_idx = teams[matchup[1]][2]
        away_idx = teams[matchup[0]][2]
        data = pd.DataFrame({'home_idx': [home_idx],
                             'away_idx': [away_idx]})
        model = models.PoissonModel(data, {t: teams[t][-1] for t in teams.keys()}, test = True)
        home_rate = model.score_rate(teams[matchup[1]][0], teams[matchup[0]][1], gamma, eta)
        away_rate = model.score_rate(teams[matchup[0]][0], teams[matchup[1]][1], gamma, -eta)
        FTHG = poisson.rvs(home_rate, size = n_matches)
        FTAG = poisson.rvs(away_rate, size = n_matches)
        match_df = pd.concat([match_df,
                              pd.DataFrame({'HomeTeam': [matchup[1]] * n_matches,
                                            'AwayTeam': [matchup[0]] * n_matches,
                                            'home_idx': [home_idx] * n_matches,
                                            'away_idx': [away_idx] * n_matches,
                                            'FTHG': FTHG,
                                            'FTAG': FTAG,
                                            'TotG': FTHG + FTAG,
                                            'HA': FTHG - FTAG})], ignore_index = True)
    return match_df


