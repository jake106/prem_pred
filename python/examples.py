import numpy as np
import pandas as pd

import utils.data_utils as dutils


def aston_villa(sim_tables):
    '''
    Simulates the probability of Aston Villa finishing in the top 5 of the 2024/2025
    Premier League based on data up to and including 17/12/24.
    '''
    villa_top5 = 0
    N_sims = len(sim_tables)
    for s in sim_tables:
        top5_teams = s.iloc[:5].index
        if 'Aston Villa' in top5_teams:
            villa_top5 += 1
    prob_villa_top5 = villa_top5 / N_sims
    print(f'Probability of Aston Villa finishing in the top 5 spots from {N_sims} simulations:')
    print(f'{prob_villa_top5:.03f}')
