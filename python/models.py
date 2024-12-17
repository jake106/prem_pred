import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import pandas as pd
import sys

import utils.data_utils as dutils


class PoissonModel:
    '''
    Poisson model class for predicting goals scored during a football
    match based on the teams competing.

    Attributes:

    data - Dataframe of data to fit or predict. Typically if used to predict, the from_saved
           method will be called with initialisation.
    team_map - Dictionary mapping team names to indices.
    test - Bool to put model in 'testing mode'.
    split_loc - Index at which to split flattened model parameters from minimiser into team
                and global parameters.
    team_indices - Per-match indices of the home and away team.
    gamma, eta - Starting values of global model parameters.
    fixed - Values of fixed model parameters needed to make the model identifiable.
    init_params -Starting values of per-team model parameters.
    '''
    def __init__(
            self,
            data: pd.DataFrame,
            team_map: dict,
            test: bool = False):
        '''
        The test flag indicates we are initialising the model on an incomplete dataset so cannot
        calculate total goals and home advantage etc.
        '''
        self.split_loc = len(team_map.keys()) - 1
        self.team_indices = dutils.extract_team_index(data)
        if not test:
            self.data = dutils.add_ha_totg(data)
            self.gamma, self.eta = dutils.get_global_params(data)
        else:
            self.gamma, self.eta = 0, 0
            self.data = data
        # Model fit initialises with a fixed param team not included in init_params
        self.fixed = (0.0, 0.0)
        self.init_params = (np.zeros(len(team_map.keys())-1),
                            np.zeros(len(team_map.keys())-1))

    def score_rate(
            self,
            alpha: float or np.ndarray,
            beta: float or np.ndarray,
            gamma: float,
            eta: float)->float:
        '''
        Individual score rate calculation, alpha and beta can be arrays or floats depending on
        how many matches the score rate is calculated for i.e. for one match, can be set to float.

        Params:
        alpha: A teams attaching strength
        beta: A teams defensive weakness (i.e. -beta is defensive strength)
        gamma: The overal goals expected
        eta: The home advantage coefficient

        To take into account away team disadvantage, negative eta should be passed to this function.
        '''
        return np.exp(alpha + beta + gamma + (eta / 2)) 

    def home_score_rate(
            self,
            team_params: (np.ndarray, np.ndarray),
            gamma: float or np.ndarray,
            eta: float or np.ndarray)->np.ndarray:
        '''
        Home team goal scoring rate

        Params:
        team_params: tuple of vectors of per-team alpha and beta values
        gamma, eta: global model parameters

        Returns:
        lamb: home team scoring rate
        '''
        lamb = self.score_rate(team_params[0][self.team_indices[0]],
                               team_params[1][self.team_indices[1]], gamma, eta)
        return lamb

    def away_score_rate(
            self,
            team_params: (np.ndarray, np.ndarray),
            gamma: float or np.ndarray,
            eta: float or np.ndarray)->np.ndarray:
        '''
        Away team goal scoring rate
    
        Params:
        team_params: tuple of vectors of per-team alpha and beta values
        gamma, eta: global model parameters

        Returns:
        mu: away team scoring rate
        '''
        mu = self.score_rate(team_params[0][self.team_indices[1]],
                             team_params[1][self.team_indices[0]], gamma, -eta)
        return mu

    def save_model(
            self,
            alpha_flat: np.ndarray,
            filename: str = 'default'):
        '''
        Saves model parameters as a csv.

        Params:
        alpha_flat: model parameters as flat array
        filename: name of csv
        '''
        np.savetxt(f'./model_params/{filename}.csv', alpha_flat, delimiter = ',')

    def load_model(
            self,
            filename: str = 'default'):
        '''
        Loads model from params csv file.

        Params:
        filename: name of model file to load
        '''
        try:
            alpha_flat = np.genfromtxt(f'./model_params/{filename}.csv', delimiter = ',')
        except OSError as e:
            print('### Error! Model not trained! ###')
            print('Run with just --simple or --extended to train model before evaluation')
            sys.exit()
        return alpha_flat


class SimplePoisson(PoissonModel):
    def __init__(self, data, team_map):
        super().__init__(data, team_map)
        self.fitdata = dutils.create_model_dataset(self.data)

    def from_saved(self):
        alpha_flat = self.load_model('simple')
        self.alpha, self.gamma, self.eta = self.convert_model_params(alpha_flat)
        return self
        
    def convert_model_params(self, alpha_flat):
        alpha = (np.append(alpha_flat[:self.split_loc], self.fixed[0]),
                 np.append(alpha_flat[self.split_loc:-2], self.fixed[1]))
        gamma = alpha_flat[-2]
        eta = alpha_flat[-1]
        return alpha, gamma, eta

    def nll(self,
            all_params: (np.ndarray, np.ndarray),
            *data: (np.ndarray, np.ndarray))->float:
        '''
        Negative log likelihood function of a simple Poisson goal model.

        Params:
        data: [home_goals, away_goals]
        all_params: All floating model parameters as [alpha, beta, gamma, eta]
        fixed: fixed values of alpha and beta for a given team to ensure model is identifiable

        Returns:
        nll: negative log likelihood osimple poisson model fit to data
        '''
        p_params = (np.append(all_params[:self.split_loc], self.fixed[0]),
                    np.append(all_params[self.split_loc:-2], self.fixed[1]))
        gamma = all_params[-2]
        eta = all_params[-1]
        nll_h = - poisson.logpmf(data[0],
                      self.home_score_rate(p_params, gamma, eta)).sum()
        nll_a = - poisson.logpmf(data[1],
                      self.away_score_rate(p_params, gamma, eta)).sum()
    
        return nll_h + nll_a

    def fit(self, save: bool = False)->np.ndarray:
        '''
        Minimizes nll for bi-poisson model, the 0th tuple value always describes
        the home team.

        Params:

        Returns:
        alpha: per-team parameters
        gamma, eta: global parameters
        nll: fit result negative log likelihood
        '''
        # Fix -1th element of alpha and beta
        init_params = np.append(self.init_params, [self.gamma, self.eta])
        result = minimize(self.nll, init_params,
                    args=(self.fitdata),
                    method='SLSQP', options={'maxiter': 30000})

        print(result.message)
        alpha_flat = result.x
        if save:
            self.save_model(alpha_flat, 'simple')
        nll = result.fun
        self.alpha, self.gamma, self.eta = self.convert_model_params(alpha_flat)
        print(f'Fitter negative log likelihood: {nll}')
        return self.alpha, self.gamma, self.eta, nll

    def scoring_rates(self)->(np.ndarray, np.ndarray):
        '''
        Calculates predicted scoring rates per-match from a trained simple model.
        returns in the form (home, away).
        '''
        home_score_rate_val = self.home_score_rate(self.alpha, self.gamma, self.eta)
        away_score_rate_val = self.away_score_rate(self.alpha, self.gamma, self.eta)
        return home_score_rate_val, away_score_rate_val



class ExtendedSeasonalPoisson(PoissonModel):
    def __init__(self, data, team_map):
        super().__init__(data, team_map)
        self.data = dutils.add_season_prop(data)
        self.fitdata = dutils.create_model_dataset_extended(self.data)

    def from_saved(self):
        alpha_flat = self.load_model('seasonal')
        self.alpha, self.gamma, self.eta = self.convert_model_params(alpha_flat)
        return self

    def convert_model_params(self, alpha_flat):
        alpha = (np.append(alpha_flat[:self.split_loc], self.fixed[0]),
                 np.append(alpha_flat[self.split_loc:-3], self.fixed[1]))
        gamma = alpha_flat[-3:-1]
        eta = alpha_flat[-1]
        return alpha, gamma, eta

    def nll(self,
            all_params: (np.ndarray, np.ndarray),
            *data: (np.ndarray, np.ndarray, np.ndarray))->float:
        '''
        Negative log likelihood function of a extended Poisson goal model with a seasonal component.

        Params:
        data: [home_goals, away_goals]
        all_params: All floating model parameters as [alpha, beta, gamma_fit_params, eta]
        fixed: fixed values of alpha and beta for a given team to ensure model is identifiable

        Returns:
        nll: negative log likelihood osimple poisson model fit to data
        '''
        prop = all_params[-3:-1]
        eta = all_params[-1]
        p_params = (np.append(all_params[:self.split_loc], self.fixed[0]),
                    np.append(all_params[self.split_loc:-3], self.fixed[1]))
        gamma = prop[1]*data[2] + prop[0]

        nll_h = - poisson.logpmf(data[0], self.home_score_rate(p_params, gamma, eta)).sum()
        nll_a = - poisson.logpmf(data[1], self.away_score_rate(p_params, gamma, eta)).sum()

        return nll_h + nll_a

    def fit(self,
            save: bool = False)->np.ndarray:
        '''
        Minimizes nll for bi-poisson model, the 0th tuple value always describes
        the home team.

        Params:
        data: [home_goals, away_goals]
        gamma, eta: initial values for global parameters

        Returns:
        alpha: per-team parameters
        gamma, eta: global parameters
        nll: fit result negative log likelihood
        '''
        # Added seasonality in total goals via a linear fit
        init_params = np.append(self.init_params, [1, 1, self.eta])
        cons = ({'type':'eq', 'fun': lambda x: np.sum(x)})
        result = minimize(self.nll, init_params, args=(self.fitdata),
                     method='SLSQP', options={'maxiter': 30000})

        print(result.message)
        alpha_flat = result.x
        if save:
            self.save_model(alpha_flat, 'seasonal')
        alpha, gamma, eta = self.convert_model_params(alpha_flat)
        nll = result.fun
        print(f'Fitter negative log likelihood: {nll}')
        return alpha, gamma, eta, nll

    def scoring_rates(self)->(np.ndarray, np.ndarray):
        '''
        Calculates predicted scoring rates per-match from a trained extended model.
        returns in the form (home, away).
        '''
        # match_gamma is the value of gamma at a given point through the season
        match_gamma = self.gamma[1]*self.fitdata[2] + \
                      self.gamma[0]
        home_score_rate_val = self.home_score_rate(self.alpha, match_gamma, self.eta)
        away_score_rate_val = self.away_score_rate(self.alpha, match_gamma, self.eta)
        return home_score_rate_val, away_score_rate_val


