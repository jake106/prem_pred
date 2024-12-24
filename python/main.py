import numpy as np
import pandas as pd
import argparse
import datetime as dt

import utils.data_utils as dutils
import models
import fitting
import prediction
import plotting
import examples


def get_args()->argparse.Namespace:
    '''Retrieve command line arguments.'''
    parser = argparse.ArgumentParser(description = 'Run all models.')
    parser.add_argument('--plot', action = 'store_true',
        help = 'Perform some EDA - plot a few features of the dataset.')
    parser.add_argument('--simple', action = 'store_true',
        help = 'Just train and evaluate the simple model.')
    parser.add_argument('--extended', action = 'store_true',
        help = 'Just train and evaluate the model with seasonal extensions.')
    parser.add_argument('--forecast', action = 'store_true',
        help = '''Forecast the results of a simulation set of data using pre-trained models.
                  Model type must be specified with forecast flag''')
    parser.add_argument('--simulate', action = 'store_true',
        help = 'Simulate a league table using pre-trained models.')
    parser.add_argument('--all', action='store_true',
        help = 'Run entire sequence.')
    parser.add_argument('--evaluate', action = 'store_true',
        help = 'When actual results of prediction dataset available, run evaluation')
    args = parser.parse_args()
    return args


def main():
    '''A number of simple models to predict the results of football matches.'''
    args = get_args()
    print(args)
    fitset, simset, team_map = dutils.create_datasets(dt.date(2024, 12, 17))
    if args.plot:
        plotting.gaussian_plot(fitset)
        plotting.plot_seasonality(fitset)
    if not args.forecast and not args.simulate:
        if args.simple or args.all:
            print('')
            print('# Simple Model:')
            print('')
            fitting.simple_model(fitset, team_map)
        if args.extended or args.all:
            print('')
            print('# Extended Model:')
            print('')
            fitting.extended_model(fitset, team_map)
    if args.forecast or args.all:
        print('')
        print('# Forecasting:')
        print('')
        if args.simple:
            simset = prediction.compute_win_loss_draw_prob(simset, 'simple', team_map)
        elif args.extended:
            simset = prediction.compute_win_loss_draw_prob(simset, 'seasonal', team_map)
        else:
            raise IOError('Model type must be specified with forecast flag')
        print('Projected league table:')
        print(prediction.predict_league_table(simset, fitset))
    if args.evaluate:
        if not args.all and not args.forecast:
            raise IOError('Forecast flag must be turned on for evaluate')
        else:
            prediction.evaluate_pred(simset)
    if args.simulate or args.all:
        print('')
        print('# Simulation:')
        print('')
        N_sim = 500
        if args.simple:
            sim_tables = prediction.simulate_league(simset, fitset, 'simple', team_map, N_sim)
        elif args.extended:
            sim_tables = prediction.simulate_league(simset, fitset, 'seasonal', team_map, N_sim)
        else:
            raise IOError('Model type must be specified with simulate flag')
        # Here we demonstrate the probability of Aston Villa finishing in the top 5 of the 
        # Premier League using the simulated league tables
        examples.aston_villa(sim_tables)



if __name__ == '__main__':
    main()

