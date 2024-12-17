import numpy as np
import pandas as pd
import argparse
import datetime as dt

import utils.data_utils as dutils
import models
import fitting
import forecast
import plotting


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
    fitset, simset, team_map = dutils.create_datasets(dt.date(2024, 6, 6))
    if args.plot:
        plotting.gaussian_plot(fitset)
        plotting.plot_seasonality(fitset)
    if not args.forecast:
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
            simset = forecast.compute_win_loss_draw_prob(simset, 'simple', team_map)
        elif args.extended:
            simset = forecast.compute_win_loss_draw_prob(simset, 'seasonal', team_map)
        else:
            raise IOError('Model type must be specified with forecast flag')
        forecast.predict_league_table(simset)
        if args.evaluate:
            forecast.evaluate_pred(simset)
    if args.simulate or args.all:
        print('')
        print('# Simulation:')
        print('')
        raise NotImplementedError('Model simulations not yet implemented')


if __name__ == '__main__':
    main()

