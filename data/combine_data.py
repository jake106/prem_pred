import pandas as pd
import numpy as  np


def combine_datasets():
    '''
    Utility function to combine downloaded datasets into one ordered dataset for model training
    '''
    datasets = ['2021', '2022', '2023', '2024']
    dataframes = []
    for d in datasets:
        dataframes += [pd.read_csv(f'{d}.csv')]
    df = pd.DataFrame()
    for d in dataframes:
        df = pd.concat([df, d], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace = True)
    df.sort_index(inplace=True)
    df.to_csv('Premier_league_all.csv')


if __name__ == '__main__':
    combine_datasets()

