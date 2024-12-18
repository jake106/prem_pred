import pandas as pd
import numpy as  np

'''
Temporary data handling files - when pulling data from API is set up and automated these will
all be changed and added to data_utils.
'''

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


def append_future_data():
    historical = pd.read_csv('Premier_league_all.csv')
    historical.Time = pd.to_datetime(historical['Time'], format='%H:%M').dt.time
    historical = historical[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                             'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF',
                             'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]
    future = pd.read_csv('2024-future.csv')
    future = future[['Date', 'Home Team', 'Away Team']]
    future['Dateidx'] = pd.to_datetime(future['Date'], dayfirst=True)
    future['Time'] = future['Dateidx'].dt.time
    future.set_index('Dateidx', inplace = True)
    future.sort_index(inplace=True)
    future.index = future.index.strftime('%Y-%m-%d')
    future['Date'] = future.index.values
    # Rename some columns
    future.rename(mapper={'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam'},
                  axis = 1, inplace = True)
    future['HomeTeam'].replace(['Man Utd', 'Spurs'], ["Man United", 'Tottenham'], inplace=True)
    future['AwayTeam'].replace(['Man Utd', 'Spurs'], ["Man United", 'Tottenham'], inplace=True)
    
    for f in future.HomeTeam.unique():
        if f not in historical.HomeTeam.unique():
            print(f)

    # Need to merge such that all historical data is kept and overwrites the future fixtures data
    future = future[future['Date'] > historical['Date'].max()]
    df = pd.merge(historical, future, on=['Date', 'Time', 'HomeTeam', 'AwayTeam'], how='outer')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv('data_prem_all.csv')


if __name__ == '__main__':
    combine_datasets()
    append_future_data()

