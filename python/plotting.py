import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np

import utils.data_utils as dutils


def gaussian_plot(df: pd.DataFrame):
    '''
    Plot histogram of per-match home advantage normalised to a PDF.
    Overlays with Gaussian PDF for comparison.
    '''
    sample = dutils.add_ha_totg(df)['HA'].to_numpy()
    gauss = norm(np.mean(sample), np.std(sample))
    x = np.linspace(-6, 7, 300)
    plt.plot(x, gauss.pdf(x), label = 'Gaussian fit')
    _, bins, _ = plt.hist(sample, histtype = 'step', bins = 13, density = True,
                     range = (-6, 7), label = 'Normalised sample distribution')
    plt.xlabel('Home team goals scored - goals conceded', fontsize = 14)
    plt.ylabel(f'Probability density', fontsize=14)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'./plots/HA_gauss.pdf')
    plt.close()


def plot_seasonality(df: pd.DataFrame):
    '''
    Bins data by month and plots the mean, standard deviation, median, and
    interquartile range for home goals (FTHG), away goals (FTAG), total goals (TotG),
    and per-match home advantage (HA).
    '''
    sample = dutils.add_ha_totg(df)
    data_columns = ['FTHG', 'FTAG', 'TotG', 'HA']
    df = dutils.add_season_prop(df)

    # Set up binnning scheme here
    n_bins = 8
    bins = [i/n_bins for i in range(n_bins)]
    bin_centres = [bins[i] + (bins[i+1]-bins[i])/2 for i in range(n_bins-1)]
    width = bins[1] - bins[0]
    df['bins'] = pd.cut(df['Season_prop'], bins=bins)
    monthly = df.groupby(['bins'])

    # Need mean nd std to bin with std err as error estimate 
    monthly_std = monthly[data_columns].std()
    monthly_mean = monthly[data_columns].mean()
    monthly_std_err = monthly_std/np.sqrt(monthly.count())
    fig, axs = plt.subplots(figsize = (12, 12), ncols = 2, nrows = 2)
    kwargs = {'capsize': 3, 'ls':'none'}
    for ax, col in zip(axs.flatten(), data_columns):
        mean = monthly_mean[col].values
        std_err = monthly_std_err[col].values
        ax.errorbar(bin_centres, mean, yerr = std_err, fmt = 'x',
            label = 'Mean and Std Err', **kwargs)
        ax.plot(bin_centres, mean - std_err, marker='None',
                color='tab:red', label = '95% CI')
        ax.plot(bin_centres, mean + std_err, marker='None',
                color='tab:red')
        ax.set_xlabel('Prop. of way through a season', fontsize = 14)
        ax.set_ylabel(f'{col} / {width:.02f}', fontsize = 14)
    axs[0, 0].legend(fontsize = 14)
    fig.tight_layout()
    plt.savefig('./plots/check_seasonality.pdf')
    plt.close()


