"""
Utilities to load package data.
"""

from typing import Tuple
from os import listdir, path
import pandas as pd


def loadPacakgeData(verbose=0) -> Tuple[dict, dict]:
    """
    Load package data for examples and testing.

    Parameters
    ----------
    verbose : int
        2 to print detailed information; 1 to print high-level information; 0 to suppress print.

    Returns:
        dict[str : pandas.DataFrame]
            Loaded daily prices of the intruments.
            Keys are the tickers, and values are dataframes with daily prices.
        dict[str : str]
            Description of the instruments.
            Keys are the tickers, and values are the description.
    """
    PATH_DATA = path.join('..', 'data')

    files = listdir(PATH_DATA)

    d_instrument_data = {}
    for f in files:
        ticker = f[:f.find('_')]
        data = pd.read_csv(path.join(PATH_DATA, f))
        data['date'] = pd.to_datetime(data['date'])
        data = data[data['close_adj']>0].reset_index(drop=True)
        d_instrument_data['ticker'] = data
        if verbose>0: print(f"{ticker} data contains {len(data)} rows, {data['date'].nunique()} dates from {data['date'].min().date()} to {data['date'].max().date()}.")

    d_instrument = {
        'IVV':'SP500',
        'VPL':'Developed Asia-Pacific',
        'IEV':'Europe',
        'EEMA':'Emerging Asia',
    }

    return d_instrument_data, d_instrument
