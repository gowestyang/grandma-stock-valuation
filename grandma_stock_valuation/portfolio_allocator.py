"""
Construct a portfolio based on valuation of a group of instruments.
"""

import numpy as np
import pandas as pd
from logging import WARNING
from itertools import permutations
from . import grandma_base

LOGPRINT = grandma_base.logger.logPandas

def getCorrelationWeight(
    d_instrument_prices,
    price_col='close',
    is_positive=False,
    is_sorted=False,
    recent_months=0,
    train_years=10,
    date_end=None,
    with_cash=False,
    cash_name='cash',
    verbose=0
) -> dict:
    """
    Caculate weight of each instrument based on correlation with other instruments.

    Derive weight of each instrument from pair-wise correlation among the instruments.
    More weight will be assign to instrument less correlated to others.

    Parameters
    ----------
    d_instrument_prices : dict of {str : pandas.DataFrame}
        Dictionary with the daily price data of each instrument.
        The keys are the tickers of the instruments, and the values are dataframes with the daily price.
        The dataframe should contain a `date` column (datetime64) and a price column named by `price_col` (float).
    price_col : str
        The column name in `input_data` to indicate daily price.
    is_positive : bool
        Indicate whether the input data has all positive price values.
        If False, the script will filter out non-positive price values.
    is_sorted : bool
        Indicate whether the input data has already been sorted by date.
        If False, the script will do the sorting.
    recent_months : int
        Number of recent months, before `date_end`, to exclude from correlation computation.
    train_years : int
        Years of historical data, after excluding `recent_months`, for correlation computation.
    date_end : str ("yyyy-mm-dd") | date | None
        The "current" date. Data after this date will not be used.
    with_cash : bool
        If True, will allocate fix weight `1 / (n_non_cash_instrument + 1)` to cash, 
        and the remaining to the non-cash instruments.
    cash_name : str
        Name of cash to be presented as key in the output dictionary.
    verbose : int
        2 to print detailed information; 1 to print high-level information; 0 to suppress print.

    Returns
    -------
    dict of {str : float}
        Weight allocated to each instrument (ticker).
    """
    if len(d_instrument_prices) == 0:
        if with_cash:
            return {cash_name:1}
        else:
            return {}

    if len(d_instrument_prices) == 1:
        if with_cash:
            return {**{t:0.5 for t in d_instrument_prices}, **{cash_name:0.5}}
        else:
            return {t:1 for t in d_instrument_prices}

    date_fit_end =  max(map(lambda df:df['date'].max(), d_instrument_prices.values()))
    if date_end is not None:
        date_fit_end = min(pd.to_datetime(date_end), date_fit_end)
    date_fit_end = date_fit_end.floor('D') + pd.DateOffset(days=1) - pd.DateOffset(months=recent_months)
    date_fit_start = date_fit_end - pd.DateOffset(years=train_years)
    
    d_cleaned_prices = {}
    for ticker, df in d_instrument_prices.items():
        df = df[['date', price_col]].copy()

        if not is_positive:
            df = df[df[price_col]>0].reset_index(drop=True)
        if not is_sorted:
            df.sort_values('date', inplace=True, ignore_index=True)
        df = df[(df['date']>=date_fit_start) & (df['date']<date_fit_end)].reset_index(drop=True)

        if len(df)==0:
            if verbose > 0: LOGPRINT(f"{ticker} has no data in the specified period!", level=WARNING)
        else:
            if verbose > 1: LOGPRINT(f"{ticker}: Selected {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
            d_cleaned_prices[ticker] = df

    d_cor_detail = {t:{} for t in d_cleaned_prices}
    for ticker1, ticker2 in permutations(d_cleaned_prices, 2):
        df1 = d_cleaned_prices[ticker1].rename(columns={price_col:'price1'})
        df2 = d_cleaned_prices[ticker2].rename(columns={price_col:'price2'})
        df = df1.merge(df2, 'inner', 'date', copy=False)
        if len(df)>1:
            cor = df['price1'].corr(df['price2'], method='pearson')
        else:
            cor = 1

        d_cor_detail[ticker1][ticker2] = cor
        d_cor_detail[ticker2][ticker1] = cor

    d_cor_agg = {t:sum([1-v for v in d_cor.values()]) for t, d_cor in d_cor_detail.items()}

    total_cor = sum(d_cor_agg.values())
    non_cash_weight = len(d_cor_agg) / (len(d_cor_agg)+1) if with_cash else 1.0
    d_weight = {t : non_cash_weight*v/total_cor for t, v in d_cor_agg.items()}

    if with_cash:
        d_weight[cash_name] = 1 / (len(d_cor_agg)+1)
    
    return d_weight


def allocatePortfolio(valuations, transformation='exponential', scale=None, center=-0.07, lower_bound=-0.2, weights=None) -> np.array:
    """
    Determine the portfolio allocation based on valuations of a group of instruments.

    Parameters
    ----------
    valuations : single-dimensional array like object
        Valuation scores of a group of `n` instruments.
    transformation : str
        Possible values are 'sigmoid', 'linear' or 'exponential'.
    scale : float
        Larger `scale` gives more weight to more under-valued instruments. Should be a positive value.
        if `None`, default based on `transformation`:
            'sigmoid' : default to 15
            'linear' : default to 3.5
            'exponential' : default to 3.2
            The above will result in weight close to 1.9 at -0.2 valuation.
    center : float
        At this value, `trasformation(center) = 1`
    lower_bound : float
        Under-value (negative valuation) will be capped by this parameter.
    weights : single-dimensional array like object
        Weight pre-allocated to each instrument, whose order should be aligned with the values in `valuations`.
    
    For a given valuation `v`,
        `exponential(v) = exp(- scale * (v - center))`
            Exponential transformation weights heavily (increasing slope) to very negative (under-valued) valuation.
            `lower_bound` should be used to prevent too much weight alloccated to one very under-valued instrument.
        `linear(v) = 1 / (scale * (v - center) + 1)` if `v > center`, else `1 - scale * (v - center)`
            Linear transformation applies linear weight (constant slop) to valuations.
        `sigmoid(v) = 2 / (1 + exp(scale * (v - center)))`
            Sigmoid places strong punishment to over-value.
            Sigmoid transformation bounds very negative or positive valuation (decreasing slop).
            `lower_bound` is not needed (can set to a big negative value).

    Returns
    -------
    numpy.array
        The suggested portfolio allocation.
    """
    assert transformation in ['exponential', 'linear', 'sigmoid'], "transformation must be 'exponential', 'linear' or 'sigmoid'."
    assert lower_bound < 0, "lower_bound should be a negative value."

    n_valuations = len(valuations)
    if n_valuations==0:
        return np.array([])

    ar_valuation = np.array(valuations).astype(float)
    ar_valuation = np.where(ar_valuation < lower_bound, lower_bound, ar_valuation)
    # if no valid valuation, distribute equally
    if np.isnan(ar_valuation).sum() == n_valuations:
        ar_valuation.fill(1 / n_valuations)
        
    if weights is not None:
        ar_weights = np.array(weights).astype(float)
        assert len(ar_weights) == n_valuations, "valuations and weights should be the same length and in the smae order."

        # if no valid weights, distribute equally
        if np.isnan(ar_weights).sum() == n_valuations:
            ar_weights.fill(1 / n_valuations)
        np.nan_to_num(ar_weights, copy=False, nan=0.0)
        assert (ar_weights < 0).sum() == 0, "weights shall be all non-negative values."
        assert ar_weights.sum() > 0, "Sum of weights shall be positive."
    else:
        ar_weights = np.full(n_valuations, 1/n_valuations)

    if scale is None:
        if transformation=='sigmoid':
            scale = 15
        if transformation=='linear':
            scale = 4.5
        else:
            scale = 3.2
    else:
        assert scale > 0, "scale should be a positive value."

    if transformation=='exponential':
        ar = - scale * (ar_valuation - center)
        ar = np.where(ar>100, 100, ar) # to prevent float overflow (up to 700)
        ar_transformed = np.exp(ar)
    if transformation=='sigmoid':
        ar = scale * (ar_valuation - center)
        ar = np.where(ar>100, 100, ar) # to prevent float overflow (up to 700)
        ar_transformed = 2 / (1 + np.exp(ar))
    if transformation=='linear':
        def f(x, scale=scale, center=center):
            if x > center:
                return 1 / (scale * (x - center) + 1)
            else:
                return  1 - scale * (x - center)
        ar_transformed = np.vectorize(f)(ar_valuation)
    
    ar_transformed = ar_transformed * ar_weights

    ar_portfolio = ar_transformed / np.nansum(ar_transformed)
    np.nan_to_num(ar_portfolio, copy=False, nan=0.0)

    return ar_portfolio
