"""
Construct a portfolio based on valuation of a group of instruments.
"""

import numpy as np
import pandas as pd
import logging
from itertools import permutations


def _printLevel(*args, level=logging.INFO):
    """
    A wrapper over `print` to support `level` argument.
    """
    print(*args)


def getCorrelationWeight(
    d_instrument_prices,
    price_col='close_adj',
    recent_months=0,
    train_years=10,
    date_end=None,
    with_cash=False,
    cash_name='cash',
    verbose=2,
    printfunc=_printLevel
) -> dict:
    """
    Caculate weight of each instrument based on correlation with other instruments.

    Derive weight of each instrument from pair-wise correlation among the instruments.
    More weight will be assign to instrument less correlated to others.

    Parameters
    ----------
    d_instrument_prices : dict
        Dictionary with the daily price data of each instrument.
        The keys are the tickers of the instruments, and the values are dataframes with the daily price.
        The dataframe should contain a `date` column and a price column named by `price_col`.
    price_col : str
        The column name in `input_data` to indicate daily price.
        Suggest to use the adjusted price.
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
    printfunc : func
        Function to output messages, which should support the `level` argument.

    Returns
    -------
    dict[str : float]
        Weight allocated to each instrument (ticker).
    """
    if len(d_instrument_prices) == 0:
        d_weight = {}
        if with_cash:
            d_weight[cash_name] = 1
        return d_weight

    if len(d_instrument_prices) == 1:
        d_cor_agg = {t:1 for t in d_instrument_prices}

    else:

        d_cleaned_prices = {}
        for ticker, df in d_instrument_prices.items():
            df['date'] = pd.to_datetime(df['date'])
            df = df[df[price_col]>0].sort_values('date').reset_index(drop=True)

            if date_end is not None:
                date_fit_end = min(pd.to_datetime(date_end), df['date'].max())
            else:
                date_fit_end = df['date'].max()
            date_fit_end = date_fit_end.floor('D') + pd.DateOffset(days=1) - pd.DateOffset(months=recent_months)
            date_fit_start = date_fit_end - pd.DateOffset(years=train_years)
            df = df[(df['date']>=date_fit_start) & (df['date']<date_fit_end)][['date', price_col]].reset_index(drop=True)

            if len(df)==0:
                printfunc(f"{ticker} has no data in the specified period!", level=logging.WARNING)
            else:
                if verbose > 1: printfunc(f"{ticker}: Selected {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
                d_cleaned_prices[ticker] = df

        d_cor_detail = {t:{} for t in d_cleaned_prices}
        for ticker1, ticker2 in permutations(d_cleaned_prices, 2):
            df1 = d_cleaned_prices[ticker1].rename(columns={price_col:'price1'})
            df2 = d_cleaned_prices[ticker2].rename(columns={price_col:'price2'})
            df = df1.merge(df2, 'inner', 'date')
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


def allocatePortfolio(valuations, transformation='sigmoid', scale=20, with_cash=False, weights=None) -> np.array:
    """
    Determine the portfolio allocation based on valuations of a group of instruments.

    Parameters
    ----------
    valuations : single-dimensional array like object
        Valuations of a group of `n` instruments. The valuation should be a representation of "% over-valued", etc.
    transformation : str
        Possible values are "sigmoid" or "exponential".
        "sigmoid" is suggested, because it weights over-valued and under-valued instruments symmetrically.
        "exponential" trends to place too much weights on very-under-valued instruments.
    scale : float
        Larger `scale` gives more weight to more under-valued instruments. Should be a positive value.
        The default `scale=20` and `transformation=sigmoid` will yield:
            `valuation = -0.2, sigmoid(scale, valuation) = 0.98`
            `valuation = 0, sigmoid(scale, valuation) = 0.5`
            `valuation = 0.2, sigmoid(scale, valuation) = 0.02`
    with_cash : bool
        If True and `scale=None`, will compensate number of instruments (including cash) `n` as `scale = 2 - 2/(n-1)`.
        Use this configuration when one of the instrument is cash.
    weights : single-dimensional array like object
        Weight pre-allocated to each instrument, whose order should be aligned with the values in `valuations`.
    
    Returns
    -------
    numpy.array
        The suggested portfolio allocation.
    """
    assert transformation in ['exponential', 'sigmoid'], "transformation must be 'exponential' or 'sigmoid'."
    assert scale > 0, "scale should be a positive value."

    n_valuations = len(valuations)
    if n_valuations==0:
        return np.array([])

    ar_valuation = np.array(valuations).astype(float)
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
        ar_weights = np.array([1/n_valuations]*n_valuations)

    if transformation=='exponential':
        ar_transformed = np.exp(- scale * ar_valuation)
    if transformation=='sigmoid':
        ar_transformed = 1 / (1 + np.exp(scale * ar_valuation))
    
    ar_transformed = ar_transformed * ar_weights

    ar_portfolio = ar_transformed / np.nansum(ar_transformed)
    np.nan_to_num(ar_portfolio, copy=False, nan=0.0)

    return ar_portfolio


def _getGrowthWeight(annualized_return, growth_scale=1.0, min_pos_growth=0.01, min_neg_growth=-0.03):
    """
    Calculate linear weight based on annualized growth.

    This functionality is still under evaluating, because it places too much punishment to instrumentw with low growth,
    regardless of valuation. The result is almost identical to filter out instruments with low growth.

    For `annualized_return > min_pos_growth`, weight linearly by `annualized_return * growth_scale`.
    For `annualized_return` between `min_neg_growth` and `min_pos_growth`, weight linearly from
        `(min_neg_growth, 0)` to `(min_pos_growth, min_pos_growth*growth_scale)`.
    For `annualized_return <= min_pos_growth`, weight as 0.
    If weights of all instruments are 0, then distribute the weights evenly.

    Parameters
    ----------
    annualized_return : single-dimensional array like object
        Annualized growth(return) of each instrument.
    growth_scale : float
    min_pos_growth : float
    min_neg_growth : float

    Returns
    -------
    numpy.array
        The growth weight of each instrument.
    """
    if len(annualized_return) == 0:
        return np.array([])

    ar = np.array(annualized_return)
    ar = np.where(
        ar>= min_pos_growth,
        ar * growth_scale,
        np.where(
            ar >= min_neg_growth,
            min_pos_growth*growth_scale / (min_pos_growth-min_neg_growth) * (ar - min_neg_growth),
            0
        ))

    ar_sum = np.nansum(ar)
    if ar_sum == 0:
        ar.fill(1 / len(ar))
    else:
        ar = ar / ar_sum
    return ar
