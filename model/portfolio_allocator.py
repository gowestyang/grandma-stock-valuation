"""
Construct a portfolio based on valuation of a group of instruments.
"""

import numpy as np
import pandas as pd

def allocatePortfolio(valuations, transformation='exponential', scale=None) -> pd.Series:
    """
    Determine the portfolio allocation based on valuations of a group of instruments.

    Parameters
    ----------
    valuations : array like object
        Valuations of a group of instruments. The valuation should be a representation of "% over-valued", "years over-valued", etc.
    transformation : str
        "exponential" or "sigmoid".
        "exponential" is suggested, because it does not publish over-valued instruments too much, and heavily weights significantly under-valued instruments.
    scale : float
        Larger value gives more weight to more under-valued instruments. Should be a value greated or equal to 1.
        If not provided, will compensate number of instruments `n` as `scale = 2 - 2/n`
    
    Returns
    -------
    pandas.Series
        The suggested portfolio allocation.
    """
    assert transformation in ['exponential', 'sigmoid'], "transformation must be 'exponential' or 'sigmoid'."
    se_valuation = pd.Series(valuations)

    if scale is None:
        n_instruments = len(se_valuation)
        if n_instruments > 1:
            scale = 2 - 2 / n_instruments
        else:
            scale = 1
    assert scale >= 1, "scale should be greater or equal to 1."

    if transformation=='exponential':
        se_transformed = np.exp(- scale * se_valuation)
    if transformation=='sigmoid':
        se_transformed = 1 / (1 + np.exp(scale * se_valuation))

    se_portfolio = se_transformed / se_transformed.sum()

    return se_portfolio
