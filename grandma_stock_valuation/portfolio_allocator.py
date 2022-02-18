"""
Construct a portfolio based on valuation of a group of instruments.
"""

import numpy as np

def allocatePortfolio(valuations, transformation='exponential', scale=None, with_cash=False) -> np.array:
    """
    Determine the portfolio allocation based on valuations of a group of instruments.

    Parameters
    ----------
    valuations : single-dimensional array like object
        Valuations of a group of `n` instruments. The valuation should be a representation of "% over-valued", "years over-valued", etc.
    transformation : str
        "exponential" or "sigmoid".
        "exponential" is suggested, because it does not publish over-valued instruments too much, and heavily weights significantly under-valued instruments.
    scale : float
        Larger value gives more weight to more under-valued instruments. Should be a value greated or equal to 1.
        If not provided, will compensate number of instruments `n` as `scale = 2 - 2/n`.
    with_cash : bool
        If True and `scale=None`, will compensate number of instruments `n` as `scale = 2 - 2/(n-1)`.
        Use this configuration when one of the instrument is cash.
    
    Returns
    -------
    numpy.array
        The suggested portfolio allocation.
    """
    assert transformation in ['exponential', 'sigmoid'], "transformation must be 'exponential' or 'sigmoid'."
    ar_valuation = np.array(valuations).astype(float)

    if scale is None:
        n_instruments = len(ar_valuation) if not with_cash else len(ar_valuation)-1
        if n_instruments > 1:
            scale = 2 - 2 / n_instruments
        else:
            scale = 1
    assert scale >= 1, "scale should be greater or equal to 1."

    if transformation=='exponential':
        ar_transformed = np.exp(- scale * ar_valuation)
    if transformation=='sigmoid':
        ar_transformed = 1 / (1 + np.exp(scale * ar_valuation))

    ar_portfolio = ar_transformed / ar_transformed.sum()

    return ar_portfolio
