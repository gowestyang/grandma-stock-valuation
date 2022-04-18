"""
Grandma Stock Valuation (GSV) Model.
"""

from typing import Tuple
from os import mkdir, path
from datetime import date
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from . import grandma_base

LOGPRINT = grandma_base.logger.logPandas
DEFAULT_IMAGE_FOLDER = '_image'

class GrandmaStockValuation():
    """
    Class of regression based Grandma Stock Valuation model.
    """

    def __init__(self, recent_months=0, train_years=10, min_train_years=5, date_end=None, verbose=0) -> None:
        """
        Initialize the Grandma Stock Valuation model.

        Parameters
        ----------
        recent_months : int
            Number of recent months, before `date_end`, to be excluded from model fitting.
            Use this parameter if you have a strong view that recent period is very abnormal and should be excluded.
        train_years : int
            Maximum years of historical data, after excluding `recent_months`, used for model fitting.
        min_train_years : int
            Minimum years of historical data required for model fitting.
        date_end : str ("yyyy-mm-dd") | date | None
            The "current" date. Data after this date will not be used.
            If None, use the latest date in the input data.
        verbose : int
            2 to print detailed information; 1 to print high-level information; 0 to suppress print.
        """
        self.recent_months = recent_months
        self.train_years = train_years
        self.min_train_years = min_train_years
        self.date_end = date_end
        self.verbose = verbose

        self._df_train = None
        self._df_recent = None
        self._fitted_model = None

        self._r2_train = np.nan
        self._train_years = np.nan
        self._annualized_growth = np.nan
        self._current_price = np.nan
        self._fair_price = np.nan
        self._over_value_range = np.nan


    def _splitTrainRecent(self, input_data, price_col, is_positive, is_sorted) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the input data into train and recent data sets.

        The model will be fitted only on the train data set, and trend will be estimated for both train and recent data sets.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Input data. It needs to contain a `date` column and a price column.
        price_col : str
            The column name in `input_data` to indicate price.
        is_positive : bool
            Indicate whether the input data has all positive price values.
            If False, the script will filter out non-positive price values.
        is_sorted : bool
            Indicate whether the input data has already been sorted by date.
            If False, the script will do the sorting.

        Returns
        -------
        pandas.DataFrame
            Train data set.
        pandas.DataFrame
            Recent data set.
        """
        df_train0, df_recent0 = pd.DataFrame(columns=['date','price']), pd.DataFrame(columns=['date','price'])

        if len(input_data) <= 1:
            if self.verbose > 1: LOGPRINT(f"Input data contains len{input_data} rows - not enough to train model.")
            return df_train0, df_recent0

        df0 = input_data.copy()
        if not str(df0['date'].dtype).startswith('datetime'):
            df0['date'] = pd.to_datetime(df0['date'])
        if not is_positive:
            df0 = df0[df0[price_col]>0].reset_index(drop=True)
        if not is_sorted:
            df0.sort_values('date', ignore_index=True, inplace=True)
        if len(df0) <= 1:
            if self.verbose > 1: LOGPRINT(f"Input data contains {len(df0)} valid price - not enough to train model.")
            return df_train0, df_recent0
        
        if self.date_end is None:
            date_recent_end = df0['date'].max()
        else:
            date_recent_end = min(pd.to_datetime(self.date_end), df0['date'].max())
        date_recent_end = date_recent_end.floor('D') + pd.DateOffset(days=1)
        date_recent_start = date_recent_end - pd.DateOffset(months=self.recent_months)

        date_train_end = date_recent_start
        date_train_start_needed = date_train_end - pd.DateOffset(years=self.min_train_years)
        date_first = df0['date'].min().floor('D')
        if date_first > date_train_start_needed:
            if self.verbose > 1: LOGPRINT(f"Not enough training data to fit the model.")
            return df_train0, df_recent0

        date_train_start = date_train_end - pd.DateOffset(years=self.train_years)
        date_train_start = max(date_train_start, date_first)

        cols_select = ['date', price_col]
        cols_map = {price_col:'price'}

        index_train0 = (df0['date']>=date_train_start) & (df0['date']<date_train_end)
        df_train0 = df0[index_train0][cols_select].reset_index(drop=True).rename(columns=cols_map)
        if self.verbose > 1: LOGPRINT(f"Train data contains {len(df_train0)} rows over {df_train0['date'].nunique()} dates from {df_train0['date'].min().date()} to {df_train0['date'].max().date()}.")

        index_recent0 = (df0['date']>=date_recent_start) & (df0['date']<date_recent_end)
        df_recent0 = df0[index_recent0][cols_select].reset_index(drop=True).rename(columns=cols_map)
        if len(df_recent0) > 0:
            if self.verbose > 1: LOGPRINT(f"Recent data contains {len(df_recent0)} rows over {df_recent0['date'].nunique()} dates from {df_recent0['date'].min().date()} to {df_recent0['date'].max().date()}.")
        else:
            if self.verbose > 1: LOGPRINT(f"No recent data specified.")
        
        return df_train0, df_recent0


    def fitTransform(self, input_data, price_col, is_positive=False, is_sorted=False, log=True, n_std=1):
        """
        Fit model, identify outliers, and estimate trend.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Daily price data of the insturment.
            It should contain a `date` column and a price column named by `price_col`.
        price_col : str
            The column name in `input_data` to be used as daily price.
        is_positive : bool
            Indicate whether the input data has all positive price values.
            If False, the script will filter out non-positive price values.
        is_sorted : bool
            Indicate whether the input data has already been sorted by date.
            If False, the script will do the sorting.
        log : bool
            If True, fit log-linear regression. If False, fit linear regression.
        n_std : float
            Outliers are identified by as examples with residual outside `mean ± n_std * std`.
            
        Returns
        -------
        GrandmaStockValuation
            The fitted model.
        """
        df_train, df_recent = self._splitTrainRecent(input_data, price_col, is_positive, is_sorted)

        n_train = len(df_train)
        if n_train > 1:
            if self.verbose > 1: LOGPRINT("Fit regression...")
            x_train = np.arange(n_train)
            x_train_max = x_train[-1]
            x_train = x_train.reshape(-1, 1)
            y_train = df_train['price'].to_numpy()
            if log:
                y_train = np.log(y_train)

            # In this use case, it has similar training and predicting speed as scipy.stats.linregress
            lm = LinearRegression().fit(x_train, y_train)
            y_pred = lm.predict(x_train)

            residuals = y_pred - y_train
            residual_std, residual_mean = residuals.std(), residuals.mean()
            upper_bond = residual_mean + n_std * residual_std
            lower_bond = residual_mean - n_std * residual_std 

            non_outliers = (residuals > lower_bond) & (residuals < upper_bond)
            df_train['is_outlier'] = ~ non_outliers
            if self.verbose > 1: LOGPRINT(f"{df_train['is_outlier'].sum()} out of {n_train} dates are outliers.")

            if self.verbose > 1: LOGPRINT("Re-fit wihtout outliers...")

            x_train_filter, y_train_filter = x_train[non_outliers], y_train[non_outliers]
            lm = LinearRegression().fit(x_train_filter, y_train_filter)
            y_pred = lm.predict(x_train)
            df_train['trend'] = np.exp(y_pred) if log else y_pred

            df_train['is_recent'] = False

            n_recent = len(df_recent)
            if n_recent > 0:
                if self.verbose > 1: LOGPRINT("Extend trend to recent data.")
                x_recent = (np.arange(n_recent) + x_train_max + 1).reshape(-1, 1)
                y_recent = lm.predict(x_recent)
                df_recent['trend'] = np.exp(y_recent) if log else y_recent
                df_recent['is_outlier'] = False
                df_recent['is_recent'] = True
            else:
                if self.verbose > 1: LOGPRINT("No recent data to estimate.")
            
            self._fitted_model = lm
        else:
            if self.verbose > 1: LOGPRINT("Not enough training data to fit the model.")
            
        self._df_train, self._df_recent = df_train, df_recent
        if self.verbose > 1: LOGPRINT("Complete model fitting!")

        return self


    def evaluateValuation(self) -> dict:
        """
        Evaluate valuation metrics of the fitted data sets with the estimated trend.

        This function should be executed after `fitTransform()`.

        Parameters
        ----------
        (None)
            
        Returns
        -------
        dict of {str : float}
            Valuation metrics:
                `R2_train`: R2 of the fitted model on train data, with outliers removed.
                `train_years`: number of years actually used to fit the model.
                `annualized_growth`: average annualized growth derived from the fitted trend.
                `current_price`: most recent price in the data.
                `fair_price`: most recent estimated price in the data, based on the fitted trend.
                `over_value_range`: `(current_price / fair_price) - 1`
        """
        df_train, df_recent = self._df_train, self._df_recent

        if len(df_train) > 1:
            non_outlier = ~ df_train['is_outlier'].to_numpy()
            price_filter = df_train['price'].to_numpy()[non_outlier]
            trend_filter = df_train['trend'].to_numpy()[non_outlier]
            self._r2_train = 1 - 1 - ((price_filter - trend_filter)**2).sum() / ((price_filter - price_filter.mean())**2).sum()

            date_train_start = df_train['date'].iloc[0]
            date_train_end = df_train['date'].iloc[-1]
            self._train_years = (date_train_end - date_train_start).days / 365
            trend_train_start = df_train['trend'].iloc[0]
            trend_train_end = df_train['trend'].iloc[-1]
            self._annualized_growth = (trend_train_end / trend_train_start)**(1/self._train_years) - 1

            df_combine = pd.concat([df_train, df_recent], ignore_index=True, copy=False)
            self._current_price = df_combine['price'].iloc[-1]
            self._fair_price = df_combine['trend'].iloc[-1]
            self._over_value_range = self._current_price / self._fair_price - 1

            d_metric = {
                'r2_train':self._r2_train,
                'train_years':self._train_years,
                'annualized_growth':self._annualized_growth,
                'current_price':self._current_price,
                'fair_price':self._fair_price,
                'over_value_range':self._over_value_range
                }

            if self.verbose > 1:
                LOGPRINT(f"R2 train = {self._r2_train:.3}, train years = {self._train_years:.3}, annualize return = {self._annualized_growth:.3}.")
                LOGPRINT(f"current price = {self._current_price:.3}, fair price = {self._fair_price:.3}, over-value range = {self._over_value_range:.3}.")

        else:
            self._r2_train = np.nan
            self._train_years = np.nan
            self._annualized_growth = np.nan
            self._current_price = np.nan
            self._fair_price = np.nan
            self._over_value_range = np.nan

        d_metric = {
            'r2_train':self._r2_train,
            'train_years':self._train_years,
            'annualized_growth':self._annualized_growth,
            'current_price':self._current_price,
            'fair_price':self._fair_price,
            'over_value_range':self._over_value_range
            }

        return d_metric


    def plotTrendline(self, title='Price and Trend', **kwargs):
        """
        Plot the data with fitted trend, and outliers highlighted.

        This function needs to be executed after `fitTransform()`.

        Parameters
        ----------
        title : str
            Title of the plot.
        **kwargs
            Additional key-word arguments passed to plotly's `update_layout` function. 
            
        Returns
        -------
        Figure
            A plotly figure object of the plot.
        """
        fig = go.Figure()

        if len(self._df_train) > 1:
            ar_date_train = self._df_train['date'].to_numpy()
            ar_price_train = self._df_train['price'].to_numpy()
            ar_trend_train = self._df_train['trend'].to_numpy()
            fig.add_trace(go.Scatter(x=ar_date_train, y=ar_price_train, name='Historic Price',
                                     line=dict(color='palegreen', width=1)))

            is_outlier = self._df_train['is_outlier'].to_numpy()
            ar_price_outlier = np.where(is_outlier, ar_price_train, np.nan)
            fig.add_trace(go.Scatter(x=ar_date_train, y=ar_price_outlier, name='Outlier',
                                     line=dict(color='red', width=1)))

            if len(self._df_recent) > 0:
                ar_date_recent = self._df_recent['date'].to_numpy()
                ar_price_recent = self._df_recent['price'].to_numpy()
                ar_trend_recent = self._df_recent['trend'].to_numpy()
                fig.add_trace(go.Scatter(x=ar_date_recent, y=ar_price_recent, name='Recent Price',
                                         line=dict(color='cyan', width=1)))
                
                ar_date_total = np.concatenate((ar_date_train, ar_date_recent))
                ar_trend_total = np.concatenate((ar_trend_train, ar_trend_recent))
            else:
                ar_date_total = ar_date_train
                ar_trend_total = ar_trend_train

            fig.add_trace(go.Scatter(x=ar_date_total, y=ar_trend_total, name='Trend',
                                     line=dict(color='lightsalmon', width=1)))

            fig.update_layout(template='plotly_dark', title=title, xaxis_title='date', yaxis_title='price', **kwargs)
        
        return fig


def batchValuation(
    d_instrument_data,
    init_parameters=dict(recent_months=0, train_years=10, min_train_years=5, date_end=None),
    fit_parameters=dict(price_col='close', is_positive=False, is_sorted=False, log=True, n_std=1),
    draw_figure=False,
    save_result=False,
    figure_folder=None,
    verbose=0,
    **kwargs
) -> Tuple[pd.DataFrame, dict]:
    """
    Carry out valuation of a group of instruments, by fitting a model on each instrument.

    Parameters
    ----------
    d_instrument_data : dict (str : pandas.dataframe)
        A dictionary containing the daily price of a group of instruments.
        Each key should be a ticker, and its value should be the daily price data of the ticker.
    init_parameters : dict
        Parameters passed to `GrandmaStockValuation` at initialization.
    fit_parameters : dict
        Parameters passed to `GrandmaStockValuation.fitTransform()`.
    draw_figure : bool
        If True, generate price chart with trend of each instrument.
    save_result : bool
        If True, save the figures to files.
    figure_folder : str
        Folder to store the price charts of each instruments.
        If `None`, save to the default folder "_images/"
    verbose : int
        2 to print detailed information; 1 to print high-level information; 0 to suppress print.
    **kwargs
        Additional key-word arguments passed to `GrandmaStockValuation.plotTrendline()`.

    Returns
    -------
    pandas.DataFrame
        A table containing valuation, which is the output of `GrandmaStockValuation.evaluateValuation()` of each instrument.
    dict of {str : figure}
        A dictionary containing the price chart, which is the output of `GrandmaStockValuation.plotTrendline()`, of each instrument.
        The keys are the tickers, and values are the plotly figures.
    """
    l_metrics = []
    d_fig = {}
    for ticker, df in d_instrument_data.items():
        if verbose > 0: LOGPRINT(f"Valuating {ticker}...")
        grandma = GrandmaStockValuation(verbose=verbose, **init_parameters)
        grandma.fitTransform(df, **fit_parameters)
        d_metrics = grandma.evaluateValuation()
        l_metrics.append({'ticker':ticker, **d_metrics})

        if draw_figure:
            fig = grandma.plotTrendline(title=ticker, **kwargs)
            d_fig[ticker] = fig

            if save_result:
                if figure_folder is None:
                    figure_folder = DEFAULT_IMAGE_FOLDER
                if not path.exists(figure_folder):
                    mkdir(figure_folder)
                fig.write_image(path.join(figure_folder, f'{ticker}.jpeg'))
    
    df_metrics = pd.DataFrame(l_metrics)

    return df_metrics, d_fig


def addCashPortfolio(df_valuation_metrics, id_col='ticker', cash_name='cash',
                     growth_col=None, growth_value=0.0,
                     value_col='over_value_range', cash_value=0.0) -> pd.DataFrame:
    """
    Add cash into the valuation metrics of an existing portfolio.

    Parameters
    ----------
    df_valuation_metrics : pandas.DataFrame
        A dataframe with the valuation metrics of an existing portfolio.
        It is usually the output of `grandma_stock_valuation.batchValuation()` function.
    id_col: str
        Column in `df_valuation_metrics` as the identifier of the instruments in the portfolio.
    cash_name: str
        A name, such as "cash", to be appended to the `id_col`.
    growth_col: str or None
        Column in `df_valuation_metrics` with annualized return of each instruments in the portfolio.
        If None, will not be used.
    growth_value: float
        Annualized return of cash, such as 0.
    value_col: str
        Column in `df_valuation_metrics` with the valuations of each instruments in the portfolio.
    cash_value: float
        Valuation of cash, such as 0 (neither over-valued nor under-valued).

    Returns
    -------
    pandas.DataFrame
        The updated valuation metrics with an additonal row indicating cash.
    """
    d_cash = {id_col:cash_name, value_col:cash_value}
    if growth_col is not None:
        d_cash = {**d_cash, growth_col:growth_value}

    df_cash = pd.Series(d_cash).to_frame().T
    df_portfolio = pd.concat([df_valuation_metrics, df_cash], ignore_index=True, copy=False)

    df_portfolio[value_col] = df_portfolio[value_col].astype(float, copy=False)
    if growth_col is not None:
        df_portfolio[growth_col] = df_portfolio[growth_col].astype(float, copy=False)

    return df_portfolio


def subtractValueGrowth(over_value_range, annualized_growth, growth_scale=1):
    """
    Adjust `over_value_range` with `annualized_growth` by subtraction.

    Adjusted over-value score = `over_value_range - growth_scale * annualized_growth`

    Parameters
    ----------
    over_value_range : single-dimensional array like object
        Valuations of a group of instruments.
        It is usually the `over_value_range` column in the output of `grandma_stock_valuation.batchValuation()` function.
    annualized_growth : single-dimensional array like object
        Annualized growth of each instrument, whose order should be aligned with the values in `over_value_range`.
    growth_scale : float
        The scale used to adjust annualized growth before being subtracted.

    Returns
    -------
    numpy.array
        Over-value score adjusted by annualized growth of each instrument.
    """
    over_value_range = np.array(over_value_range)
    annualized_growth = np.array(annualized_growth)
    assert len(over_value_range) == len(annualized_growth), "valuations and growth should be the same length and in the smae order."

    return over_value_range - growth_scale * annualized_growth
    

def divideValueGrowth(over_value_range, annualized_growth, growth_scale=0.1):
    """
    Calculate `over_value_years` based on `over_value_range` and `annualized_growth`.

    Step 1:
        If `over_value_range >= 0`:
            If `annualized_growth > 0`: use `over_value_range / annualized_growth` to indicate number of years over-valued.
            If `annualized_growth <= 0`: nan.
        If `over_value_range < 0`:
            If `annualized_growth >= 1%`: use `over_value_range * annualized_growth * 100`.
            If `annualized_growth within ± 1%`: fix at `over_value_range * 1% * 100 = over_value_range`
            If `annualized_growth <= 1%`: use `over_value_range / annualized_growth / 100`.
    Step 2:
        Divide by 10 to keep similar magnitude as `over_value_range`.
    Step 3"
        Subtract `growth_scale * annualized_growth`. This is mainly to account for growth when over-value range is very close to 0.
        Because without this step, the result will always be 0 if over-value range is 0, regardless of growth.

    Parameters
    ----------
    over_value_range : single-dimensional array like object
        Valuations of a group of instruments.
        It is usually the `over_value_range` column in the output of `grandma_stock_valuation.batchValuation()` function.
    annualized_growth : single-dimensional array like object
        Annualized growth of each instrument, whose order should be aligned with the values in `over_value_range`.
    growth_scale : float
        The scale used to adjust annualized growth before being subtracted.
        Here the growth_scale is mainly for tie-breaker near 0 valuation.

    Returns
    -------
    numpy.array
        Over-value score adjusted by annualized growth of each instrument.
    """
    over_value_range = np.array(over_value_range)
    annualized_growth = np.array(annualized_growth)
    assert len(over_value_range) == len(annualized_growth), "valuations and growth should be the same length and in the smae order."

    def _calc_one_overvalue_year(t_values, growth_scale=growth_scale):
        _over_value_range, _annualized_growth = t_values
        if _over_value_range >= 0:
            if _annualized_growth > 0:
                score = _over_value_range / _annualized_growth
            else:
                return np.nan
        else:
            if _annualized_growth >= 0.01:
                score = _over_value_range * _annualized_growth * 100
            elif _annualized_growth > -0.01:
                score = _over_value_range
            else:
                score = _over_value_range / (- _annualized_growth) / 100
        
        score = score/10 - growth_scale*_annualized_growth

        return score

    return np.array(list(map(_calc_one_overvalue_year, zip(over_value_range, annualized_growth))))
