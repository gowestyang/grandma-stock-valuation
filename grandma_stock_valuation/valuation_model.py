"""
Grandma Stock Valuation (GSV) Model.

"""

from typing import Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from os import mkdir, path
from datetime import date


DEFAULT_OUTPUT_FOLDER = '_output'


def _printLevel(*args, level=logging.INFO):
    """
    A wrapper over `print` to support `level` argument.
    """
    print(*args)


class GrandmaStockValuation():
    """
    Class of regression based Grandma Stock Valuation model.
    """

    def __init__(self, recent_months=0, train_years=10, min_train_years=5, date_end=None, verbose=0, printfunc=_printLevel) -> None:
        """
        Initialize the Grandma Stock Valuation model.

        Parameters
        ----------
        recent_months : int
            Number of recent months, before `date_end`, to exclude from model fitting.
        train_years : int
            Maximum years of historical data, after excluding `recent_months`, for model fitting.
        min_train_years : int
            Minimum years of historical data required for model fitting.
        date_end : str ("yyyy-mm-dd") | date | None
            The "current" date. Data after this date will not be used.
            If None, use the latest date in the input data.
        verbose : int
            2 to print detailed information; 1 to print high-level information; 0 to suppress print.
        printfunc : func
            Function to output messages, which should support the `level` argument.
        """
        self.recent_months = recent_months
        self.train_years = train_years
        self.min_train_years = min_train_years
        self.date_end = date_end
        self.verbose = verbose
        self.printfunc = printfunc

        self._df_train = None
        self._df_recent = None
        self._fitted_model = None

        self._r2_train = np.nan
        self._train_years = np.nan
        self._annualized_return = np.nan
        self._current_price = np.nan
        self._fair_price = np.nan
        self._over_value_range = np.nan
        self._over_value_years = np.nan


    def _splitTrainRecent(self, input_data, price_col) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the input data into train and recent data sets.

        The model will be fitted only on the train data set, and trend will be estimated for both train and recent data sets.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Input data. It needs to contain a `date` column and a price column.
        price_col : str
            The column name in `input_data` to indicate price.
        
        Returns
        -------
        pandas.DataFrame
            Train data set.
        pandas.DataFrame
            Recent data set.
        """
        df_train0, df_recent0 = pd.DataFrame(columns=['date','price']), pd.DataFrame(columns=['date','price'])

        if len(input_data) <= 1:
            if self.verbose > 1: self.printfunc(f"Input data contains len{input_data} rows - not enough to train model.")
            return df_train0, df_recent0

        df0 = input_data.copy()
        df0['date'] = pd.to_datetime(df0['date'])
        df0 = df0[df0[price_col]>0].sort_values('date').reset_index(drop=True)
        if len(df0) <= 1:
            if self.verbose > 1: self.printfunc(f"Input data contains {len(df0)} valid price - not enough to train model.")
            return df_train0, df_recent0
        
        if self.date_end is None:
            date_recent_end = df0['date'].max()
        else:
            date_recent_end = min(pd.to_datetime(self.date_end), df0['date'].max())
        date_recent_end = date_recent_end.floor('D') + pd.DateOffset(days=1)
        date_recent_start = date_recent_end - pd.DateOffset(months=self.recent_months)
        date_train_end = date_recent_start

        date_train_start_needed = date_train_end - pd.DateOffset(years=self.min_train_years)
        date_first = df0['date'].min()
        if date_first >= date_train_start_needed:
            if self.verbose > 1: self.printfunc(f"Not enough training data to fit the model.")
            return df_train0, df_recent0

        date_train_start = date_train_end - pd.DateOffset(years=self.train_years)
        date_train_start = max(date_train_start, date_first)

        cols_select = ['date', price_col]
        cols_map = {price_col:'price'}

        df_train0 = df0[(df0['date']>=date_train_start) & (df0['date']<date_train_end)][cols_select].reset_index(drop=True).rename(columns=cols_map)
        if self.verbose > 1: self.printfunc(f"Train data contains {len(df_train0)} rows over {df_train0['date'].nunique()} dates from {df_train0['date'].min().date()} to {df_train0['date'].max().date()}.")

        df_recent0 = df0[(df0['date']>=date_recent_start) & (df0['date']<date_recent_end)][cols_select].reset_index(drop=True).rename(columns=cols_map)
        if len(df_recent0) > 0:
            if self.verbose > 1: self.printfunc(f"Recent data contains {len(df_recent0)} rows over {df_recent0['date'].nunique()} dates from {df_recent0['date'].min().date()} to {df_recent0['date'].max().date()}.")
        else:
            if self.verbose > 1: self.printfunc(f"No recent data specified.")
        
        return df_train0, df_recent0


    def fitTransform(self, input_data, price_col='close_adj', log=True, n_std=1.5):
        """
        Fit model, identify outliers, and estimate trend.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Daily price data of the insturment.
            It should contain a `date` column and a price column named by `price_col`.
        price_col : str
            The column name in `input_data` to indicate daily price.
            Suggest to use the adjusted price.
        log : bool
            If True, fit log-linear regression. If False, fit linear regression.
        n_std : float
            Outliers are identified by as examples with residual outside `mean ± n_std * std`.
            
        Returns
        -------
        GrandmaStockValuation
            The fitted model.
        """
        df_train, df_recent = self._splitTrainRecent(input_data, price_col)

        if len(df_train) > 1:
            if self.verbose > 1: self.printfunc("Fit regression...")
            df_train['x'] = range(len(df_train))
            x_train_max = df_train['x'].max()
            x_train = np.array(df_train['x']).reshape(-1, 1)

            y_train = np.log(df_train['price']) if log else df_train['price']

            lm = LinearRegression().fit(x_train, y_train)

            y_pred = lm.predict(x_train)
            df_train['trend'] = np.exp(y_pred) if log else y_pred
            df_train['residual'] = y_pred - y_train

            residual_std = df_train['residual'].std()
            residual_mean = df_train['residual'].mean()
            upper_bond = residual_mean + n_std * residual_std
            lower_bond = residual_mean - n_std * residual_std 

            df_train['is_outlier'] = (df_train['residual'] > upper_bond) | (df_train['residual'] < lower_bond)
            if self.verbose > 1: self.printfunc(f"{df_train['is_outlier'].sum()} out of {len(df_train)} dates are outliers.")

            if self.verbose > 1: self.printfunc("Re-fit wihtout outliers...")
            index_select = ~df_train['is_outlier']
            x_train_filter = np.array(df_train[index_select]['x']).reshape(-1, 1)
            y_train_filter = np.log(df_train[index_select]['price']) if log else df_train[index_select]['price']
            lm = LinearRegression().fit(x_train_filter, y_train_filter)
            y_pred = lm.predict(x_train)
            df_train['trend'] = np.exp(y_pred) if log else y_pred

            df_train['is_recent'] = False
            df_train.drop(columns=['residual'], inplace=True)

            if len(df_recent) > 0:
                if self.verbose > 1: self.printfunc("Extend trend to recent data.")
                df_recent['x'] = np.arange(0, len(df_recent)) + x_train_max + 1
                x_recent = np.array(df_recent['x']).reshape(-1, 1)
                y_recent = lm.predict(x_recent)
                df_recent['trend'] = np.exp(y_recent) if log else y_recent
                df_recent['is_outlier'] = False
                df_recent['is_recent'] = True
            else:
                if self.verbose > 1: self.printfunc("No recent data to estimate.")
            
            self._fitted_model = lm
            
        self._df_train, self._df_recent = df_train, df_recent
        if self.verbose > 1: self.printfunc("Complete model fitting!")

        return self


    def evaluateValuation(self, value_by_years=True) -> dict:
        """
        Evaluate valuation metrics of the fitted data sets with the estimated trend.

        This function should be executed after `fitTransform()`.

        Parameters
        ----------
        value_by_years : bool
            If True, also calculate `over_value_years`. See explaination below.
            
        Returns
        -------
        dict
            Valuation metrics:
                `R2_train`: R2 of the fitted model on train data, with outliers removed.
                `train_years`: number of years actually used to fit the model.
                `annualized_return`: average annualized return derived from the fitted trend.
                `current_price`: most recent price in the data.
                `fair_price`: most recent estimated price in the data, based on the fitted trend.
                `over_value_range`: `(current_price / fair_price) - 1`
                `over_value_years`: only calculated with `value_by_years=True`.
                    If *over_value_range >= 0*:
                        If *annualized_return > 0*: use `over_value_range / annualized_return` to indicate number of years over-valued.
                        If *annualized_return <= 0*: nan.
                    If *over_value_range < 0*:
                        If *annualized_return >= 1%*: use `over_value_range * annualized_return * 100`.
                        If *annualized_return within ± 1%*: fix at `over_value_range * 1% * 100`
                        If *annualized_return <= 1%*: use `over_value_range / annualized_return / 100`.
                    Note that by these formulations, when *over_value_range* is 0, *over_value_years* is 0 regardless of any positive *annualized_return*.
        """
        df_train, df_recent = self._df_train.copy(), self._df_recent.copy()

        if len(df_train) > 1:
            df_train_filter = df_train[~df_train['is_outlier']][['price','trend']]
            self._r2_train = 1 - ((df_train_filter['price'] - df_train_filter['trend'])**2).sum() / ((df_train_filter['price'] - df_train_filter['price'].mean())**2).sum()

            date_train_start = df_train['date'].min()
            date_train_end = df_train['date'].max()
            self._train_years = (date_train_end - date_train_start).days / 365
            trend_train_start = df_train['trend'].iloc[0]
            trend_train_end = df_train['trend'].iloc[-1]
            self._annualized_return = (trend_train_end / trend_train_start)**(1/self._train_years) - 1

            df_combine = pd.concat([df_train, df_recent]).reset_index(drop=True)
            self._current_price = df_combine['price'].iloc[-1]
            self._fair_price = df_combine['trend'].iloc[-1]
            self._over_value_range = self._current_price / self._fair_price - 1

            if value_by_years:
                if self._over_value_range >= 0:
                    if self._annualized_return > 0:
                        self._over_value_years = self._over_value_range / self._annualized_return
                    else:
                        self._over_value_years = np.nan
                else:
                    if self._annualized_return >= 0.01:
                        self._over_value_years = self._over_value_range * self._annualized_return * 100
                    elif self._annualized_return > -0.01:
                        self._over_value_years = self._over_value_range * 0.01 * 100
                    else:
                        self._over_value_years = self._over_value_range / self._annualized_return / 100
            else:
                self._over_value_years = np.nan

            d_metric = {
                'r2_train':self._r2_train,
                'train_years':self._train_years,
                'annualized_return':self._annualized_return,
                'current_price':self._current_price,
                'fair_price':self._fair_price,
                'over_value_range':self._over_value_range,
                'over_value_years':self._over_value_years
                }

            if self.verbose > 1:
                self.printfunc(f"R2 train = {self._r2_train:.3}, train years = {self._train_years:.3}, annualize return = {self._annualized_return:.3}.")
                self.printfunc(f"current price = {self._current_price:.3}, fair price = {self._fair_price:.3}, over-value range = {self._over_value_range:.3}, over-value years = {self._over_value_years:.3}.")

        else:
            self._r2_train = np.nan
            self._train_years = np.nan
            self._annualized_return = np.nan
            self._current_price = np.nan
            self._fair_price = np.nan
            self._over_value_range = np.nan
            self._over_value_years = np.nan

        d_metric = {
            'r2_train':self._r2_train,
            'train_years':self._train_years,
            'annualized_return':self._annualized_return,
            'current_price':self._current_price,
            'fair_price':self._fair_price,
            'over_value_range':self._over_value_range,
            'over_value_years':self._over_value_years
            }

        return d_metric


    def plotTrendline(self, title='Price and Trend', **kwargs):
        """
        Plot the data with outliers and fitted trend.

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
        df_train, df_recent = self._df_train.copy(), self._df_recent.copy()
        fig = go.Figure()

        if len(df_train) > 1:

            fig.add_trace(go.Scatter(x=df_train['date'], y=df_train['price'], name='Historic Price',
                                    line=dict(color='palegreen', width=1)))

            df_outlier = df_train[['date','price']].copy()
            index_outlier = df_train['is_outlier']
            df_outlier.loc[~index_outlier, 'price'] = None
            fig.add_trace(go.Scatter(x=df_outlier['date'], y=df_outlier['price'], name='Outlier',
                                    line=dict(color='red', width=1)))

            if len(df_recent) > 0:
                fig.add_trace(go.Scatter(x=df_recent['date'], y=df_recent['price'], name='Recent Price',
                                        line=dict(color='cyan', width=1)))

            df_trend = pd.concat([df_train, df_recent])[['date','trend']].reset_index(drop=True)
            fig.add_trace(go.Scatter(x=df_trend['date'], y=df_trend['trend'], name='Trend',
                                    line=dict(color='lightsalmon', width=1)))

            fig.update_layout(template='plotly_dark', title=title, xaxis_title='date', yaxis_title='price', **kwargs)
        
        return fig


def batchValuation(
    d_instrument_data,
    init_parameters={'recent_months':0, 'train_years':10, 'min_train_years':5, 'date_end':None},
    fit_parameters={'price_col':'close_adj', 'log':True, 'n_std':1.5},
    valuate_parameters={'value_by_years':True},
    draw_figure=True,
    save_result=True,
    metric_file = None,
    figure_folder = None,
    verbose=0,
    printfunc=_printLevel,
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
    valuate_parameters : dict
        Parameters passed to `GrandmaStockValuation.evaluateValuation()`.
    draw_figure : bool
        If True, generate price chart with trend.
    save_result : bool
        If True, save the valuation metrics and figures to files.
    metric_file : str
        File to store the valuation metrics.
        If `None`, save to the default location "_output/valuation_metrics_<today>.csv".
    figure_folder : str
        Folder to store the price charts of each instruments.
        If `None`, save to the default folder "_output/images/"
    verbose : int
        2 to print detailed information; 1 to print high-level information; 0 to suppress print.
    printfunc : func
        Function to output messages, which should support the `level` argument.
    **kwargs
        Additional key-word arguments passed to `GrandmaStockValuation.plotTrendline()`.

    Returns
    -------
    pandas.DataFrame
        A table containing valuation, which is the output of `GrandmaStockValuation.evaluateValuation()`, of each instrument.
    dict (str : figure)
        A dictionary containing the price chart, which is the output of `GrandmaStockValuation.plotTrendline()`, of each instrument.
        The keys are the tickers, and values are the figures.
    """
    if metric_file is None:
        metric_file = path.join(DEFAULT_OUTPUT_FOLDER, f'valuation_metrics_{date.today()}.csv')
        _createDefaultOutputFolder()
    
    if figure_folder is None:
        figure_folder = path.join(DEFAULT_OUTPUT_FOLDER, 'images')
        _createDefaultOutputFolder()

    l_metrics = []
    d_fig = {}
    for ticker, df in d_instrument_data.items():

        if verbose > 0: printfunc(f"Valuating {ticker}...")
        grandma = GrandmaStockValuation(verbose=verbose, printfunc=printfunc, **init_parameters)
        grandma.fitTransform(df, **fit_parameters)
        d_metrics = grandma.evaluateValuation(**valuate_parameters)
        df_metrics = pd.Series({'ticker':ticker, **d_metrics}).to_frame().T
        l_metrics.append(df_metrics)

        if draw_figure:
            fig = grandma.plotTrendline(title=ticker, **kwargs)
            d_fig[ticker] = fig
            if save_result:
                fig.write_image(path.join(figure_folder, f'{ticker}.jpeg'))
    
    df_metrics = pd.concat(l_metrics, ignore_index=True)
    if save_result: df_metrics.to_csv(metric_file, index=False)

    return df_metrics, d_fig


def _createDefaultOutputFolder():
    """
    Create the default output folders if not existed.

    """
    if not path.exists(DEFAULT_OUTPUT_FOLDER):
        mkdir(DEFAULT_OUTPUT_FOLDER)

    image_folder = path.join(DEFAULT_OUTPUT_FOLDER, 'images')
    if not path.exists(image_folder):
        mkdir(image_folder)


def addCashPortfolio(df_valuation_metrics, id_col='ticker', cash_name='cash',
                     growth_col='annualized_return', growth_value=0.0,
                     value_col='over_value_years', cash_value=0.0) -> pd.DataFrame:
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
        The updated valuation metrics with an additonal row indicating cahs.
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
