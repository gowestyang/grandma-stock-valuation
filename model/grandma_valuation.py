"""
Grandma Valuation Model.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

class GrandmaValuationModel():

    def __init__(self, data_input, recent_months=0, train_years=10, date_end=None, col_price='close', printfunc=print) -> None:
        """
        Grandma Valuation Model.

        Parameters
        ----------
        data_input : pandas.DataFrame
            Input data with "date" column
        recent_months : str ("yyyy-mm-dd") | date | None
            Start date of query. If None, will derive a start date from an existing data file.
        train_years : str ("yyyy-mm-dd") | date | None
            End date of query, which is exclusive - the last date returned will be smaller then it.
        date_end : str ("yyyy-mm-dd") | date | None
            End date of query, which is exclusive - the last date returned will be smaller then it.
        col_price : str ("yyyy-mm-dd") | date | None
            End date of query, which is exclusive - the last date returned will be smaller then it.
        printfunc : a function to output messages, which should handle `end` and `level` arguments.
        """
        self.data_input = data_input
        self.recent_months = recent_months
        self.train_years = train_years
        self.date_end = date_end
        self.col_price = col_price
        self.printfunc = printfunc

        self._df_train = None,
        self._df_recent = None,
        self._rmse_train = np.nan,
        self._train_years = np.nan,
        self._annualized_return = np.nan,
        self._currenct_price = np.nan,
        self._base_price = np.nan,
        self._over_value_range = np.nan,
        self._over_value_years = np.nan


    def _splitTrainRecent(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Set start and end dates, and load existing data file if needed.

        Parameters
        ----------
        name : str
            Name used for logging and default file name.
        load_file : bool
            If True, load data file stored at `file_name`
        file_name: str
            The file containing the existing data.
            
        Returns
        -------
        pandas.DataFrame
            The loaded data (if any).
        """
        df0 = self.data_input.copy()
        if self.date_end is None:
            date_recent_end = df0['date'].max()
        else:
            date_recent_end = min(self.date_end, date_max = df0['date'].max())
        date_recent_start = date_recent_end - pd.DateOffset(months=self.recent_months) + pd.DateOffset(days=1)
        date_train_end = date_recent_start - pd.DateOffset(days=1)
        date_train_start = date_train_end - pd.DateOffset(years=self.train_years) + pd.DateOffset(days=1)
        date_train_start = max(date_train_start, df0['date'].min())

        cols_select = ['date', self.col_price]
        cols_map = {self.col_price:'price'}

        df_train0 = df0[(df0['date']>=date_train_start) & (df0['date']<=date_train_end)][cols_select].reset_index(drop=True).rename(columns=cols_map)
        self.printfunc(f"Train data contains {len(df_train0)} rows over {df_train0['date'].nunique()} dates from {df_train0['date'].min().date()} to {df_train0['date'].max().date()}")

        df_recent0 = df0[(df0['date']>=date_recent_start) & (df0['date']<=date_recent_end)][cols_select].reset_index(drop=True).rename(columns=cols_map)
        if len(df_recent0) > 0:
            self.printfunc(f"Recent data contains {len(df_recent0)} rows over {df_recent0['date'].nunique()} dates from {df_recent0['date'].min().date()} to {df_recent0['date'].max().date()}")
        else:
            df_recent0 = pd.DataFrame()
            self.printfunc(f"No recent data specified.")
        
        return df_train0, df_recent0


    def fitTransform(self, log=True, n_std=2):
        """
        """
        df_train, df_recent = self._splitTrainRecent()

        self.printfunc("Fit regression...", end=' ')
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
        self.printfunc(f"{df_train['is_outlier'].sum()} out of {len(df_train)} dates are outliers...", end=' ')

        self.printfunc("Re-fit wihtout outliers...", end=' ')
        index_select = ~df_train['is_outlier']
        x_train_filter = np.array(df_train[index_select]['x']).reshape(-1, 1)
        y_train_filter = np.log(df_train[index_select]['price']) if log else df_train[index_select]['price']
        lm = LinearRegression().fit(x_train_filter, y_train_filter)
        y_pred = lm.predict(x_train)
        df_train['trend'] = np.exp(y_pred) if log else y_pred

        df_train['is_recent'] = False
        df_train.drop(columns=['residual'], inplace=True)

        if len(df_recent) > 0:
            self.printfunc("Estimate recent...", end=' ')
            df_recent['x'] = np.arange(0, len(df_recent)) + x_train_max + 1
            x_recent = np.array(df_recent['x']).reshape(-1, 1)
            y_recent = lm.predict(x_recent)
            df_recent['trend'] = np.exp(y_recent) if log else y_recent
            df_recent['is_outlier'] = False
            df_recent['is_recent'] = True
        else:
            self.printfunc("No recent data to estimate...", end=' ')
        
        self._df_train, self._df_recent = df_train, df_recent
        self.printfunc("done!")

        return df_train, df_recent


    def evaluateStatistics(self, return_statistics=False):
        """
        """
        df_train, df_recent = self._df_train.copy(), self._df_recent.copy()

        df_train_filter = df_train[~df_train['is_outlier']][['price','trend']]
        self._rmse_train = np.sqrt(((df_train_filter['price'] - df_train_filter['trend'])**2).sum() / len(df_train_filter))
        self.printfunc(f"Train RMSE = {self._rmse_train:.4f}.", end=' ')

        date_train_start = df_train['date'].min()
        date_train_end = df_train['date'].max()
        self._train_years = (date_train_end - date_train_start).days / 365
        trend_train_start = df_train['trend'].iloc[0]
        trend_train_end = df_train['trend'].iloc[-1]
        self._annualized_return = (trend_train_end / trend_train_start)**(1/self._train_years) - 1
        self.printfunc(f"Annualized Return = {self._annualized_return:.4f} over {self._train_years:.2f} years.", end=' ')

        df_combine = pd.concat([df_train, df_recent]).reset_index(drop=True)
        self._currenct_price = df_combine['price'].iloc[-1]
        self._base_price = df_combine['trend'].iloc[-1]
        self._over_value_range = self._currenct_price / self._base_price - 1
        self.printfunc(f"Compared to base price {self._base_price:.3f}, the current price {self._currenct_price:.3f} is over-valued by {self._over_value_range:.4f}", end='')
        self._over_value_years = self._over_value_range/self._annualized_return if self._over_value_range>0 else np.nan
        if self._over_value_range>0:
            self.printfunc(f" or {self._over_value_years:.2f} years.")
        else:
            self.printfunc(f".")

        if return_statistics:
            return self._rmse_train, self._train_years, self._annualized_return, self._currenct_price, self._base_price, self._over_value_range, self._over_value_years


    def plotTrendline(self, title='a chart'):
        """
        """
        df_train, df_recent = self._df_train.copy(), self._df_recent.copy()
        fig = go.Figure()

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

        fig.update_layout(template='plotly_dark', title=title, xaxis_title='date', yaxis_title='price')
        
        return fig
