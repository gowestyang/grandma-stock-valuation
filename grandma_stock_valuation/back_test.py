"""
Back test of Grandma Stock Valuation and Grandma Portfolio Allocation.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import grandma_base
from .valuation_model import batchValuation, addCashPortfolio, subtractValueGrowth, divideValueGrowth
from .portfolio_allocator import getCorrelationWeight, allocatePortfolio

LOGPRINT = grandma_base.logger.logPandas

class GrandmaBackTester():
    """
    Class to back test Grandma Stock Valuation and Grandma Portfolio Allocation.
    """

    def __init__(self, backtest_years=10, adjust_freq_months=1,
                 init_parameters=dict(recent_months=0, train_years=10, min_train_years=5, date_end=None),
                 fit_parameters=dict(price_col='close', is_positive=False, is_sorted=False, log=True, n_std=1),
                 adjust_growth_func=subtractValueGrowth,
                 growth_scale=1,
                 allocation_parameters=dict(transformation='exponential', scale=None, center=-0.07, lower_bound=-0.2),
                 with_cash=False,
                 cash_parameters = dict(cash_value=0.0),
                 with_correlation_weights=False,
                 verbose=0):
        """
        Initialize the back tester.

        Parameters
        ----------
        xxxxx

        verbose : int
            2 to print detailed information; 1 to print high-level information; 0 to suppress print.
        """
        self.backtest_years = backtest_years
        self.adjust_freq_months = adjust_freq_months
        self.init_parameters = init_parameters
        self.fit_parameters = fit_parameters
        self.adjust_growth_func = adjust_growth_func
        self.growth_scale = growth_scale
        self.allocation_parameters = allocation_parameters
        self.with_cash = with_cash
        self.cash_parameters = cash_parameters
        self.with_correlation_weights = with_correlation_weights
        self.verbose = verbose

        self._minimum_training_years = init_parameters['min_train_years']
        self._maximum_training_years = init_parameters['train_years']
        self._date_end = init_parameters['date_end']

        self._price_col = fit_parameters['price_col']
        self._is_positive = fit_parameters['is_positive']
        self._is_sorted = fit_parameters['is_sorted']

        self._d_instrument_start_dates = None
        self._d_instrument_end_dates = None
        self._backtest_start_date = None
        self._backtest_end_date_ex = None
        self._index_start = None
        self._df_instrument_prices = None

        self.df_total_value = None
        self.df_adjustments = None
        self.df_portfolio = None
        self.df_average_value = None


    def _cleanInputData(self, d_instrument_data):
        """
        Transform the daily price data of instruments into a dataframe for back-testing.
        """
        # dates of back-test periods
        backtest_end_date_ex =  max(map(lambda df:df['date'].max(), d_instrument_data.values()))
        if self._date_end is not None:
            backtest_end_date_ex = min(pd.to_datetime(self._date_end), backtest_end_date_ex)
        backtest_end_date_ex = backtest_end_date_ex.floor('D') + pd.DateOffset(days=1)
        backtest_start_date = backtest_end_date_ex - pd.DateOffset(years=self.backtest_years)
        if self.verbose > 0: LOGPRINT(f"To backtest {self.backtest_years} years, from {backtest_start_date.date()} to {backtest_end_date_ex.date()}")

        at_least_start_date = backtest_start_date - pd.DateOffset(years=self._minimum_training_years)
        at_most_start_date = backtest_start_date - pd.DateOffset(years=self._maximum_training_years)

        # clean price data, gether start and end date of each ticker
        d_instrument_prices = {}
        d_instrument_start_dates = {}
        d_instrument_end_dates = {}
        for ticker, df_prices in d_instrument_data.items():
            df_prices = df_prices[['date', self._price_col]].copy()

            if not self._is_positive:
                df_prices = df_prices[df_prices[self._price_col]>0].reset_index(drop=True)
            if not self._is_sorted:
                df_prices.sort_values('date', ignore_index=True, inplace=True)

            date_start_i = df_prices['date'].iloc[0]
            if date_start_i > at_least_start_date:
                if self.verbose > 0: LOGPRINT(f"{ticker}'s start date {date_start_i} is beyond {at_least_start_date.date()} for full back-test.")
            d_instrument_start_dates[ticker] = date_start_i
            d_instrument_end_dates[ticker] = df_prices['date'].iloc[-1]

            index_select = (df_prices['date']>=at_most_start_date) & (df_prices['date']<backtest_end_date_ex)
            if index_select.sum() > 0:
                df_prices = df_prices[index_select].set_index('date')
                df_prices.rename(columns={self._price_col:'price_'+ticker}, inplace=True)
            else:
                df_prices = pd.DataFrame()

            d_instrument_prices[ticker] = df_prices

        df_instrument_prices = pd.concat(d_instrument_prices.values(), axis=1, copy=False)
        # if no instrument has valid data, this will yield empty dataframe without index nor column
        if len(df_instrument_prices) > 0:
            df_instrument_prices.fillna(method='pad', inplace=True)
            index_start = np.argmax(df_instrument_prices.index >= backtest_start_date)
        else:
            index_start = None
        
        # to self
        self._backtest_start_date = backtest_start_date
        self._backtest_end_date_ex = backtest_end_date_ex
        self._d_instrument_start_dates = d_instrument_start_dates
        self._d_instrument_end_dates = d_instrument_end_dates

        self._index_start = index_start
        self._df_instrument_prices = df_instrument_prices


    def _updateValue(self, df_portfolio_i, df_instrument_prices, index_i):
        """
        Update value of portfolio with updated instrument prices.
        """
        new_price = df_instrument_prices.iloc[index_i]
        new_price.index = new_price.index.str.replace('price_','')
        new_price = new_price.to_dict()
        ar_new_price = df_portfolio_i['ticker'].map(new_price).to_numpy()

        ar_current_price = df_portfolio_i['current_price'].to_numpy()
        ar_current_value = df_portfolio_i['current_value'].to_numpy()

        index_null = np.isnan(ar_current_price) | np.isnan(ar_current_value) | np.isnan(ar_new_price)
        df_portfolio_i['current_value'] = np.where(index_null, ar_current_value, ar_current_value*ar_new_price/ar_current_price)
        df_portfolio_i['current_price'] = ar_new_price

        total_value = df_portfolio_i['current_value'].sum()
        if total_value > 0:
            df_portfolio_i['current_portfolio_pct'] = df_portfolio_i['current_value'] / total_value
        else:
            df_portfolio_i['current_portfolio_pct'] = np.nan

        return df_portfolio_i, total_value


    def _getHistoricalData(self, df_instrument_prices, index_i, price_col):
        """
        Retrive all historical prices up to the given index location.
        """
        d_instrument_data_i = {}
        for col in df_instrument_prices.columns:
            ticker = col.replace('price_','')
            df = df_instrument_prices[col].iloc[:index_i+1].copy().rename(price_col).dropna().reset_index()

            if len(df)>0:
                d_instrument_data_i[ticker] = df
        
        return d_instrument_data_i


    def _getAllocation(self, d_instrument_data_i, total_value):
        """
        Get portfolio allocation of each instrument.
        """
        df_metrics_i, _ = batchValuation(
            d_instrument_data=d_instrument_data_i,
            init_parameters=self.init_parameters,
            fit_parameters=self.fit_parameters,
            draw_figure=False,
            save_result=False,
            figure_folder=None,
            verbose=self.verbose - 1
        )

        df_metrics_i['over_value_score'] = self.adjust_growth_func(
            df_metrics_i['over_value_range'],
            df_metrics_i['annualized_growth'],
            growth_scale=self.growth_scale
            )
        
        valid_tickers = df_metrics_i[df_metrics_i['over_value_score'].notnull()]['ticker'].to_list()

        if self.with_cash:
            df_metrics_i = addCashPortfolio(df_metrics_i, value_col='over_value_score', cash_value=self.cash_parameters['cash_value'])

        if self.with_correlation_weights:
            weights = getCorrelationWeight(
                d_instrument_prices={k:v for k,v in d_instrument_data_i.items() if k in valid_tickers},
                price_col=self._price_col,
                is_positive=True,
                is_sorted=True,
                recent_months=self.init_parameters['recent_months'],
                train_years=self.init_parameters['train_years'],
                with_cash=self.with_cash,
                verbose=self.verbose - 1
            )
        else:
            n_inst = len(valid_tickers) + self.with_cash
            w_inst = 1/n_inst if n_inst>0 else np.nan
            weights = {t:w_inst for t in valid_tickers + ['cash']*self.with_cash}

        df_metrics_i['weight'] = df_metrics_i['ticker'].map(weights)

        df_metrics_i['portfolio_allocation'] = allocatePortfolio(
            df_metrics_i['over_value_score'],
            **self.allocation_parameters,
            weights=df_metrics_i['weight']
        )

        df_metrics_i['current_value'] = total_value * df_metrics_i['portfolio_allocation']

        return df_metrics_i


    def runBackTest(self, d_instrument_data):
        """
        Run back test.

        Parameters
        ----------
        d_instrument_data : dict
            xxxx
        """
        self._cleanInputData(d_instrument_data)

        d_total_value = {}
        l_adjustments = []
        l_portfolio = []

        # initialize
        total_value_start = 1
        dt = self._df_instrument_prices.index[self._index_start]

        d_instrument_data_i = self._getHistoricalData(self._df_instrument_prices, self._index_start, self._price_col)
        df_metrics_i = self._getAllocation(d_instrument_data_i, total_value=total_value_start)
        df_metrics_i['date'] = dt

        cols_select = ['ticker','current_price','current_value','portfolio_allocation']
        df_portfolio_i = df_metrics_i[cols_select].rename(columns={'portfolio_allocation':'current_portfolio_pct'})
        df_portfolio_i['date'] = dt

        d_total_value[dt] = total_value_start
        l_adjustments.append(df_metrics_i)
        l_portfolio.append(df_portfolio_i)

        next_adjust_date = self._backtest_start_date + pd.DateOffset(months=self.adjust_freq_months)

        for index_i in range(self._index_start+1, len(self._df_instrument_prices)):

            dt = self._df_instrument_prices.index[index_i]

            df_portfolio_i, total_value = self._updateValue(df_portfolio_i, self._df_instrument_prices, index_i)

            if dt >= next_adjust_date:
                next_adjust_date = next_adjust_date + pd.DateOffset(months=self.adjust_freq_months)

                d_instrument_data_i = self._getHistoricalData(self._df_instrument_prices, index_i, self._price_col)
                df_metrics_i = self._getAllocation(d_instrument_data_i, total_value=total_value)
                df_portfolio_i = df_metrics_i[cols_select].rename(columns={'portfolio_allocation':'current_portfolio_pct'})

                df_metrics_i['date'] = dt
                l_adjustments.append(df_metrics_i)
                if self.verbose > 0: LOGPRINT(f"Adjust portfolio on {dt.date()}, total value = {total_value:.6f}")
            
            d_total_value[dt] = total_value
            df_portfolio_i['date'] = dt
            l_portfolio.append(df_portfolio_i)

        if self.verbose > 0: LOGPRINT(f"final portfolio increased by {total_value/total_value_start-1:.3f} over {self.backtest_years} years, which is {(total_value/total_value_start)**(1/self.backtest_years)-1:.4f} annualized growth.")

        self.df_total_value = pd.DataFrame({'date':d_total_value.keys(), 'Grandma':d_total_value.values()})

        cols_first, cols_fill = ['date','ticker'], ['portfolio_allocation', 'current_value']
        self.df_adjustments = pd.concat(l_adjustments, ignore_index=True, copy=False)
        self.df_adjustments = self.df_adjustments[cols_first + list(self.df_adjustments.columns.drop(cols_first))]
        self.df_adjustments[cols_fill].fillna(0, inplace=True)

        cols_first, cols_fill = ['date','ticker'], ['current_value', 'current_portfolio_pct']
        self.df_portfolio = pd.concat(l_portfolio, ignore_index=True, copy=False)
        self.df_portfolio = self.df_portfolio[cols_first + list(self.df_portfolio.columns.drop(cols_first))]
        self.df_portfolio[cols_fill].fillna(0, inplace=True)

        self.df_average_value = self.df_portfolio.groupby('ticker')['current_value'].mean().fillna(0).reset_index()
        self.df_average_value['avg_pct_allocation'] = self.df_average_value['current_value'] / self.df_average_value['current_value'].sum()
        self.df_average_value = self.df_average_value[['ticker', 'avg_pct_allocation']]


    def plotBackTest(self, **kwargs):
        """
        Plot back test results. Should be run after `runBackTest()`.

        """
        # prepare hover text of portfolio adjustments
        cols_adj = ['date', 'ticker', 'train_years', 'annualized_growth', 'over_value_range', 'over_value_score', 'weight', 'portfolio_allocation']
        df_adjs = self.df_adjustments[cols_adj].copy()

        df_adjs['train_years'] = df_adjs['train_years'].astype(float, copy=False).round(1).astype(str, copy=False)
        df_adjs['annualized_growth'] = (100*df_adjs['annualized_growth'].astype(float, copy=False)).round(1).astype(str, copy=False)
        df_adjs['over_value_range'] = (100*df_adjs['over_value_range'].astype(float, copy=False)).round(1).astype(str, copy=False)
        df_adjs['over_value_score'] = df_adjs['over_value_score'].astype(float, copy=False).round(2).astype(str, copy=False)
        df_adjs['weight'] = df_adjs['weight'].astype(float, copy=False).round(2).astype(str, copy=False)
        df_adjs['portfolio_allocation'] = (100*df_adjs['portfolio_allocation'].astype(float, copy=False)).round(1).astype(str, copy=False)

        df_adjs['adj_hover_text'] = ''

        index_train = df_adjs['train_years']!='nan'
        df_adjs['train_text'] = df_adjs['annualized_growth'] + '% ann. growth in ' + df_adjs['train_years'] + 'yrs'
        df_adjs['train_text'] = df_adjs['train_text'] + '<br>valuation = ' + df_adjs['over_value_range'] + '%'
        df_adjs.loc[index_train, 'adj_hover_text'] = df_adjs['train_text'][index_train] + '<br>'

        index_value = df_adjs['over_value_score']!='nan'
        df_adjs['value_text'] = 'weight = ' + df_adjs['weight'] + '; valuation = ' + df_adjs['over_value_score'] + ' yrs'
        df_adjs.loc[index_value, 'adj_hover_text'] = (df_adjs['adj_hover_text'] + df_adjs['value_text'])[index_value] + '<br>'

        df_adjs['adj_hover_text'] = df_adjs['adj_hover_text'] + df_adjs['portfolio_allocation']  + '% Portfolio'

        index_cash = df_adjs['ticker']=='cash'
        df_adjs.loc[index_cash, 'adj_hover_text'] = df_adjs['adj_hover_text'][index_cash].str.replace('Portfolio', 'Cash')
        df_adjs = df_adjs[['date','ticker','adj_hover_text']]

        # prepare daily growth charts of instruments
        df_daily_portfolio_pct = self.df_portfolio[['date','ticker','current_portfolio_pct']]

        ## growth of Grandma
        df_grandma_growth = self.df_total_value.melt(id_vars='date', var_name='ticker', value_name='growth')
        df_grandma_growth['ticker'] = 'cash' # to merge with cash % for hover text display
        df_grandma_growth = df_grandma_growth.merge(df_daily_portfolio_pct, 'left', ['date','ticker']).fillna(0)
        df_grandma_growth.rename(columns={'current_portfolio_pct':'hover_text'}, inplace=True)
        df_grandma_growth['hover_text'] = (100*df_grandma_growth['hover_text'].astype(float, copy=False)).round(1).astype(str, copy=False) + '% Cash'

        ## growth of each instrument
        df_instrument_growth = self._df_instrument_prices.iloc[self._index_start:].copy()
        d_first_prices = df_instrument_growth.apply(lambda se: se.dropna().iloc[0])

        df_instrument_growth = df_instrument_growth.apply(lambda se: se/d_first_prices[se.name])
        df_instrument_growth.columns = [c.replace('price_', '') for c in df_instrument_growth.columns]
        df_instrument_growth = df_instrument_growth.reset_index().melt(id_vars='date', var_name='ticker', value_name='growth')

        df_instrument_growth = df_instrument_growth.merge(df_daily_portfolio_pct, 'left', ['date','ticker']).fillna(0)
        df_instrument_growth.rename(columns={'current_portfolio_pct':'hover_text'}, inplace=True)
        df_instrument_growth['hover_text'] = (100*df_instrument_growth['hover_text'].astype(float, copy=False)).round(1).astype(str, copy=False) + '% Portfolio'

        ## combine together for plotting
        ## NOTE: the sequence will matter when plotting
        df_instrument_growth = pd.concat([df_grandma_growth, df_instrument_growth], ignore_index=True, copy=False)

        ## change hover text of portfolio adjustment dates
        df_instrument_growth = df_instrument_growth.merge(df_adjs, 'left', ['date', 'ticker'], copy=False)
        index_adj = df_instrument_growth['adj_hover_text'].notnull()
        df_instrument_growth.loc[index_adj, 'hover_text'] = df_instrument_growth['adj_hover_text'][index_adj]
        df_instrument_growth.drop(columns='adj_hover_text', inplace=True)

        index_grandma = df_instrument_growth['ticker'] == 'cash'
        df_instrument_growth.loc[index_grandma, 'ticker'] = 'Grandma'

        # plot the chart
        fig = px.line(df_instrument_growth, x='date', y='growth',
                      color='ticker', color_discrete_map={'Grandma':'white'},
                      custom_data = ['hover_text'],
                     )

        fig.update_traces(hovertemplate='%{y:.2f}<br>%{customdata}')
        fig.update_traces(line=dict(width=0.75))
        fig['data'][0]['line']['width'] = 1.5

        fig.update_layout(template='plotly_dark', title='Back-test',
                          xaxis_title='date', yaxis_title='growth',
                          hovermode='x unified',
                          xaxis_hoverformat='%Y-%m-%d',
                          **kwargs)

        index_adj_scatter = df_grandma_growth['date'].isin(self.df_adjustments['date'])
        df_adj_scatter = df_grandma_growth[index_adj_scatter][['date','growth']]

        fig.add_trace(go.Scatter(x=df_adj_scatter['date'], y=df_adj_scatter['growth'], mode='markers', showlegend=False,
                                 marker={'color':'yellow', 'size':8, 'symbol':'diamond-tall'},
                                 hoverinfo='skip'))

        return fig
