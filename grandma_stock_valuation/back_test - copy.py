"""
Back test of Grandma Stock Valuation and Grandma Portfolio Allocation.

"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .valuation_model import batchValuation, addCashPortfolio
from .portfolio_allocator import getCorrelationWeight, allocatePortfolio


DEFAULT_OUTPUT_FOLDER = '_output'


def _printLevel(*args, level=logging.INFO):
    """
    A wrapper over `print` to support `level` argument.
    """
    print(*args)


class GrandmaBackTester():
    """
    Class to back test Grandma Stock Valuation and Grandma Portfolio Allocation.
    """

    def __init__(self, backtest_years=10, adjust_freq_months=1,
                 init_parameters={'recent_months':0, 'train_years':10, 'min_train_years':5, 'date_end':None},
                 fit_parameters={'price_col':'close_adj', 'log':True, 'n_std':1.5},
                 valuate_parameters={'min_annual_return':0.01},
                 allocation_parameters={'transformation':'sigmoid', 'scale':1},
                 with_cash=True, with_correlation_weights=True,
                 verbose=0,
                 printfunc=_printLevel):
        """
        Initialize the back tester.

        Parameters
        ----------
        xxxxx

        verbose : int
            2 to print detailed information; 1 to print high-level information; 0 to suppress print.
        printfunc : func
            Function to output messages, which should support the `level` argument.

        """
        self.backtest_years = backtest_years
        self.adjust_freq_months = adjust_freq_months
        self.init_parameters = init_parameters
        self.fit_parameters = fit_parameters
        self.valuate_parameters = valuate_parameters
        self.allocation_parameters = allocation_parameters
        self.with_cash = with_cash
        self.with_correlation_weights = with_correlation_weights
        self.verbose = verbose
        self.printfunc = printfunc

        self._minimum_training_years = init_parameters['min_train_years']
        self._maximum_training_years = init_parameters['train_years']
        self._date_end = init_parameters['date_end']
        self._price_col = fit_parameters['price_col']

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
        # clean price data, get start and end date of each ticker
        d_instrument_prices = {}
        d_instrument_start_dates = {}
        d_instrument_end_dates = {}
        for ticker, df_prices in d_instrument_data.items():
            df_prices = df_prices[df_prices[self._price_col]>0][['date', self._price_col]].reset_index(drop=True)
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            d_instrument_prices[ticker] = df_prices
            d_instrument_start_dates[ticker] = df_prices['date'].min()
            d_instrument_end_dates[ticker] = df_prices['date'].max()

        # dates of back-test periods
        if self._date_end is None:
            backtest_end_date = max(d_instrument_end_dates.values())
        else:
            backtest_end_date = pd.to_datetime(self._date_end)
        backtest_end_date_ex = backtest_end_date.floor('D') + pd.DateOffset(days=1)

        backtest_start_date = backtest_end_date_ex - pd.DateOffset(years=self.backtest_years)
        if self.verbose > 0: self.printfunc(f"To backtest {self.backtest_years} years, from {backtest_start_date.date()} to {backtest_end_date.date()}")

        at_least_start_date = backtest_start_date - pd.DateOffset(years=self._minimum_training_years)
        at_most_start_date = backtest_start_date - pd.DateOffset(years=self._maximum_training_years)

        for ticker, d in d_instrument_start_dates.items():
            if d > at_least_start_date:
                self.printfunc(f"{ticker}'s start date {d.date()} is beyond {at_least_start_date.date()} for full back-test.")

        # combine cleaned price data of all instruments into one dateframe
        df_instrument_prices = pd.DataFrame()
        for ticker, df_prices in d_instrument_prices.items():
            index_select = (df_prices['date']>=at_most_start_date) & (df_prices['date']<backtest_end_date_ex)
            df = df_prices[index_select].copy()
            df.rename(columns={self._price_col:'price_'+ticker}, inplace=True)

            if len(df_instrument_prices)==0:
                df_instrument_prices = df.copy()
                continue
            else:
                df_instrument_prices = df_instrument_prices.merge(df, 'outer', 'date')

        df_instrument_prices = df_instrument_prices.sort_values('date').reset_index(drop=True)
        df_instrument_prices.fillna(method='pad', inplace=True)

        ## index of the backtest start date in the combined dateframe
        index_start = df_instrument_prices.index[df_instrument_prices['date'].tolist().index(backtest_start_date)]

        df_instrument_prices.index = df_instrument_prices['date']
        df_instrument_prices.drop(columns=['date'], inplace=True)

        self._d_instrument_start_dates = d_instrument_start_dates
        self._d_instrument_end_dates = d_instrument_end_dates
        self._backtest_start_date = backtest_start_date
        self._backtest_end_date_ex = backtest_end_date_ex
        self._index_start = index_start
        self._df_instrument_prices = df_instrument_prices


    def _updateValue(self, df_portfolio_i, df_instrument_prices, index_i):
        """
        Update value of portfolio with updated instrument prices.
        """
        df_new_price = df_instrument_prices.iloc[index_i].reset_index()
        df_new_price.columns = ['ticker', 'new_price']
        df_new_price['ticker'] = df_new_price['ticker'].str.replace('price_','')

        df_portfolio_i = df_portfolio_i.merge(df_new_price, 'left', 'ticker')

        index_null = df_portfolio_i['current_price'].isnull() | df_portfolio_i['current_value'].isnull() | df_portfolio_i['new_price'].isnull()
        df_portfolio_i.loc[~index_null, 'current_value'] = (df_portfolio_i['current_value'] * df_portfolio_i['new_price'] / df_portfolio_i['current_price'])[~index_null] 
        df_portfolio_i['current_price'] = df_portfolio_i['new_price']
        df_portfolio_i.drop(columns='new_price', inplace=True)

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
            valuate_parameters=self.valuate_parameters,
            draw_figure=False,
            save_result=False,
            metric_file = None,
            figure_folder = None,
            verbose=self.verbose - 1,
            printfunc=self.printfunc
        )

        valid_tickers = df_metrics_i[df_metrics_i['over_value_years'].notnull()]['ticker'].to_list()

        if self.with_cash:
            df_metrics_i = addCashPortfolio(df_metrics_i)

        if self.with_correlation_weights:
            weights = getCorrelationWeight(
                d_instrument_prices={k:v for k,v in d_instrument_data_i.items() if k in valid_tickers},
                price_col=self._price_col,
                recent_months=self.init_parameters['recent_months'],
                train_years=self.init_parameters['train_years'],
                with_cash=self.with_cash,
                verbose=self.verbose - 1,
                printfunc=self.printfunc
            )
        else:
            n_inst = len(valid_tickers) + self.with_cash
            w_inst = 1/n_inst if n_inst>0 else np.nan
            weights = {t:w_inst for t in valid_tickers + ['cash']*self.with_cash}

        df_metrics_i['weight'] = df_metrics_i['ticker'].apply(lambda t: weights.get(t, np.nan))

        df_metrics_i['portfolio_allocation'] = allocatePortfolio(
            df_metrics_i['over_value_years'],
            **self.allocation_parameters,
            with_cash=self.with_cash,
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
                df_portfolio_i = df_metrics_i[cols_select].copy().rename(columns={'portfolio_allocation':'current_portfolio_pct'})

                df_metrics_i['date'] = dt
                l_adjustments.append(df_metrics_i)
                if self.verbose > 0: self.printfunc(f"Adjust portfolio on {dt.date()}, total value = {total_value:.6f}")
            
            d_total_value[dt] = total_value
            df_portfolio_i['date'] = dt
            l_portfolio.append(df_portfolio_i)

        if self.verbose > 0: self.printfunc(f"final portfolio increased by {total_value/total_value_start-1:.3f} over {self.backtest_years} years, which is {(total_value/total_value_start)**(1/self.backtest_years)-1:.4f} annualized growth.")

        self.df_total_value = pd.DataFrame({'date':d_total_value.keys(), 'Grandma':d_total_value.values()})

        cols_first, cols_fill = ['date','ticker'], ['portfolio_allocation', 'current_value']
        self.df_adjustments = pd.concat(l_adjustments, ignore_index=True, copy=False)
        self.df_adjustments = self.df_adjustments[cols_first + list(self.df_adjustments.columns.drop(cols_first))]
        self.df_adjustments[cols_fill] = self.df_adjustments[cols_fill].fillna(0)

        cols_first, cols_fill = ['date','ticker'], ['current_value', 'current_portfolio_pct']
        self.df_portfolio = pd.concat(l_portfolio, ignore_index=True, copy=False)
        self.df_portfolio = self.df_portfolio[cols_first + list(self.df_portfolio.columns.drop(cols_first))]
        self.df_portfolio[cols_fill] = self.df_portfolio[cols_fill].fillna(0)

        self.df_average_value = self.df_portfolio.groupby('ticker')['current_value'].mean().reset_index()
        self.df_average_value['current_value'] = self.df_average_value['current_value'].fillna(0)
        self.df_average_value['avg_pct_allocation'] = self.df_average_value['current_value'] / self.df_average_value['current_value'].sum()
        self.df_average_value = self.df_average_value[['ticker', 'avg_pct_allocation']].copy()


    def plotBackTest(self, **kwargs):
        """
        Plot back test results. Should be run after `runBackTest()`.

        """
        # prepare hover text of portfolio adjustments
        cols_adj = ['date', 'ticker', 'train_years', 'annualized_return', 'over_value_range', 'over_value_years', 'weight', 'portfolio_allocation']
        df_adjs = self.df_adjustments[cols_adj].copy()

        df_adjs['train_years'] = df_adjs['train_years'].astype(float).round(1).astype(str)
        df_adjs['annualized_return'] = (100*df_adjs['annualized_return'].astype(float)).round(1).astype(str)
        df_adjs['over_value_range'] = (100*df_adjs['over_value_range'].astype(float)).round(1).astype(str)
        df_adjs['over_value_years'] = df_adjs['over_value_years'].astype(float).round(2).astype(str)
        df_adjs['weight'] = df_adjs['weight'].astype(float).round(2).astype(str)
        df_adjs['portfolio_allocation'] = (100*df_adjs['portfolio_allocation'].astype(float)).round(1).astype(str)

        df_adjs['adj_hover_text'] = ''

        index_train = df_adjs['train_years']!='nan'
        df_adjs['train_text'] = df_adjs['annualized_return'] + '% ann. growth in ' + df_adjs['train_years'] + 'yrs'
        df_adjs['train_text'] = df_adjs['train_text'] + '<br>valuation = ' + df_adjs['over_value_range'] + '%'
        df_adjs.loc[index_train, 'adj_hover_text'] = df_adjs['train_text'][index_train] + '<br>'

        index_value = df_adjs['over_value_years']!='nan'
        df_adjs['value_text'] = 'weight = ' + df_adjs['weight'] + '; valuation = ' + df_adjs['over_value_years'] + ' yrs'
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
        df_grandma_growth['hover_text'] = (100*df_grandma_growth['hover_text'].astype(float)).round(1).astype(str) + '% Cash'

        ## growth of each instrument
        df_instrument_growth = self._df_instrument_prices.iloc[self._index_start:].copy()
        d_first_prices = df_instrument_growth.apply(lambda se: se.dropna().iloc[0])

        df_instrument_growth = df_instrument_growth.apply(lambda se: se/d_first_prices[se.name])
        df_instrument_growth.columns = [c.replace('price_', '') for c in df_instrument_growth.columns]
        df_instrument_growth = df_instrument_growth.reset_index()
        df_instrument_growth = df_instrument_growth.melt(id_vars='date', var_name='ticker', value_name='growth')

        df_instrument_growth = df_instrument_growth.merge(df_daily_portfolio_pct, 'left', ['date','ticker']).fillna(0)
        df_instrument_growth.rename(columns={'current_portfolio_pct':'hover_text'}, inplace=True)
        df_instrument_growth['hover_text'] = (100*df_instrument_growth['hover_text'].astype(float)).round(1).astype(str) + '% Portfolio'

        ## combine together for plotting
        ## NOTE: the sequence will matter when plotting
        df_instrument_growth = pd.concat([df_grandma_growth, df_instrument_growth], ignore_index=True, copy=False)

        ## change hover text of portfolio adjustment dates
        df_instrument_growth = df_instrument_growth.merge(df_adjs, 'left', ['date', 'ticker'])
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
