"""
Utilities for querying data from Yahoo.
"""

import pandas as pd
from logging import INFO, WARNING, ERROR
from os import path, mkdir
from datetime import date
from . import grandma_base

LOGPRINT = grandma_base.logger.logPandas
DEFAULT_DATA_FOLDER = '_data'

class YahooDataLoader():
    """
    Class of the data loader to query data from Yahoo Finance.
    """

    def __init__(self, ticker, date_start=None, date_end=None, verbose=0) -> None:
        """
        Initialize Yahoo data loader.

        Parameters
        ----------
        ticker : str
            Ticker of the instrument, such as "IVV".
        date_start : str ("yyyy-mm-dd") | date | None
            Start date of query. If None, will derive a start date from an existing data file.
        date_end : str ("yyyy-mm-dd") | date | None
            End date of query, which is inclusive.
            If None, will use the execution date.
        verbose : int
            2 to print detailed information; 1 to print key information; 0 to suppress print.
        """
        self.ticker = ticker
        self.date_start = date_start
        self.date_end = date_end
        self.verbose = verbose
        
        self._date_start = None
        self._date_end_ex = None
        self._sec_start = None
        self._sec_end = None
        self._url_eod = None
        self._url_dividend = None

        self._file_name_last = None # file loaded by the most recent operation - for debugging
        self._file_save_last = None # file saved by the most recent operation- for debugging

        self.default_folder = DEFAULT_DATA_FOLDER


    def _getFileAndDates(self, name, load_file, file_name) -> pd.DataFrame:
        """
        Set start and end dates, and load existing data file if needed.

        Parameters
        ----------
        name : str
            A name used to construct the default file name.
        load_file : bool
            If True, load data file stored at `file_name`.
        file_name: str
            The file containing the existing data.
            
        Returns
        -------
        pandas.DataFrame
            The loaded data (if any).
        """
        # initialize the loaded data.
        df0 = pd.DataFrame()

        # If specified to load an existing file, or if date_start is None: load existing file
        if load_file or (self.date_start is None):
            self._file_name_last = file_name

            if path.exists(file_name):
                if self.verbose > 0: LOGPRINT(f"{self.ticker}: Existing {name} data file found at {file_name}.")
                df0 = pd.read_csv(file_name)
                df0['date'] = pd.to_datetime(df0['date'])
                n_rows = len(df0)
                n_date = df0['date'].dt.strftime('%Y-%m-%d').nunique()
                if n_rows != n_date:
                    LOGPRINT(f"{self.ticker}: Existing data has {n_rows} rows from {n_date} dates.", level=WARNING)
                if self.verbose > 0: LOGPRINT(f"{self.ticker}: Existing {name} data file contains {n_rows} rows over {n_date} dates from {df0['date'].min().date()} to {df0['date'].max().date()}.")

        # set start date: if not specified, use the most recent date from the loaded existing file 
        if self.date_start is not None:
            self._date_start = pd.to_datetime(self.date_start).floor('D')
        else:
            if len(df0)>0:
                self._date_start = df0['date'].max().floor('D')
            else:
                LOGPRINT(f'If date_start if None, {file_name} must contain data with a date column.', level=ERROR)
                raise Exception('Incorrect inputs.')
            
        self._sec_start = self._getYahooSec(self._date_start)

        # set end date: use execution date if not specified
        if self.date_end is None:
            self._date_end_ex = pd.to_datetime(date.today()).floor('D') + pd.DateOffset(days=1)
        else:
            self._date_end_ex = pd.to_datetime(self.date_end).floor('D') + pd.DateOffset(days=1)

        self._sec_end = self._getYahooSec(self._date_end_ex)

        return df0


    def _getYahooSec(self, date) -> int:
        """
        Function to convert a date to yahoo format.

        Parameters
        ----------
        date : str ("yyyy-mm-dd") | date
            The date to be coverted
        
        Returns
        -------
        int
            The Yahoo format, which is a second presenting the start of the date.
        """
        yahoo_sec = (pd.to_datetime(date)-pd.to_datetime('1970-01-01')).days*24*3600

        return yahoo_sec


    def _queryYahooUrl(self, url, date_col, price_col, name, map_cols=None) -> pd.DataFrame:
        """
        Function to query Yahoo data from a url.

        Parameters
        ----------
        url : str
            The Yahoo url to query from.
        date_col : str
            Column of date.
        price_col : str
            Column of price, which should contain only positive values.
        name : str
            Name of data to be queried; only used for logging and output file name.
        map_cols: dict of {str:str} | None
            Rename the columns of data queried.

        Returns
        -------
        pandas.DataFrame
            The queried data.
        """
        df = pd.DataFrame()

        try:
            df = pd.read_csv(url).dropna()
            df = df[df[price_col]>0]
            df.sort_values(date_col, ignore_index=True, inplace=True)
            if map_cols is not None:
                df.rename(columns=map_cols, inplace=True)
            
            n_rows = len(df)
            if n_rows > 0:
                df['date'] = pd.to_datetime(df['date'])
                n_date = df['date'].dt.strftime('%Y-%m-%d').nunique()
                if n_rows != n_date:
                    LOGPRINT(f"{self.ticker}: Queried data has {n_rows} rows from {n_date} dates.", level=WARNING)
                if self.verbose > 0: LOGPRINT(f"{self.ticker}: Queried {name} data contains {n_rows} rows over {n_date} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
            else:
                LOGPRINT(f"{self.ticker}: No {name} data returned.", level=WARNING)
        except:
            LOGPRINT(f"{self.ticker}: Failed to query {name} data!", level=ERROR)

        return df


    def _refreshDataFile(self, df_exist, df_query, file_name) -> pd.DataFrame:
        """
        Function to refresh the stored data file.

        Parameters
        ----------
        df_exist: pandas.DataFrame
            Previously stored data.
        df_query: pandas.DataFrame
            New data queried.
        file_name: str
            The csv file to save the combined data.

        csv is only saved if `df_query` contains data.

        Returns
        -------
        pandas.DataFrame
            The refreshed data.
        """
        if len(df_query)==0:
            LOGPRINT(f"{self.ticker}: No new data to refresh.", level=WARNING)
            return df_exist

        self._file_save_last = file_name
        if len(df_exist)==0:
            if self.verbose > 0: LOGPRINT(f"{self.ticker}: Save queried data to {file_name}.")
            df_query.to_csv(file_name, index=False, compression='gzip')
            return df_query

        else:
            date_low = df_query['date'].min()
            date_high = df_query['date'].max()
            df = df_exist[(df_exist['date']<date_low) | (df_exist['date']>date_high)]
            df = pd.concat([df, df_query], ignore_index=True, copy=False).sort_values('date', ignore_index=True)
            n_rows = len(df)
            n_date = df['date'].dt.strftime('%Y-%m-%d').nunique()
            if n_rows != n_date:
                LOGPRINT(f"{self.ticker}: Refreshed data has {n_rows} rows from {n_date} dates.", level=WARNING)
            if self.verbose > 0: LOGPRINT(f"{self.ticker}: Amended data file contains {n_rows} rows over {n_date} dates from {df['date'].min().date()} to {df['date'].max().date()}.")

            df.to_csv(file_name, index=False, compression='gzip')
            return df


    def queryEOD(self, save=True, file_name=None) -> pd.DataFrame:
        """
        Function to query Yahoo EOD data.

        Parameters
        ----------
        save: bool
            If True, save the queried data to a csv specified by `file_name`.
            If the csv file already exists, amend the existing file with the queried data.
        file_name: str | None
            The csv file to save the queried data. If None, default to '_data/<ticker>_EOD.csv.gz'.
            
        Returns
        -------
        pandas.DataFrame
            If `save=True`, return the refreshed data, else return the queried data.
        """
        if file_name is None:
            file_name = path.join(self.default_folder, f'{self.ticker}_EOD.csv.gz')
            if not path.exists(self.default_folder):
                mkdir(self.default_folder)
        df_exist = self._getFileAndDates(name='EOD', load_file=save, file_name=file_name)

        s_head = 'https://query1.finance.yahoo.com/v7/finance/download/'
        s_tail = '&interval=1d&events=history&includeAdjustedClose=true'
        self._url_eod = s_head+self.ticker+'?period1='+str(self._sec_start)+'&period2='+str(self._sec_end)+s_tail

        df_query = self._queryYahooUrl(
            url=self._url_eod,
            date_col='Date',
            price_col='Close',
            name='EOD',
            map_cols={
                'Date':'date',
                'Open':'open', 'High': 'high', 'Low':'low', 'Close':'close',
                'Adj Close':'close_adj',
                'Volume':'volume'
                }
            )
        
        if save:
            df_refresh = self._refreshDataFile(df_exist, df_query, file_name)
            return df_refresh
        else:
            return df_query


    def queryDividend(self, save=True, file_name=None) -> pd.DataFrame:
        """
        Function to query Yahoo dividend data.

        Parameters
        ----------
        save: bool
            If True, save the queried data to a csv specified by `file_name`.
            If the csv file already exists, amend the existing file with the queried data.
        file_name: str | None
            The csv file to save the queried data. Default to '__data__/<ticker>_dividend.csv.gz'.
            
        Returns
        -------
        pandas.DataFrame
            If `save=True`, return the refreshed data, else return the queried data.
        """
        if file_name is None:
            file_name = path.join(self.default_folder, f'{self.ticker}_dividend.csv.gz')
            if not path.exists(self.default_folder):
                mkdir(self.default_folder)
        df_exist = self._getFileAndDates(name='dividend', load_file=save, file_name=file_name)

        s_head = 'https://query1.finance.yahoo.com/v7/finance/download/'
        s_tail = '&interval=1d&events=div&includeAdjustedClose=true'
        self._url_dividend = s_head+self.ticker+'?period1='+str(self._sec_start)+'&period2='+str(self._sec_end)+s_tail

        df_query = self._queryYahooUrl(
            url=self._url_dividend,
            date_col='Date',
            price_col='Dividends',
            name='dividend',
            map_cols={'Date':'date', 'Dividends':'dividend'}
        )
        
        if save:
            df_refresh = self._refreshDataFile(df_exist, df_query, file_name)
            return df_refresh
        else:
            return df_query
