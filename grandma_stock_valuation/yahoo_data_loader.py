"""
Utilities for querying data from Yahoo.
"""

import pandas as pd
from datetime import date
import logging
from os import mkdir
from os import path


DEFAULT_DATA_FOLDER = '_data'


def _printLevel(msg, level=logging.INFO):
    """
    A wrapper over `print` to support `level` argument.
    """
    print(msg)


class YahooDataLoader():
    """
    Class of the data loader to query data from Yahoo Finance.
    """

    def __init__(self, ticker, date_start=None, date_end_ex=None, verbose=0, printfunc=_printLevel) -> None:
        """
        Initialize Yahoo data loader.

        Parameters
        ----------
        ticker : str
            Ticker of the instrument, such as "IVV".
        date_start : str ("yyyy-mm-dd") | date | None
            Start date of query. If None, will derive a start date from an existing data file.
        date_end_ex : str ("yyyy-mm-dd") | date | None
            End date of query, which is exclusive - the last date returned will be smaller then it.
            If None, will use the execution date.
        verbose : int
            2 to print detailed information; 1 to print key information; 0 to suppress print.
        printfunc : func
            Function to output messages, which should support the `level` argument.
        """
        self.ticker = ticker
        self.date_start = date_start
        self.date_end_ex = date_end_ex
        self.verbose = verbose
        self.printfunc = printfunc
        
        self._date_start = None
        self._date_end_ex = None
        self._sec_start = None
        self._sec_end = None
        self._url_eod = None
        self._url_dividend = None
        self._file_name_last = '' # file name involved by the most recent operation

        self.default_folder = DEFAULT_DATA_FOLDER


    def _getFileAndDates(self, name, load_file, file_name) -> pd.DataFrame:
        """
        Set start and end dates, and load existing data file if needed.

        Parameters
        ----------
        name : str
            A name used to construct the default file name.
        load_file : bool
            If True, load data file stored at `file_name`
        file_name: str
            The file containing the existing data.
            
        Returns
        -------
        pandas.DataFrame
            The loaded data (if any).
        """
        # load existing file if specified, or if date_start is not specified
        df0 = pd.DataFrame()

        if load_file or (self.date_start is None):
            self._file_name_last = file_name

            if path.exists(file_name):
                if self.verbose > 0: self.printfunc(f"{self.ticker}: Existing {name} data file found at {file_name}.")
                df0 = pd.read_csv(file_name)
                df0['date'] = pd.to_datetime(df0['date'])
                if self.verbose > 0: self.printfunc(f"{self.ticker}: Existing {name} data file contains {len(df0)} rows over {df0['date'].nunique()} dates from {df0['date'].min().date()} to {df0['date'].max().date()}.")

        # set start date: derive from the loaded existing file if not specified
        if self.date_start is not None:
            self._date_start = pd.to_datetime(self.date_start)
        else:
            if len(df0)>0:
                self._date_start = df0['date'].max() + pd.DateOffset(days=1)
            else:
                self.printfunc(f'{file_name} must contain data with a date column, if date_start is None.', level=logging.ERROR)
                raise Exception('Incorrect inputs.')
            
        self._sec_start = self._getYahooSec(self._date_start)

        # set end date: use execution date if not specified
        if self.date_end_ex is None:
            self._date_end_ex = pd.to_datetime(date.today()) + pd.DateOffset(days=1)
        else:
            self._date_end_ex = pd.to_datetime(self.date_end_ex)

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


    def _queryYahooUrl(self, url, name, map_cols={}) -> pd.DataFrame:
        """
        Function to query Yahoo data from a url.

        Parameters
        ----------
        url : str
            The Yahoo url to query from.
        name : str
            Name of data to be queried; only used for logging and output file name.
        map_cols: dict(str:str)
            Rename the columns of data queried.

        Returns
        -------
        pandas.DataFrame
            The queried data.
        """
        try:
            df = pd.read_csv(url).sort_values('Date').dropna()
            if len(df)>0:
                if len(map_cols)>0:
                    df.rename(columns=map_cols, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                if self.verbose > 0: self.printfunc(f"{self.ticker}: Queried {name} data contains {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
            else:
                self.printfunc(f"{self.ticker}: No {name} data returned.", level=logging.WARNING)
                df = pd.DataFrame()
        except:
            self.printfunc(f"{self.ticker}: Failed to query {name} data!", level=logging.ERROR)
            df = pd.DataFrame()

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

        Returns
        -------
        pandas.DataFrame
            The refreshed data.
        """
        if len(df_query)==0:
            self.printfunc(f"{self.ticker}: No new data to refresh.", level=logging.WARNING)
            return df_exist

        self._file_save_last_ = file_name
        if len(df_exist)==0:
            if self.verbose > 0: self.printfunc(f"{self.ticker}: Save queried data to {file_name}.")
            df_query.to_csv(file_name, index=False, compression='gzip')
            return df_query

        else:
            date_low = df_query['date'].min().floor('D')
            date_high = (df_query['date'].max() + pd.DateOffset(days=1)).floor('D')
            df = df_exist[(df_exist['date']<date_low) | (df_exist['date']>=date_high)].copy()
            df = pd.concat([df, df_query]).sort_values('date').reset_index(drop=True)
            if self.verbose > 0: self.printfunc(f"{self.ticker}: Amended data file contains {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")

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
        file_name: str
            The csv file to save the queried data. Default to '__data__/<ticker>_EOD.csv.gz'.
            
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
            url = self._url_eod,
            name = 'EOD',
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


    def queryDividend(self, save=False, file_name=''):
        """
        Function to query Yahoo dividend data.

        Parameters
        ----------
        save: bool
            If True, save the queried data to a csv specified by `file_name`.
            If the csv file already exists, amend the existing file with the queried data.
        file_name: str
            The csv file to save the queried data. Default to '__data__/<ticker>_dividend.csv.gz'.
            
        Returns
        -------
        pandas.DataFrame
            If `save=True`, return the refreshed data, else return the queried data.
        """
        if file_name == '':
            file_name = path.join(self.default_folder, f'{self.ticker}_EOD.csv.gz')
            if not path.exists(self.default_folder):
                mkdir(self.default_folder)
        df_exist = self._getFileAndDates(name='dividend', load_file=save, file_name=file_name)

        s_head = 'https://query1.finance.yahoo.com/v7/finance/download/'
        s_tail = '&interval=1d&events=div&includeAdjustedClose=true'
        self._url_dividend = s_head+self.ticker+'?period1='+str(self._sec_start)+'&period2='+str(self._sec_end)+s_tail

        df_query = self._queryYahooUrl(
            url = self._url_dividend,
            name = 'dividend',
            map_cols={'Date':'date', 'Dividends':'dividend'}  ,
            )
        
        if save:
            df_refresh = self._refreshDataFile(df_exist, df_query, file_name)
            return df_refresh
        else:
            return df_query
