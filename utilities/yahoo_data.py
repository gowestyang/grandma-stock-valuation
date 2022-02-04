"""
Utilities for querying data from Yahoo.
"""

import pandas as pd
import os

class YahooDataLoader():
    def __init__(self, ticker, date_start, date_end_ex, printfunc=print) -> None:
        """
        Data loader to query data from Yahoo finance.

        Parameters
        ----------
        ticker : str
            Ticker of the instrument, such as "IVV".
        date_start : str ("yyyy-mm-dd") | date
            Start date of query
        date_end_ex : str ("yyyy-mm-dd") | date
            End date of query, which is exclusive - the last date returned will be smaller then it.
        printfunc : a function to output messages, which should handle `end` and `level` arguments.
        """
        self.ticker = ticker
        self.date_start = pd.to_datetime(date_start)
        self.date_end_ex = pd.to_datetime(date_end_ex)
        self.printfunc = printfunc

        self._sec_start_ = self._getYahooSec(self.date_start)
        self._sec_end_ = self._getYahooSec(self.date_end_ex)
        # Yahoo EOD url
        s_head = 'https://query1.finance.yahoo.com/v7/finance/download/'
        s_tail = '&interval=1d&events=history&includeAdjustedClose=true'
        self._url_eod_ = s_head+self.ticker+'?period1='+str(self._sec_start_)+'&period2='+str(self._sec_end_)+s_tail
        # Yahoo dividend url
        s_head = 'https://query1.finance.yahoo.com/v7/finance/download/'
        s_tail = '&interval=1d&events=div&includeAdjustedClose=true'
        self._url_dividend_ = s_head+self.ticker+'?period1='+str(self._sec_start_)+'&period2='+str(self._sec_end_)+s_tail

        self._file_save_last_ = '' # last saved file name

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

    def _queryYahooData(self, url, name, map_cols={}, save=False, file_save='') -> pd.DataFrame:
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
        save: bool
            If True, save the queried data to a csv specified by `file_save`.
        file_save: str
            The csv file to save the queried data.

        Returns
        -------
        pandas.DataFrame
            The queried data.
        """
        is_data_returned = False # initialize flag
        try:
            df = pd.read_csv(url).sort_values('Date').dropna()
            if len(df)>0:
                if len(map_cols)>0:
                    df.rename(columns=map_cols, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                self.printfunc(f"{self.ticker}: Queried {name} data contains {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
                is_data_returned = True
            else:
                self.printfunc(f"{self.ticker}: No {name} data returned.", level=30)
                df = pd.DataFrame()
        except:
            self.printfunc(f"{self.ticker}: Failed to query {name} data!", level=40)
            df = pd.DataFrame()
        
        if is_data_returned and save:
            if file_save == '':
                self._file_save_last_ = f'data/{self.ticker}_{name}.csv.gz'
            else:
                self._file_save_last_ = file_save

            if os.path.exists(self._file_save_last_):
                self.printfunc(f"{self.ticker}: Existing {name} data file found at {self._file_save_last_}.")
                # amend existing data file
                df0 = pd.read_csv(self._file_save_last_)
                df0['date'] = pd.to_datetime(df0['date'])
                self.printfunc(f"{self.ticker}: Existing {name} data file contains {len(df0)} rows over {df0['date'].nunique()} dates from {df0['date'].min().date()} to {df0['date'].max().date()}.")

                index_select = (df0['date']<self.date_start) | (df0['date']>=self.date_end_ex)
                df0 = pd.concat([df0[index_select], df]).sort_values('date').reset_index(drop=True)
                self.printfunc(f"{self.ticker}: Amended {name} data file contains {len(df0)} rows over {df0['date'].nunique()} dates from {df0['date'].min().date()} to {df0['date'].max().date()}.")
                df0.to_csv(self._file_save_last_, index=False, compression='gzip')
            else:
                self.printfunc(f"{self.ticker}: Save {name} data at {self._file_save_last_}.")
                df.to_csv(self._file_save_last_, index=False, compression='gzip')

        return df

    def queryEOD(self, save=False, file_save='') -> pd.DataFrame:
        """
        Function to query Yahoo EOD data.

        Parameters
        ----------
        save: bool
            If True, save the queried data to a csv specified by `file_save`.
            If the csv file already exists, amend the existing file with the queried data.
        file_save: str
            The csv file to save the queried data. Default to 'data/<ticker>_EOD.csv.gz'.
            
        Returns
        -------
        pandas.DataFrame
            The queried data.
        """
        df = self._queryYahooData(
            url = self._url_eod_,
            name = 'EOD',
            map_cols={
                'Date':'date',
                'Open':'open', 'High': 'high', 'Low':'low', 'Close':'close',
                'Adj Close':'close_adj',
                'Volume':'volume'
                },
            save=save,
            file_save=file_save
            )
        return df

    def queryDividend(self, save=False, file_save=''):
        """
        Function to query Yahoo dividend data.

        Parameters
        ----------
        save: bool
            If True, save the queried data to a csv specified by `file_save`.
            If the csv file already exists, amend the existing file with the queried data.
        file_save: str
            The csv file to save the queried data. Default to 'data/<ticker>_dividend.csv.gz'.
            
        Returns
        -------
        pandas.DataFrame
            The queried data.
        """
        df = self._queryYahooData(
            url = self._url_dividend_,
            name = 'dividend',
            map_cols={'Date':'date', 'Dividends':'dividend'}  ,
            save=save,
            file_save=file_save
            )
        return df
