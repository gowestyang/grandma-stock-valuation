"""
Utilities for logging.
"""

import os
import logging
import pandas as pd
from datetime import datetime


simple_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


class DefaultLogger():
    """
    Class of a logger with default setups.
    """
    def __init__(self, hdlr, formatter=simple_formatter, **kwargs):
        """
        Initialize a logger.

        Parameters
        ----------
        hdlr : logging.Handler
            Handler of the log.
        formatter : logging.Formatter
            Formatter of the handler.
        **kwargs :
            Additional key-word arguments passed to `hdlr`.
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.addHandler(hdlr=hdlr, formatter=formatter, **kwargs)

    def addHandler(self, hdlr, formatter=simple_formatter, **kwargs):
        """
        Add a handler to the logger.
        """
        hdlr = hdlr(**kwargs)
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    def logPandas(self, *args, level=logging.INFO) -> None:
        """
        Log with additional formatting for pandas Series and DataFrame.
        
        Parameters
        ----------
        *args : str | pandas.Series | pandas.DataFrame
            The messages to be logged.
        level : int
            Level of the log message. 10 = DEBUG; 20 = INFO; 30 = WARNING; 40 = ERROR; 50 = CRITICAL.
        """
        messages = args
        messages = ['\n'+msg.to_string()+'\n' if type(msg) in [pd.Series, pd.DataFrame] else msg for msg in messages]
        msg = ' '.join(messages)
        self.logger.log(level, msg)
    
    def log_pandas(self, msg, level=logging.INFO) -> None:
        """
        To be depreciated. Please use `logPandas()`.
        """
        if type(msg) in [pd.Series, pd.DataFrame]: msg = '\n' + msg.to_string()
        self.logger.log(level, msg)


class FileLogger(DefaultLogger):
    """
    Class of a logger which writes to a log file and print on screen.
    """
    def __init__(self, formatter=simple_formatter, log_file=None, default_folder='_log', append=False) -> None:
        """
        Initialize the file-screen logger.

        Parameters
        ----------
        formmater : logging.Formmater
            Formatter to apply.
        log_file : str
            Path to the log file. If None, a log file will be created under the `default_folder`.
        default_folder: str
            Default folder for the log files.
        append : bool
            If True, append to the existing log file. If False, start from an empty log file.
        """
        if log_file is None:
            if not os.path.exists(default_folder):
                os.mkdir(default_folder)
            date_log = datetime.today().strftime("%Y%m%d")
            self.log_file = os.path.join(default_folder, date_log+'.log')
        else:
            self.log_file = log_file

        super().__init__(hdlr=logging.FileHandler, formatter=formatter, filename=self.log_file)

        self.addHandler(hdlr=logging.StreamHandler, formatter=formatter)

        if not append:
            with open(self.log_file, 'w'):
                pass

    def clearLog(self) -> None:
        """
        Clear the log file.
        """
        with open(self.log_file, 'w'):
            pass
