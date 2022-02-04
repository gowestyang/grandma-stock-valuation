"""
Utilities for logging.
"""

import pandas as pd
from datetime import datetime
import logging

logging.addLevelName(39, 'ERR')
logging.ERR = 39

class StandardLogger():

    def __init__(self, path_log, append=False) -> None:
        """
        Initialize logger.

        Parameters
        ----------
        path_log : str
            Path to the log file. It will be appended with '_yyyymmdd.log' to form the full log file string.
        append : bool, default to False
            If True, append to the existing log file. If False, start from an empty log file.
        """
        self.date_log = datetime.today().strftime("%Y%m%d")
        self.file_log = path_log + '_' + self.date_log + '.log'
        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.hdlr_log = logging.FileHandler(self.file_log)
        self.hdlr_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.hdlr_log)
        self.logger.setLevel(logging.INFO)
        if not append:
            with open(self.file_log, 'w'):
                pass

    def clearLog(self) -> None:
        """
        Clear the log file.
        """
        with open(self.file_log, 'w'):
            pass

    def logPrint(self, msg, level=logging.INFO, end='\n') -> None:
        """
        Write to the log, as well as print on screen.
        
        Parameters
        ----------
        msg : str | pandas.Series | pandas.DataFrame
            The message to be logged.
        level : int
            Level of the log message. 10 = DEBUG; 20 = INFO; 30 = WARNING; 40 = ERROR; 50 = CRITICAL.
        end : str
            The end character to be passed to print().
        """
        if type(msg) in [pd.Series, pd.DataFrame]: msg = '\n' + msg.to_string()
        self.logger.log(level, msg)
        print(msg, end=end)
        self.hdlr_log.close()
