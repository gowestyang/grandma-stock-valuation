# Author: Yang Xi
# Date Created: 2019-02-07
# Date Modified: 2020-07-03 

import pandas as pd
from datetime import datetime
import logging

logging.addLevelName(39, 'ERR')
logging.ERR = 39

class StandardLogger():
    def __init__(self, pathLog, append=False):
        self.dateLog = datetime.today().strftime("%Y%m%d")
        self.logFile = pathLog + '_' + self.dateLog + '.log'
        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.logHdlr = logging.FileHandler(self.logFile)
        self.logHdlr.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.logHdlr)
        self.logger.setLevel(logging.INFO)
        if not append:
            with open(self.logFile, 'w'):
                pass

    def clearLog(self):
        with open(self.logFile, 'w'):
            pass

    def logPrint(self, msg, level=logging.INFO, end='\n'):
        if type(msg) in [pd.Series, pd.DataFrame]: msg = '\n' + msg.to_string()
        self.logger.log(level, msg)
        print(msg, end=end)
        self.logHdlr.close()
