from grandma_stock_valuation import FileLogger, loadPacakgeData
from grandma_stock_valuation import batchValuation, addCashPortfolio, getCorrelationWeight, allocatePortfolio

# Refer to example_0_FileLogger.ipynb for details of the FileLogger.
logger = FileLogger()
logPrint = logger.logPandas


d_instrument_data, d_instrument = loadPacakgeData(verbose=2)

#### TEMP ####
l_select_inst = ['EEMA', 'VPL'] # , 'IVV', 

d_instrument_data = {k:v for k,v in d_instrument_data.items() if k in l_select_inst}

logPrint("Keys of d_instrument_data:", str(d_instrument_data.keys()))

from grandma_stock_valuation.back_test import GrandmaBackTester

backtester = GrandmaBackTester(
    backtest_years=10,
    adjust_freq_months=3,
    init_parameters={'recent_months':0, 'train_years':10, 'min_train_years':5, 'date_end':None},
    fit_parameters={'price_col':'close_adj', 'log':True, 'n_std':1.5},
    valuate_parameters={'min_annual_return':0.01},
    allocation_parameters={'transformation':'sigmoid', 'scale':1},
    with_cash=False,
    with_correlation_weights=True,
    verbose=1,
    printfunc=logPrint
)

backtester.runBackTest(d_instrument_data)

backtester.df_average_value

