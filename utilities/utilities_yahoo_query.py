# Author: Yang Xi
# Date Created: 2021-07-31
# Date Modified: 2021-07-31 

import pandas as pd

# Function to convert datetime to yahoo format
def GetYahooSec(dt):
    return (dt-pd.to_datetime('1970-01-01')).days*24*3600

# Function to query Yahoo EOD data
# Yahoo: Close price adjusted for splits.
# Yahoo: Adjusted close price adjusted for both dividends and splits.
def QueryYahooEOD(ticker, dateStart, dateEndEx, printfunc):
    dateStart = pd.to_datetime(dateStart)
    dateEndEx = pd.to_datetime(dateEndEx) # exclusive
    secStart = GetYahooSec(dateStart)
    secEnd = GetYahooSec(dateEndEx)
    printfunc(f"{ticker}: To query EOD data from {dateStart}({secStart}) to {dateEndEx}({secEnd}).")

    def GetYahooEodUrl(ticker, secStart, secEnd):
        sHead = 'https://query1.finance.yahoo.com/v7/finance/download/'
        sTail = '&interval=1d&events=history&includeAdjustedClose=true'
        return sHead+ticker+'?period1='+str(secStart)+'&period2='+str(secEnd)+sTail
    urlYahoo = GetYahooEodUrl(ticker, secStart, secEnd)

    try:
        df = pd.read_csv(urlYahoo).sort_values('Date').dropna()
        if len(df)>0:
            mapCols = {'Date':'date', 'Open':'open', 'High': 'high', 'Low':'low', 'Close':'close', 'Adj Close':'close_adj', 'Volume':'volume'}
            df.rename(columns=mapCols, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            printfunc(f"{ticker}: Queried EOD data contains {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
        else:
            printfunc(f"{ticker}: No EOD data queried.", level=30)
            df = pd.DataFrame()
    except:
        printfunc(f"{ticker}: Failed to query EOD data !", level=40)
        df = pd.DataFrame()
    
    return df


# Function to query Yahoo dividend data
def QueryYahooDividend(ticker, dateStart, dateEndEx, printfunc):
    dateStart = pd.to_datetime(dateStart)
    dateEndEx = pd.to_datetime(dateEndEx) # exclusive
    secStart = GetYahooSec(dateStart)
    secEnd = GetYahooSec(dateEndEx)
    printfunc(f"{ticker}: To query dividend data from {dateStart}({secStart}) to {dateEndEx}({secEnd}).")

    def GetYahooDividendUrl(ticker, secStart, secEnd):
        sHead = 'https://query1.finance.yahoo.com/v7/finance/download/'
        sTail = '&interval=1d&events=div&includeAdjustedClose=true'
        return sHead+ticker+'?period1='+str(secStart)+'&period2='+str(secEnd)+sTail
    urlYahoo = GetYahooDividendUrl(ticker, secStart, secEnd)

    try:
        df = pd.read_csv(urlYahoo).sort_values('Date').dropna()
        if len(df)>0:
            mapCols = {'Date':'date', 'Dividends':'dividend'}    
            df.rename(columns=mapCols, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            printfunc(f"{ticker}: Queried dividend data contains {len(df)} rows over {df['date'].nunique()} dates from {df['date'].min().date()} to {df['date'].max().date()}.")
        else:
            printfunc(f"{ticker}: No dividend data queried.", level=30)
            df = pd.DataFrame()
    except:
        printfunc(f"{ticker}: Failed to query dividend data !", level=40)
        df = pd.DataFrame()
    
    return df