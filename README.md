# Grandma Stock Valuation (WIP)

"Grandma-style" traders look at only the historical prices, buy when the price is low, and sell when it is high.

<br>

This repo is to quantify this simple idea
* Run a linear regression on historical (10 years) daily close prices, optionally without the recent period (12 months)
    * Remove extreme values based on residuals
    * Run linear regression again without extreme values
    * Estimate average **annualized gain %**
* Estimate the current **fair value** based on the fitted regression model
* Compare the current price against the estimated fair value
    * If current price > fair value, the stock is over-valued
        * Estimate "years-over-values" as *current price / fair value / annualized gain %*
    * If current price < fair value, the stock is under-valued

<br>

Note that the model is most suitable for **broad ETF**, such as
* iShares Core S&P 500 ETF (IVV)
* SPDR FTSE Greater China ETF (3073.HK)
* iShares Europe ETF (IEV)
* Global X FTSE Southeast Asia ETF (ASEA)
* Vanguard FTSE Pacific Fund ETF (VPL)

<br>

With the valuation, a portfolio can be constructed, which gives more weights to the under-valued stocks.
