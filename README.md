# Grandma Stock Valuation (WIP)

"Grandma-style" traders look at only the historical prices, buy when the price is low, and sell when it is high.

<br>

The Grandma Stock Valuation model enhanced this simple idea with qualitative considerations:
* Type of curves to fit historical prices
* Outliers identification
* Evaluate *fair price*, *over-valued %*, and *over-valued years*

In addition, the model includes a portfolio constructor, which suggests the portfolio allocation of your selected group of instruments.<br>
The portfolio constructor allows several adjustments, with default values recommanded:
* Sensitivity to over/under valuation
* Compensation to number of instruments

<br>

Important Notes
* The model is most suitable for **broad ETF**, such as S&P 500 ETF.
* The portfolio constructor only works with **instruments with sufficient positive return** (suggest > 1% historical annualized return).
