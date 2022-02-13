# Grandma Stock Valuation (GSV)

"Grandma-style" traders look at only the historical prices, buy when the price is low, and sell when it is high.

<br>

The **Grandma Stock Valuation** model enhanced this simple idea with a number of qualitative considerations:
* Type of curves to fit on historical prices
* Outliers identification
* Evaluate *fair price*, *over-valued %*, and *over-valued years*

In addition, the project includes a **portfolio allocator**, which suggests the portfolio allocation of your selected group of instruments.<br>
It allocates larger proportion to more under-valued instruments, with several configurations:
* Sensitivity to over/under valuation
* Compensation to the number of instruments

<br>

Important Notes
* The valuation model is most suitable for **broad ETF**, such as S&P 500 ETF.
* The portfolio allocator only works with **instruments with sufficient positive return** (suggest > 1% historical annualized return).
