# Grandma Stock Valuation
*A simple, manageable valuation tool and portfolio builder for retail investors - even grandma can use it!*
<br>
* Valuate instrument with historical trend - no sophiscated technical analysis
* Build a flexible portfolio with your personal interest
* Invest at your own pace - you can even trade only once per month/quarter
* Back your investment decision with firm numbers - no more frustration from all the media noise!

<br>

Please kindly star:star: if you found the project interest:blush:

<br>

* [Introduction](#introduction)
    * [Grandma Stock Valuation Model](#grandma-stock-valuation-model)
    * [Grandma Portfolio Allocation](#grandma-portfolio-allocation)
    * [Yahoo Data Loader](#yahoo-data-loader)
* [Documentation and Examples](#documentation-and-examples)
* [Installation](#installation)

<br>

## Introduction

### Grandma Stock Valuation Model

"Grandma-style" traders look at only the historical daily prices - buy when the price is low - sell when the price is high.

The **Grandma Stock Valuation (GSV)** model automated this simple idea, with a number of enhancements:
* Derive price trend based on historical daily price
* Identify extreme historical periods as outliers
* Evaluate *fair price*, *over-valued %*, and *over-valued years*

Note that the model is most suitable for **broad ETF** (country / region level ETF), such as S&P 500.

<br>

### Grandma Portfolio Allocation

Give a group of your interested stocks, the **Grandma Portfolio Allocation (GPA)** suggests how much % of your budget to be allocated to each stock.<br>
It allocates larger proportion to more under-valued stocks, with additional capabilities:
* Sensitivity to over/under valuation
* Weight by both correlation and valuation
* Compensation to the number of instruments
* Include cash as part of the portfolio, in order to realize profit

Please note that
* The portfolio allocator only works with **instruments with sufficient positive return** (suggest > 1% historical annualized return).
* The portfolio allocator does not take **exchange rate** into consideration. If needed, you will need to adjust the data by yourself.

<br>

### Yahoo Data Loader

The package also includes a data loader to query daily data from Yahoo Finance - for free.

<br>

## Documentation and Examples

Please refer to https://github.com/gowestyang/grandma-stock-valuation/tree/main/doc for detailed documentations and examples:
* Step-by-step guide to use the package, with python scripts and package data.
    * Query data from Yahoo Finance
    * Valuate stocks with Grandma Stock Valuation model, and visualize the result
    * Construct a portfolio with Grandma Portfolio Allocation
    * Back-test (WIP)
* Detailed explaination to the parameters, math and design considerations.

<br>

## Installation
### Dependencies
grandma-stock-valuation requires
* Python (>=3.7)
* NumPy
* pandas
* scikit-learn
* Plotly
* kaleido

<br>

### User Installation
If you already have the dependencies installed, you can install grandma-stock-valuation using `pip`:

    pip install -U grandma-stock-valuation
