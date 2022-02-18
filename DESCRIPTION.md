# Grandma Stock Valuation (GSV)
*A simple, manageable valuation tool and portfolio builder for retail investors - even grandma can use it!*
<br>
* Valuate instrument with historical trends - no sophiscated technical analysis
* Build a flexible portfolio with your personal interest
* Trade at your own pace - you can even trade only once per month/quarter
* Back your investment decision with firm numbers - no more frustration from all the media noise!

<br>

## Introduction
"Grandma-style" traders look at only the historical daily prices - buy when the price is low - sell when the price is high.

<br>

The **Grandma Stock Valuation** model enhanced this simple idea with a number of qualitative considerations:
* Type of curves to fit on historical daily prices
* Outliers identification
* Evaluate *fair price*, *over-valued %*, and *over-valued years*

In addition, the project includes a **portfolio allocator**, which suggests the portfolio allocation of your selected group of instruments.<br>
It allocates larger proportion to more under-valued instruments, with several configurations:
* Sensitivity to over/under valuation
* Compensation to the number of instruments
* Option to include cash as part of the portfolio, in order to realize profit

<br>

**Important Notes**
* The valuation model is most suitable for **broad ETF**, such as S&P 500 ETF.
* The portfolio allocator only works with **instruments with sufficient positive return** (suggest > 1% historical annualized return).

<br>

## Examples
Please visit the [project page](https://github.com/gowestyang/grandma-stock-valuation) for detailed information and examples.

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



