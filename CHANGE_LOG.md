# Change Log

*Version 0.2.2*
* Fixed a bug where `plotTrendline()` resulted in failed comparison.

*Version 0.2.1*
* Fixed a bug where `allocatePortfolio()` failed assertion when `weights=None`.

*Version 0.2.0*
* Fixed a bug where `getCorrelationWeight()` resulted in exception when the input data is empty.
* Fixed a bug where `allocatePortfolio()` failed when input valuation array contains nan.
* Fixed a bug where `allocatePortfolio()` failed when input valuation array is empty.
* Added an option in `batchValuation()` to not generate price charts.

*Version 0.1.0*
* Fixed a bug where `date_end` had no effect in `GrandmaStockValuation`.
* Added `loadPacakgeData()` function to load package data.
* Added `getCorrelationWeight()` function to calculate correlation weight.

<br>

*Version 0.0.4*
* Initial alpha release

