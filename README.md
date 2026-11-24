# used-car-regression

Can a used car dealer predict what a car will sell for on its lot well enough to decide which cars to buy at auction, and what is that worth compared with how it buys today? This repo fits a linear regression to 240 past sales, checks it on cars it never saw, and tests the buy/pass rule on 40 auction cars.

## Run

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Mac/Linux: .venv/bin/pip
.venv\Scripts\python analysis.py                # prints every result, writes outputs/ and outputs/charts/
```

## Files

| file | what it does |
|---|---|
| `data/used_car_sales.csv` | 240 cars the dealer sold in the last year (the assignment's data) |
| `data/auction_lots.csv` | 40 auction cars with the winning bid and the later sale price, used only to test the buying rule |
| `analysis.py` | 70/30 split, full and reduced regressions, baseline vs v1 on the test cars, `decide()`, auction results, charts |
| `outputs/` | coefficients, test fit, test predictions, lot-by-lot decisions, auction results, model summaries, console output |
| `outputs/charts/` | predicted vs actual on the test cars, auction results, which lots each rule buys |

## Choices

- **Split.** 72 of the 240 cars (30%) held out at random as a test set. Seed 42.
- **Model.** Linear regression of sale price on mileage, age, horsepower, accident and number of owners, fit on the 168 training cars.
- **Dropped feature.** Number of owners had a p-value of 0.71. Without it, adjusted R² went from 0.856 to 0.857, so v1 leaves it out.
- **Baseline.** The training-set average sale price, $25,875, for every car.
- **Prep cost.** $1,500 per car to get it on the lot.

## Results

v1 coefficients: each 1,000 miles lowers the price by about $98, each year of age by $1,038, and an accident by $2,016. Each extra horsepower adds about $70.

On the 72 test cars:

| | R² | typical error (RMSE) |
|---|---|---|
| baseline: average price | -0.01 | $7,345 |
| v1 | 0.86 | $2,742 |

v1's adjusted R² on the training cars is 0.857.

On the 40 auction lots:

| rule | cars bought | money-losing cars bought | total profit |
|---|---|---|---|
| baseline | 14 | 10 | -$6,000 |
| v1 | 18 | 8 | $10,700 |

Buying all 40 cars would have lost $35,500.

AI use: an AI coding assistant helped write the code, including the chart code. I ran every script myself, so the charts and numbers come from running this code.
