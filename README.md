# Analyst_Rating

## The Optimal Model: Machine-Learning Based Sell-Side Equity Research

![Intro Image](read_me_images/intro_image.png)

Indeed, no one has a magic crystal ball that can predict the market with 100% accuracy. However, this does not stop the billion-dollar industry of active money managers who attempt to beat the benchmark by executing on a particular view in the market. These firms, like quant hedge funds, multi-manager funds, fundamental/quantamental funds, etc., rely on equity research that is distributed by major investment banks on the street. 

At almost every major investment bank, there is an equity research division, where an Analyst is responsible for covering a small universe of say 10 stocks in a particular sector. Currently, they rely primarily on fundamentals and traditional financial modeling - something we feel is now antiquated in the age of Machine Learning. 

We built this project to challenge this notion; to build a classification model that can predict the rating of a stock (BUY/SELL/HOLD) to a greater degree of accuracy than fundamentals could. Essentially, we built a classification model that replaces the sell-side equity analyst covering a particular stock.

We take a look at multiple factors within four categories: Technical, Fundamental, Macro, and News Sentiment, and after building a dataset containing these factors, we look to see if these are drivers for Price Action. Our goal is to construct a successful Predicted Model utilizing `RandomForest`, `LogisticalRegression`, and `Neural Network`. 

## Usage: 

> Note: `analyst_rating` is found in main folder. `sentiment_analysis_data` is found the Analysis Folder. The three models can be found in the Machine Learning Folder.

Import the following dependencies and libraries: 

### Analyst Rating: 
```python
import pandas as pd
from pathlib import Path
import yfinance as yf
from sklearn.impute import SimpleImputer
import numpy as np
```

### Sentiment Analysis: 
```python
from alpaca_trade_api import REST, Stream
from transformers import pipeline
import pandas as pd
from pathlib import Path
import numpy as np
```

### Neural Network Model: 
```python
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

### Logistic Regression Model: 
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

### RandomForest Model: 
```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Read and open `analyst_rating.csv` to the three models (Neural Network, RandomForest, LogisticRegression) 
```python
pd.read_csv("Resources/dataset.csv", index_col="2021 - 2023 AAPL Daily Data (Index)", infer_datetime_format=True, parse_dates=True)
```

## Usage for `analyst_rating.ipynb`:

Split the data by quarterly time ranges. This will make it easier to combine fundamental data. 

For example: 

```python
# Pull aapl ticker with date range "03/32/2023" to "07/03/2023"
aapl_1 = yf.download("AAPL", start='2023-03-31', end='2023-07-03', interval="1d")

aapl_1['Total Debt'] = aapl_debt.loc['2023-03-31']
aapl_1['Shares'] = aapl_shares.loc['2023-03-31']
aapl_1['Cash'] = aapl_cash.loc['2023-03-31']
aapl_1['EPS'] = 1.53
aapl_1['EBITDA'] = 31260000000
```

> Note: EPS and EBITDA were hard-coded into the data frame. You can view these values in Yahoo Finance. 


Make sure to replace string values provided in the `cashflow` and `balancesheet` data. The only way the data frame can recognize all data is to transform them to float64 values.

```python
# Begin to clean strings dtypes by replacing commas 
aapl_df['Total Debt']= aapl_df['Total Debt'].str.replace(',','')
aapl_df['Shares']= aapl_df['Shares'].str.replace(',','')
aapl_df['Cash']= aapl_df['Cash'].str.replace(',','')

# Change all columns to float64
aapl_df = aapl_df.astype(float)
```

For the missing values (NaN) found when finding the moving averages, use `SimpleImputer` to replace NaN values to the mean.

```python
# Replace NaN values with SimpleImputer. Fit and Transform this data.
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
aapl_df = imp_mean.fit_transform(aapl_df)

# Make aapl_df as a DataFrame and review first 5 rows in data
aapl_df = pd.DataFrame(aapl_df)
aapl_df.head()
```

In Part 6, we create 'Buy' and 'Sell' Signals. When importing data and running the ML Models, change the string to float64 values to be able to run the analysis properly. This is the Y Target. 


```python
aapl_df['Recommendation'] = aapl_df['Recommendation'].replace({'Buy': 1, 'Sell': -1})
```

## Sentiment Analysis Usage: 

Run `pipeline` as a natural language processing predicted model to be able to fit and transform a 'Positive or Negative' news and confidence rating of this news. 

```python
# Call pipeline to run `sentiment-analysis`
classifier = pipeline('sentiment-analysis')
```

Running a for loop will allow you to scan through the news articles and report date, news, and result. 

```python
result1 = []
date1=[]

for story in news1:
    # print(story.headline)
    score = classifier(story.summary)
    result1.append(score)
    time = story.created_at
    date1.append(time)
```

Create an export .csv file to be able to import this data to analyst_rating, which is the main dataset script. 

```python
## Export to .csv
ch3.to_csv('sentiment_analysis.csv')
```
## Dataset Building: 

Now that we have built out the primary dataset, we can begin to use Machine Learning to build a predictive classification model.

![Intro Image](read_me_images/dataset.png)

## RandomForest Model: 

![Intro Image](read_me_images/random_forest.png)

Our first attempt the predictors using Apples P/E ratio, and 50 and 200 day MA, and included other variables of 1 and 5 year treasury yields and a news score based on sentiment. Running this model resulted in a 57% Precision Score. The second Random Forest attempt using 23 years of price data with predictors of daily Apple; volume, open, high, low and close prices resulted in a 55% Precision score.

```
#Imprt Random Forest
from sklearn.ensemble import RandomForestClassifier
```
```
#initial model, n_estimators = number of decision trees, min_sample_spit protects from overfitting, 
#random_state =1 to be able to re-run model with same results. train = -100 is all rows except last 100 rows.
model = RandomForestClassifier(n_estimators=175, min_samples_split=100, random_state=1)
train = data.iloc[:-100]
test = data.iloc[-100:]
#predictors used
predictors = ["Close", "Open", "High", "Low", "Volume"]
model.fit(train[predictors], train["Signal"])
```

```
#create prediction function to put everythin into one function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Signal"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Signal"], preds], axis=1)
    return combined
```

```
#Backtest function that considers the AAPL data, the ML Model, predictors, a start value and step value (15 years and yearly)  
def backtest(data, model, predictors, start=3750, step=250):
    all_predictions = []
    
    #create function to loop through data year by year
    #train year is all years prior to the current year, test set is current year and then concat all prodiction together
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
```
Improve the model with extra predictor columns using rolling moving averages using weekly, monthly, quarterly, yearly, and 2 years.
compare ratios of todays closing price vs calculated averages
create new columns for new predictions
```
horizons = [2,5,60,250,800]
new_predictors = []

for horizon in horizons:
    rolling_averages = data.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"    
    data[ratio_column] = data["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum() ["Signal"]
    
    new_predictors += [ratio_column, trend_column]
```
insert control to what becomes one and what becomes a zero by adding the probability to 55% to increase the probability 
when we say the price will go up we are more confident which will reduce the number of trading days.+
```
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Signal"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .55] = 1
    preds[preds < .55] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Signal"], preds], axis=1)
    return combined
```

New precision score
```
precision_score(predictions["Signal"], predictions["Predictions"])
```

Plot the closing price history
```
data.plot.line(y="Close", use_index=True)
```

## LogisticRegression Model: 

![Intro Image](read_me_images/logistic_regression.png)

Using 'EBIDTA', 'EV/EBITDA', 'EPS', 'P/E', '1 YR', '5 YR', '30 YR', '50 MA', '200 MA', 'News', 'News Score', 'Percent Change', 'Target - B/H/S (based on close - daily % change). We had to change the news and target columns  to numerical values  so that the model would be able to accept the data.

```
# Select features and target variable
X = df.drop(columns=["Target - B/H/S (based on close - daily % change)"])
y = df["Target - B/H/S (based on close - daily % change)"]

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create a pipeline for preprocessing and modeling
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Define the hyperparameters grid for grid search
param_grid = {
    "model__C": [0.1, 1.0, 10.0],
    "model__penalty": ["l1", "l2"],
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model on training data
y_train_pred = best_model.predict(X_train)
train_report = classification_report(y_train, y_train_pred)
print("Training Report:")
print(train_report)

# Evaluate the model on testing data
y_test_pred = best_model.predict(X_test)
test_report = classification_report(y_test, y_test_pred)
print("Testing Report:")
print(test_report)
```

## NeuralNetwork Model: 

![Intro Image](read_me_images/nn.png)

After backtesting and manual optimization, we found the neural network to be less than ideal for predicting the correct BUY or SELL classification. We utilized 2 models. The first model yielded an accuracy score of 0.4701, while the second model gave 0.5203 accuracy score. The difference between the first two models can be found within the hyperparameter tuning. For example, we changed the loss function from Categorical Cross-Entropy to Mean Squared Error, number of hidden nodes from 10 to 20, number of neurons form 2 to 3, and optimizer function from sigmoid to adam. 

Model 1
```
# Compile the Sequential model
nn.compile(loss="binary_crossentropy", optimizer="sigmoid", metrics=["accuracy"])
fit_model = nn.fit(X_train_scaled, y_train, epochs = 50)
```
Model 2
```
# Compile the Sequential model
nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
fit_model = nn.fit(X_train_scaled, y_train, epochs = 50)
```
## Results - Our best model: 

![Intro Image](read_me_images/logistic_regression_results.png)

Our best model was the Logistic Regression Model. This was in line with our expectations, given the formatting of our dataset and the fact that this is ultimately a classification problem. Please see the image outlining the precision, recall, F1 score, support, and accuracy score of the model.

## Next Steps 

![Intro Image](read_me_images/next_steps.png)

We found this classification problem to be very interesting, and found that there were many opportunities to further enhance our model:

* Further Feature engineering and selection of factors for all three ML models; for example, utilizing more novel factors like alternative data, or utilizing a wider variety of fundamental/technical/macro factors to better improve the predictive accuracy of the model
* More hyperparameter tuning, specifically with the Neural Network. Experimenting with different optimizers, loss, activation functions, and number of epochs 
* Taking into account the street consensus - and maybe picking a different stock - AAPL has gone up consistently for 8 years - so maybe better to just hold instead of trade this stock - or pick a more volatile stock
* Building a larger dataset with 10+ years of stock data instead of 2 years to capture economic cycles. 

## SOURCES:

* https://www.youtube.com/watch?v=gfwNK3o45ng
* https://towardsdatascience.com/is-it-possible-to-predict-stock-prices-with-a-neural-network-d750af3de50b
