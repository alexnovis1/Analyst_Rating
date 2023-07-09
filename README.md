# Analyst_Rating

## Introduction: 

Taking four categories: Technical, Fundamental, Macro, and News Sentiment, the project takes this data into account as the main drivers for Price Action. Our goal is to construct a successful Predicted Model utilizing `RandomForest`, `LogisticalRegression`, and `Neural Network`. 

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

> Note: EPS and EBITDA were hard coded into the dataframe. You can view these values in Yahoo Finance. 


Make sure to replace string values provided in the `cashflow` and `balancesheet` data. The only way the dataframe can recongnize all data is to transform them to float64 values.

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