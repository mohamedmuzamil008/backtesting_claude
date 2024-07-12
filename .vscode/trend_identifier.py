import pandas as pd

data = pd.read_csv('./data/ACC_daily.csv', parse_dates=['datetime'])

import pandas as pd
import pandas_ta as ta

import pandas as pd
import pandas_ta as ta

# Assuming 'data' is your cleaned DataFrame with a 'Close' column for stock prices

# Calculate technical indicators for different time periods
periods = [3, 6, 9]

for period in periods:
    data[f'Williams %R_{period}'] = ta.willr(data['High'], data['Low'], data['Close'], lbperiod=14, length=period)
    data[f'Rate of Change_{period}'] = ta.roc(data['Close'], length=period)
    data[f'Momentum_{period}'] = ta.mom(data['Close'], length=period)
    data[f'RSI_{period}'] = ta.rsi(data['Close'], length=14, length_rsi=period)
    data[f'CCI_{period}'] = ta.cci(data['Close'], data['High'], data['Low'], length=20, length_cci=period)
    adx_df = ta.adx(data['High'], data['Low'], data['Close'], length=14, length_adx=period)
    adx_column_name = adx_df.filter(like='ADX_').columns[0]  # Get the column name containing 'ADX'
    data[f'ADX_{period}'] = adx_df[adx_column_name]
    trix_df = ta.trix(data['Close'], length=15, length_trix=period)
    trix_column_name = trix_df.filter(like='TRIX_').columns[0]  # Get the column name containing 'TRIX'
    data[f'TRIX_{period}'] = trix_df[trix_column_name]
    data[f'MACD_{period}'] = ta.macd(data['Close'], length_macd=period).iloc[:, 0]
    data[f'OBV_{period}'] = ta.obv(data['Close'], data['Volume'], length=period)

    # Linear regression estimate with different time periods
    data[f'Linear Reg_{period}'] = data['Close'].rolling(window=period).apply(lambda x: pd.Series(x).interpolate().bfill().iloc[-1])

    # Average True Range
    data[f'ATR_{period}'] = ta.atr(data['High'], data['Low'], data['Close'], length=14, length_atr=period)

# Shift data to align with the target variable (next 3-day average closing price)
data = data.shift(1)

# Calculate the target variable (next 3-day average closing price)
data['Next_3_Day_Avg'] = data['Close'].rolling(window=3).mean().shift(-3)

# Drop any remaining NaN values
data.dropna(inplace=True)

# Define the target label
data['Target'] = (data['Next_3_Day_Avg'] > data['Close']).astype(int)
data['Target'] = data['Target'].replace({0: 'Down', 1: 'Up'})

data_2 = data.drop(columns=['Target', 'Next_3_Day_Avg'], axis=1)
print(data.columns)
print(data_2.columns)

''' Preprocessing starts here'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, date_col):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col], format='%d-%m-%Y').view('int64') // (10**9 * 60 * 60 * 24)
        return X_

# Separate numeric and non-numeric features
numeric_features = data_2.select_dtypes(include=['float64', 'int64']).columns
non_numeric_features = data_2.select_dtypes(exclude=['float64', 'int64']).columns

# Encode non-numeric features
numeric_transformer = 'passthrough'
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
date_transformer = DateEncoder('datetime')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, non_numeric_features.drop('datetime')),
        ('date', date_transformer, ['datetime'])
    ])

# Apply the preprocessing
X = preprocessor.fit_transform(data_2)
#X = data.drop(['Close', 'Target'], axis=1)
y = data['Target']

'''''Feature Selection Starts here'''
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the SVM classifier
svm = SVC(kernel='linear')

# Create the RFECV object
rfecv = RFECV(estimator=svm, step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)

# Fit the RFECV on the training data
rfecv.fit(X_train, y_train)

# Print the ranking of features
print("Ranking of features:")
print(sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), X_train.columns)))

# Select the top 40% of features
num_features_to_select = int(0.4 * X_train.shape[1])
selected_features = X_train.columns[rfecv.ranking_ <= num_features_to_select]

# Create the new feature matrix with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train the SVM classifier on the selected features
svm.fit(X_train_selected, y_train)

# Evaluate the model on the test set
accuracy = svm.score(X_test_selected, y_test)
print(f"Accuracy on test set with selected features: {accuracy:.4f}")





