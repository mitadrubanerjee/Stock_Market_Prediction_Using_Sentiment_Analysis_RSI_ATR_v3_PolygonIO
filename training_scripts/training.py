#Install these packages to start the execution of the file:
#!pip install ta
#!pip install yfinance
#pip install scikit-learn
#python -m pip install nltk
model_path = 'model/model.pkl'
scaler_path = 'model/scaler.pkl'
#---------------------------------------------------------------------------------------------------------
#1) Package Imports and downloads:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
import ta  # Technical Analysis library
import nltk
import re
import gdown
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#pip install scikit-learn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

#https://drive.google.com/file/d/185EBhgGnuM0TuJKgWONGxSp03KS5Nd0Z/view?usp=drive_link

# Download NLTK data files (only need to run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#---------------------------------------------------------------------------------------------------------

# Replace 'your-file-id' with the actual file ID from the shareable link
#https://drive.google.com/file/d/185EBhgGnuM0TuJKgWONGxSp03KS5Nd0Z/view?usp=drive_link
file_id = '185EBhgGnuM0TuJKgWONGxSp03KS5Nd0Z'
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'Combined_News_DJIA.csv', quiet=False)

# Load the downloaded file
df = pd.read_csv('Combined_News_DJIA.csv', encoding='latin1')
df.drop(columns='Label', inplace=True)
news_df = df
#df.head()
#print(len(df))

# Define the start and end dates based on the news data
start_date = '2008-08-08'
end_date = '2016-07-01'

#---------------------------------------------------------------------------------------------------------

# Fetch DJIA data (Extracting the Dow Jones Industrial Average Data from Yahoo Finance)
djia = yf.download('^DJI', start=start_date, end=end_date, interval='1d')

# Reset index to have 'Date' as a column
djia.reset_index(inplace=True)

# Check if columns are MultiIndex
if isinstance(djia.columns, pd.MultiIndex):
    # Flatten MultiIndex columns
    djia.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in djia.columns.values]
    # Rename columns to standard names
    djia.rename(columns={
        'Date_': 'Date',
        'Open_^DJI': 'Open',
        'High_^DJI': 'High',
        'Low_^DJI': 'Low',
        'Close_^DJI': 'Close',
        'Adj Close_^DJI': 'Adj Close',
        'Volume_^DJI': 'Volume'
    }, inplace=True)
else:
    # If columns are already single-level, ensure 'Date' is correct
    djia.rename(columns={'Date': 'Date'}, inplace=True)

# Convert 'Date' column to datetime
djia['Date'] = pd.to_datetime(djia['Date'])

# Set 'Date' as the index
djia.set_index('Date', inplace=True)

# Display the columns and first few rows
print("Columns in djia DataFrame after adjustment:")
print(djia.columns.tolist())
print(djia.head())

# Resample data to weekly frequency starting on Monday
djia_weekly = djia.resample('W-MON').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'Adj Close': 'last'
}).dropna()

# Ensure 'Close' prices are numeric
djia_weekly['Close'] = pd.to_numeric(djia_weekly['Close'])

# Calculate Weekly Price Change (%)
djia_weekly['Weekly_Price_Change_%'] = djia_weekly['Close'].pct_change() * 100
djia_weekly['Weekly_Volume_Change_%'] = djia_weekly['Volume'].pct_change() * 100

# Calculate Relative Strength Index (RSI)
djia_weekly['RSI'] = ta.momentum.RSIIndicator(djia_weekly['Close'], window=14).rsi()

# Calculate Average True Range (ATR)
djia_weekly['ATR'] = ta.volatility.AverageTrueRange(
    high=djia_weekly['High'],
    low=djia_weekly['Low'],
    close=djia_weekly['Close'],
    window=14
).average_true_range()

# Calculate Moving Averages
djia_weekly['MA_5'] = djia_weekly['Close'].rolling(window=5).mean()
djia_weekly['MA_7'] = djia_weekly['Close'].rolling(window=7).mean()
djia_weekly['MA_10'] = djia_weekly['Close'].rolling(window=10).mean()

# Reset index to have 'Date' as a column
djia_weekly.reset_index(inplace=True)

# Rename 'Date' column to 'Week_Start' to align with news data
djia_weekly.rename(columns={'Date': 'Week_Start'}, inplace=True)

# Remove the time component
djia_weekly['Week_Start'] = djia_weekly['Week_Start'].dt.date

djia_weekly.dropna(inplace=True)
djia_weekly.head(5)


#---------------------------------------------------------------------------------------------------------

import pandas as pd
import yfinance as yf

# Define the commodity tickers
commodity_tickers = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Oil': 'CL=F',
    'Natural_Gas': 'NG=F'
}

commodity_dfs = []

# Fetch data for each commodity
for name, ticker in commodity_tickers.items():
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    # Select 'Close' column and rename it to the commodity name
    data = data[['Close']].rename(columns={'Close': name})
    # Ensure the index is named 'Date' for consistency
    data.index.rename('Date', inplace=True)
    # Append to list
    commodity_dfs.append(data)

# Concatenate all commodity DataFrames along columns
commodity_data = pd.concat(commodity_dfs, axis=1)

# Reset index to have 'Date' as a column
commodity_data.reset_index(inplace=True)

# Convert 'Date' column to datetime
commodity_data['Date'] = pd.to_datetime(commodity_data['Date'])

# Set 'Date' as the index
commodity_data.set_index('Date', inplace=True)

# Resample data to weekly frequency starting on Monday
commodity_weekly = commodity_data.resample('W-MON').last().dropna().reset_index()

# Rename 'Date' to 'Week_Start' to align with other datasets
commodity_weekly.rename(columns={'Date': 'Week_Start'}, inplace=True)

# Remove timezone information if present
commodity_weekly['Week_Start'] = commodity_weekly['Week_Start'].dt.tz_localize(None)


if isinstance(commodity_weekly.columns, pd.MultiIndex):
    commodity_weekly.columns = ['_'.join(col).strip() for col in commodity_weekly.columns.values]

# Rename columns
commodity_weekly.rename(columns={
    'Week_Start_': 'Week_Start',
    'Gold_GC=F': 'Gold',
    'Silver_SI=F': 'Silver',
    'Oil_CL=F': 'Oil',
    'Natural_Gas_NG=F': 'Natural_Gas'
}, inplace=True)

# List of commodities
commodities = ['Gold', 'Silver', 'Oil', 'Natural_Gas']

# Calculate weekly price change percentage
for commodity in commodities:
    commodity_weekly[f'{commodity}_Price_Change_%'] = commodity_weekly[commodity].pct_change() * 100

# Calculate moving averages for each commodity
for commodity in commodities:
    commodity_weekly[f'{commodity}_MA_5'] = commodity_weekly[commodity].rolling(window=5).mean()
    commodity_weekly[f'{commodity}_MA_7'] = commodity_weekly[commodity].rolling(window=7).mean()
    commodity_weekly[f'{commodity}_MA_10'] = commodity_weekly[commodity].rolling(window=10).mean()

commodity_weekly.head(20)
# Drop rows with NaN values resulting from rolling calculations
commodity_weekly.dropna(inplace=True)

commodity_weekly['Week_Start'] = commodity_weekly['Week_Start'].dt.date

commodity_weekly.head()


#---------------------------------------------------------------------------------------------------------


# Merge DJIA weekly data with commodity weekly data on 'Week_Start'
financial_data = pd.merge(djia_weekly, commodity_weekly, on='Week_Start', how='inner')
financial_data.head()

#---------------------------------------------------------------------------------------------------------


# Convert 'Date' column to datetime format
news_df['Date'] = pd.to_datetime(news_df['Date'], format='%Y-%m-%d') # Changed the format string to '%Y-%m-%d' to match the actual date format in the data.
# Assign each entry to the Monday of its week
news_df['Week_Start'] = news_df['Date'] - pd.to_timedelta(news_df['Date'].dt.weekday, unit='D')

# List of headline columns
top_columns = [f'Top{i}' for i in range(1, 26)]

# Concatenate headlines into 'Combined_Text'
news_df['Combined_Text'] = news_df[top_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Download NLTK data files (only need to run once)
nltk.download('stopwords')
nltk.download('punkt')

# Define text preprocessing function
def preprocess_text(text):
    # Remove byte string indicators (e.g., b'')
    text = text.replace("b'", "").replace("b\"", "").replace("'", "").replace("\"", "")
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Join the words back into a single string
    processed_text = ' '.join(words)
    return processed_text

# Apply preprocessing to 'Combined_Text'
news_df['Processed_Text'] = news_df['Combined_Text'].apply(preprocess_text)

#---------------------------------------------------------------------------------------------------------

# Download VADER lexicon (only need to run once)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define function to compute sentiment score
def get_sentiment_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


# Compute sentiment score for each row
news_df['Sentiment_Score'] = news_df['Processed_Text'].apply(get_sentiment_score)


# Group by 'Week_Start' and calculate average and standard deviation of sentiment scores
weekly_sentiment = news_df.groupby('Week_Start')['Sentiment_Score'].agg(
    Weekly_Sentiment_Score='mean',
    Sentiment_Volatility='std'
).reset_index()


# Concatenate all processed texts per week
weekly_texts = news_df.groupby('Week_Start')['Processed_Text'].apply(' '.join).reset_index()


# Merge weekly sentiment metrics with aggregated texts
weekly_data = pd.merge(weekly_sentiment, weekly_texts, on='Week_Start')

weekly_data['Week_Start'] = weekly_data['Week_Start'].dt.date

# The final weekly news DataFrame contains:
# - 'Week_Start': The starting date of the week (Monday)
# - 'Weekly_Sentiment_Score': Average sentiment score of the week
# - 'Sentiment_Volatility': Standard deviation of sentiment scores within the week
# - 'Processed_Text': Combined and preprocessed text of all articles in the week

# Display the first few rows
weekly_data.head()

#---------------------------------------------------------------------------------------------------------

# Merge financial data with weekly news data on 'Week_Start'
combined_data = pd.merge(weekly_data, financial_data, on='Week_Start', how='inner')
combined_data.head()

# Identify missing values
missing_values = combined_data.isnull().sum()

# Print missing values per column
#print("Missing values per column:")
#print(missing_values)

# Handle missing values (e.g., drop or impute)
combined_data.dropna(inplace=True)  # You can choose to impute instead

# Sort the data by 'Week_Start'
combined_data.sort_values('Week_Start', inplace=True)

# Reset index
combined_data.reset_index(drop=True, inplace=True)

# Define the number of lags
num_lags = 2

# List of columns to create lagged features for
lag_columns = [
    'Weekly_Sentiment_Score', 'Sentiment_Volatility', 'Weekly_Price_Change_%', 'RSI', 'ATR'
    # Add commodity price changes and moving averages if calculated
]

# Create lagged features
for col in lag_columns:
    for lag in range(1, num_lags + 1):
        combined_data[f'{col}_lag_{lag}'] = combined_data[col].shift(lag)

# Drop rows with NaN values resulting from lagging
combined_data.dropna(inplace=True)

# Define directionality based on DJIA weekly price change
def get_directionality(change):
    if change > 0:
        return 1  # Up
    else:
        return 0  # Down or No Change

combined_data['Directionality'] = combined_data['Weekly_Price_Change_%'].apply(get_directionality)


combined_data.head()

#---------------------------------------------------------------------------------------------------------


# Define features and target
feature_cols = [col for col in combined_data.columns if 'lag' in col]
print(feature_cols)
target_col = 'Directionality'
print(target_col)

X = combined_data[feature_cols]
y = combined_data[target_col]

# Optional: Include additional features like 'Week_of_Year', 'Economic_Cycle_Indicator'
# if they are available in your data


# Gradient Boosting Classifier Implementation without Class Imbalance Handling


# Step 1: Prepare the Data
# Assuming X and y are already defined and preprocessed

# If not already defined, here's an example of how you might define X and y:

# Example data preparation (uncomment and modify as needed)
# combined_data = pd.read_csv('your_combined_data.csv')  # Load your combined dataset
# combined_data.sort_values('Week_Start', inplace=True)
# combined_data.reset_index(drop=True, inplace=True)
# target_col = 'Directionality'
# feature_cols = [col for col in combined_data.columns if '_lag_' in col]
# X = combined_data[feature_cols]
# y = combined_data[target_col]

# Step 2: Preprocessing
# Identify numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing: Scaling
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler(feature_range=(-1, 1)))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Step 3: Define the Model and Pipeline with Fixed Hyperparameters
gb_model = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=3,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    subsample=1.0,
    random_state=42
)

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb_model)
])

# Step 4: Time-Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Step 5: Evaluate the Model Using Cross-Validation
accuracy_scores = []
for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the pipeline (preprocessing + classifier) on the training fold
    gb_pipeline.fit(X_train_fold, y_train_fold)
    
    # Predict on the test fold
    y_pred = gb_pipeline.predict(X_test_fold)
    
    # Evaluate accuracy for the fold
    accuracy = accuracy_score(y_test_fold, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test_fold, y_pred))

print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")

#---------------------------------------------------------------------------------------------------------
# Assuming 'model' is your trained model
with open(model_path, 'wb') as file:
    pickle.dump(gb_model, file)

print('*********Model has been Saved*********')

# Save the scaler
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(numeric_transformer, scaler_file)

print('*********Scaler has been Saved*********')

#---------------------------------------------------------------------------------------------------------