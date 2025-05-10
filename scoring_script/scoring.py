import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from dotenv import load_dotenv

# Load secrets from .env (API keys)
openai_api_key = st.secrets["openai_api_key"]
bing_api_key = st.secrets["bing_api_key"]
POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]

# Load the trained model and scaler
model_path = 'model/model.pkl'
scaler_path = 'model/scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    numeric_transformer = pickle.load(scaler_file)


# --- Fetch financial data using Polygon.io ---
def fetch_financial_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=30)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 120,
        "apiKey": POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if "results" not in data:
        raise ValueError("Polygon API returned no data.")

    df = pd.DataFrame(data["results"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("Date", inplace=True)

    # Resample to weekly frequency (Monday)
    weekly = df.resample('W-MON').agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()

    return weekly


# --- Compute Technical Indicators ---
def compute_technical_indicators(data):
    data['User_Ticker_pct_change'] = data['Close'].pct_change() * 100
    data['RSI'] = RSIIndicator(close=data['Close'], window=3).rsi()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=3).average_true_range()
    return data.dropna()


# --- Fetch News Sentiment from Bing ---
def fetch_news_sentiment(query, count=10):
    url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "count": count, "mkt": "en-US", "freshness": "Week"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    news_data = response.json()

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [
        sia.polarity_scores(article['description'])['compound']
        for article in news_data.get("value", [])
        if article.get("description")
    ]

    return (
        np.mean(sentiment_scores) if sentiment_scores else 0,
        np.std(sentiment_scores) if sentiment_scores else 0
    )


# --- Prepare Data for Scoring ---
def prepare_scoring_data(data, sentiment_score, sentiment_volatility):
    latest = data.iloc[-1]

    features = {
        'Weekly_Sentiment_Score_lag_1': sentiment_score,
        'Weekly_Sentiment_Score_lag_2': sentiment_score,
        'Sentiment_Volatility_lag_1': sentiment_volatility,
        'Sentiment_Volatility_lag_2': sentiment_volatility,
        'Weekly_Price_Change_%_lag_1': latest['User_Ticker_pct_change'],
        'Weekly_Price_Change_%_lag_2': latest['User_Ticker_pct_change'],
        'RSI_lag_1': latest['RSI'],
        'RSI_lag_2': latest['RSI'],
        'ATR_lag_1': latest['ATR'],
        'ATR_lag_2': latest['ATR']
    }

    return pd.DataFrame([features])


# --- Predict Direction ---
def make_predictions(data):
    X = data.to_numpy()
    prediction = model.predict(X)
    data = data.copy()
    data['Predicted_Direction'] = prediction
    return data[['Predicted_Direction']]


# --- Main Entry Point ---
def make_prediction(company_name, ticker_symbol):
    financial_data = fetch_financial_data(ticker_symbol)
    financial_data = compute_technical_indicators(financial_data)

    sentiment_score, sentiment_volatility = fetch_news_sentiment(company_name)
    scoring_data = prepare_scoring_data(financial_data, sentiment_score, sentiment_volatility)

    predictions = make_predictions(scoring_data)
    return "upward" if predictions.iloc[-1]['Predicted_Direction'] == 1 else "downward"
