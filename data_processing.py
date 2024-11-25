import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import talib  
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def calculate_technical_indicators(data):
    df = data.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['MA20'] + (std20 * 2)
    df['Bollinger_Lower'] = df['MA20'] - (std20 * 2)
    
    return df

def prepare_data(stock_data, symbol):
    
    technical_features = prepare_technical_features(stock_data)
    sentiment_features = prepare_sentiment_features(symbol)
    
    combined_features = merge_features(technical_features, sentiment_features)
    
    targets = prepare_targets(stock_data)
    
    return combined_features, targets

def prepare_technical_features(stock_data):

    df = stock_data.copy()
    
    # Price-based indicators
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility indicators
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    
    # Momentum indicators
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    return df

def prepare_sentiment_features(symbol):

    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    headlines_data = fetch_headlines(symbol)
    
    sentiment_df = calculate_sentiment_scores(headlines_data, sentiment_analyzer)
    
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    
    return daily_sentiment

def fetch_headlines(symbol, pages=5):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    headlines_data = []
    for page in range(1, pages + 1):
        try:
            url = f"https://economictimes.indiatimes.com/markets/stocks/news?keyword={symbol}&pageno={page}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = soup.find_all('div', {'class': ['eachStory', 'story-box']})
            for article in articles:
                headline = article.find('h3')
                if headline:
                    headlines_data.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),  
                        'headline': headline.text.strip()
                    })
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            
    return pd.DataFrame(headlines_data)

def calculate_sentiment_scores(headlines_df, analyzer):

    def get_sentiment_scores(text):
        scores = analyzer.polarity_scores(text)
        return pd.Series({
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })
    
    sentiment_scores = headlines_df['headline'].apply(get_sentiment_scores)
    return pd.concat([headlines_df, sentiment_scores], axis=1)

def aggregate_daily_sentiment(sentiment_df):

    daily_sentiment = sentiment_df.groupby('date').agg({
        'compound': 'mean',
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean',
        'headline': 'count'
    }).rename(columns={'headline': 'article_count'})
    
    return daily_sentiment

def merge_features(technical_df, sentiment_df):

    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    
    merged_df = technical_df.merge(
        sentiment_df,
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Forward fill sentiment values for days without news
    sentiment_columns = ['compound', 'positive', 'negative', 'neutral']
    merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(method='ffill')
    
    # Fill remaining NAs with 0 (for days before first news)
    merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(0)
    
    return merged_df

def prepare_targets(stock_data):

    targets = {}
    
    # Calculate returns for different periods
    for days in [5, 10, 15]:
        future_returns = stock_data['Close'].shift(-days) / stock_data['Close'] - 1
        targets[f'{days}d'] = (future_returns > 0).astype(int)
    
    return targets

def normalize_features(features_df):

    scaler = StandardScaler()
    
    # Columns to normalize
    numeric_columns = features_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Normalize features
    features_df[numeric_columns] = scaler.fit_transform(features_df[numeric_columns])
    
    return features_df, scaler