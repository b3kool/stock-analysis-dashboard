import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class StockPredictor:
    def __init__(self, model_path='market_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.sia = SentimentIntensityAnalyzer()
            self._update_sentiment_lexicon()
            print(f"Model loaded successfully. Last trained: {self.model_data['training_metadata']['last_trained']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise Exception("Failed to load pre-trained model")
            
    def _update_sentiment_lexicon(self):
        financial_lexicon = {
            'bullish': 4.0, 'bearish': -4.0, 'surge': 3.5, 'plunge': -3.5,
            'breach': -2.0, 'upgrade': 3.0, 'downgrade': -3.0, 'outperform': 3.0,
            'underperform': -3.0, 'volatile': -1.5, 'rally': 3.0, 'crash': -4.0,
            'profit': 2.5, 'loss': -2.5, 'growth': 2.0, 'decline': -2.0,
            'beat': 2.0, 'miss': -2.0, 'exceeded': 2.0, 'missed': -2.0
        }
        self.sia.lexicon.update(financial_lexicon)
    
    def _calculate_technical_features(self, data):
        df = data.copy()
        
        # Price and returns features 
        df['Returns'] = df['Close'].pct_change().replace([np.inf, -np.inf], 0)
        df['LogReturns'] = np.log1p(df['Returns'].replace([np.inf, -np.inf], 0))
        df['Volatility'] = df['Returns'].rolling(window=20).std().fillna(0)
        
        # Moving averages 
        for ma_period in [5, 10, 20, 50]:
            df[f'MA_{ma_period}'] = df['Close'].rolling(window=ma_period).mean().ffill() 
            df[f'MA_{ma_period}_Slope'] = df[f'MA_{ma_period}'].pct_change(5).replace([np.inf, -np.inf], 0)
        
        # MA Crossover signals 
        df['MA_Cross_5_10'] = (df['MA_5'] > df['MA_10']).astype(int)
        df['MA_Cross_10_20'] = (df['MA_10'] > df['MA_20']).astype(int)
        df['MA_Cross_20_50'] = (df['MA_20'] > df['MA_50']).astype(int)
        
        # RSI 
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        # RSI
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
        df['RSI_Slope'] = df['RSI'].diff(5) / (5 * df['RSI'].shift(5))
        df['RSI_Slope'] = df['RSI_Slope'].replace([np.inf, -np.inf], 0).fillna(0)   
                    
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        df['MACD_Cross'] = (df['MACD'] > df['Signal_Line']).astype(int)
        
        # Bollinger Bands 
        for mult in [2, 2.5]:
            df[f'BB_Middle_{mult}'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df[f'BB_Upper_{mult}'] = df[f'BB_Middle_{mult}'] + (bb_std * mult)
            df[f'BB_Lower_{mult}'] = df[f'BB_Middle_{mult}'] - (bb_std * mult)
            # Safe width calculation
            middle = df[f'BB_Middle_{mult}'].replace(0, np.nan)
            df[f'BB_Width_{mult}'] = ((df[f'BB_Upper_{mult}'] - df[f'BB_Lower_{mult}']) / middle).fillna(0)
            # Safe position calculation
            width = (df[f'BB_Upper_{mult}'] - df[f'BB_Lower_{mult}']).replace(0, np.nan)
            df[f'BB_Position_{mult}'] = ((df['Close'] - df[f'BB_Lower_{mult}']) / width).fillna(0.5)
        
        # Volume Analysis 
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean().ffill()  
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean().ffill()  
        df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA5'].replace(0, np.nan)).fillna(1)
        df['Volume_ROC'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0)
        df['Volume_Trend'] = (df['Volume'] > df['Volume_MA20']).astype(int)
        
        # OBV 
        df['OBV'] = (df['Volume'] * (~df['Returns'].le(0) * 2 - 1)).cumsum()
        df['OBV_ROC'] = df['OBV'].pct_change().replace([np.inf, -np.inf], 0)
        
        # price Momentum 
        for period in [3, 5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(period).replace([np.inf, -np.inf], 0)
            df[f'Momentum_{period}'] = (df['Close'] / df['Close'].shift(period).replace(0, np.nan)).fillna(1)
        
        # support and Resistance
        window = 20
        df['Support'] = df['Low'].rolling(window=window).min().ffill()  
        df['Resistance'] = df['High'].rolling(window=window).max().ffill()  
        range_diff = (df['Resistance'] - df['Support']).replace(0, np.nan)
        df['Price_Position'] = ((df['Close'] - df['Support']) / range_diff).fillna(0.5)
        
        # replace any remaining infinities or NaNs
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        return df
    
    def _scrape_news_sentiment(self, symbol):
        try:
            symbol = symbol.replace('.NS', '')
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            sentiment_scores = []
            url = f"https://economictimes.indiatimes.com/markets/stocks/news?keyword={symbol}"
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = soup.find_all('div', {'class': ['eachStory', 'story-box']})
            for article in articles[:20]: 
                headline = article.find('h3')
                if headline:
                    scores = self.sia.polarity_scores(headline.text.strip())
                    sentiment_scores.append(scores['compound'])
            
            return np.mean(sentiment_scores) if sentiment_scores else 0
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return 0
    
    def predict(self, symbol, days=15):
        try:
            # Fetch recent data
            stock = yf.Ticker(symbol)
            hist = stock.history(period='3mo')
            
            if hist.empty:
                raise Exception(f"No data found for {symbol}")
            
            # Calculate features
            tech_features = self._calculate_technical_features(hist)
            sentiment_score = self._scrape_news_sentiment(symbol)
            
            # Prepare features for prediction
            feature_cols = [
                'Returns', 'LogReturns', 'Volatility',
                'MA_5_Slope', 'MA_20_Slope', 'RSI', 'MACD', 'Volume_Ratio'
            ]
            
            X = tech_features[feature_cols].iloc[-1:].fillna(0)
            
            # Get current price
            current_price = hist['Close'].iloc[-1]
            
            # Make predictions for different timeframes
            predictions = {}
            signal_strengths = []
            
            for timeframe in ['5d', '10d', '15d']:
                if timeframe in self.model_data['models']:
                    model = self.model_data['models'][timeframe]['random_forest']
                    scaler = self.model_data['models'][timeframe]['scaler']
                    
                    X_scaled = scaler.transform(X)
                    prob = model.predict_proba(X_scaled)[0][1]
                    
                    signal_strength = abs(prob - 0.5) * 2
                    signal_strengths.append(signal_strength)
                    predictions[f'{timeframe}_prob'] = prob
                else:
                    predictions[f'{timeframe}_prob'] = 0.5
                    signal_strengths.append(0)
            
            # Calculate overall confidence
            confidence = np.mean(signal_strengths) * 100
            
            # Adjust predictions based on sentiment
            sentiment_factor = 1 + (sentiment_score * 0.40) 
            
            return {
                'price': float(current_price),
                'confidence': float(confidence),
                '5d_prob': float(predictions.get('5d_prob', 0.5)),
                '10d_prob': float(predictions.get('10d_prob', 0.5)),
                '15d_prob': float(predictions.get('15d_prob', 0.5)),
                'sentiment_score': float(sentiment_score),
                'prediction_timestamp': datetime.now().isoformat(),
                'technical_signals': {
                    'rsi': float(tech_features['RSI'].iloc[-1]),
                    'macd': float(tech_features['MACD'].iloc[-1]),
                    'volume_ratio': float(tech_features['Volume_Ratio'].iloc[-1])
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None