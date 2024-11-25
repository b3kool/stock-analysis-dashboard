import pickle
from dash import Dash, html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from data_processing import prepare_technical_features, prepare_sentiment_features
from datetime import datetime
from enhanced_model import StockPredictor

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# start StockPredictor
try:
    predictor = StockPredictor('market_model.pkl')
    print("Successfully loaded pre-trained model")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None


# default data
DEFAULT_SYMBOL = "RELIANCE.NS"
default_stock = yf.Ticker(DEFAULT_SYMBOL)
default_hist = default_stock.history(period='1y')

#  default features
default_technical_features = prepare_technical_features(default_hist)
default_sentiment_features = prepare_sentiment_features(DEFAULT_SYMBOL)


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
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['MA20'] + (std20 * 2)
    df['Bollinger_Lower'] = df['MA20'] - (std20 * 2)
    
    return df

def create_chart(data, show_rsi=False, show_bb=False, show_macd=False):

    df = data.copy()
    
    # calculate MA
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # calculate MACD 
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                        row_heights=[0.7, 0.3],
                        shared_xaxes=True,
                        vertical_spacing=0.05)

    # price line and moving averages to main plot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Price',
            line=dict(color='#4040ed', width=2),
            fill='tozeroy',
            fillcolor='rgba(64, 68, 237, 0.1)'
        ),
        row=1, col=1
    )
    
    # add MA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA5'],
            name='MA5',
            line=dict(color='yellow', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA10'],
            name='MA10',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='MA20',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )

    # add Bollinger Bands
    if show_bb:
        rolling_std = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['MA20'] + (rolling_std * 2)
        df['Bollinger_Lower'] = df['MA20'] - (rolling_std * 2)
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Bollinger_Upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Bollinger_Lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5,
                fill='tonexty'
            ),
            row=1, col=1
        )

    # handle RSI and MACD in bottom panel
    if show_rsi:
        
        # calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )
        
        # add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
    elif show_macd:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal_Line'],
                name='Signal Line',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=np.where(df['MACD_Histogram'] >= 0, 'green', 'red'),
                opacity=0.5
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    # update axes
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    return fig

app.layout = dbc.Container([ 

    # nav Bar 
    dbc.Navbar([
        dbc.NavbarBrand("SAD", className="ms-2"),
        dbc.Form([
            dbc.Input(
                id="stock-input",
                placeholder="Enter Stock Symbol (e.g., RELIANCE.NS)",
                type="text",
                value=DEFAULT_SYMBOL, 
                className="me-2",
            ),
            dbc.Button("Search", id="search-button", n_clicks=0)
        ], className="d-flex")
    ], dark=True, color="dark"),
    
    # main Content
    dbc.Row([
        # left Column - Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("Stock Price Chart", className="mb-0"),
                    html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("1D", id="1d-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("1W", id="1w-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("1M", id="1m-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("3M", id="3m-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("1Y", id="1y-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("2Y", id="2y-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("5Y", id="5y-btn", size="sm", outline=True, n_clicks=0),
                        dbc.Button("MAX", id="max-btn", size="sm", outline=True, n_clicks=0),
                    ], className="me-2"),
                        # technical indicator toggles
                    dbc.ButtonGroup([
                        dbc.Button("RSI", id="rsi-toggle", size="sm", outline=True, n_clicks=0, color="wow"),
                        dbc.Button("BB", id="bb-toggle", size="sm", outline=True, n_clicks=0, color="wow"),
                        dbc.Button("MACD", id="macd-toggle", size="sm", outline=True, n_clicks=0, color="wow"),
                    ])
                    ], className="me-2")
                ], className="d-flex justify-content-between align-items-center"),
                dbc.CardBody([
                    dcc.Graph(
                        id="main-chart",
                        figure=create_chart(default_hist))
                ])
            ])
        ], width=8),
        
        # right col - Stock Details
        dbc.Col([
            # stock price - Overview Card
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="stock-symbol", className="mb-0"),
                    html.Div([
                        html.H2(id="stock-price", className="mb-0 d-inline"),
                        html.Span(id="price-change", className="ms-2")
                    ])
                ])
            ], className="mb-3"),

            # Company Details Card
            dbc.Card([
                dbc.CardHeader("Company Details"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P("Day Range", className="mb-4"),
                            html.P("52 Week", className="mb-4"),
                            html.P("Market Cap", className="mb-4"),
                            html.P("P/E Ratio", className="mb-4"),
                            html.P("Industry", className="mb-4"),
                        ], width=4),
                        dbc.Col([
                            html.P(id="day-range", className="mb-4"),
                            html.P(id="52-week-range", className="mb-4"),
                            html.P(id="market-cap", className="mb-4"),
                            html.P(id="pe-ratio", className="mb-4"),
                            html.P(id="industry", className="mb-4"),
                        ])
                    ])
                ])
            ], className="mb-3"),
        ], width=4)
    ], className="mt-3"),
    
    # bottom Row - predictions
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Predictions & Analysis"),
                dbc.CardBody([
                    dbc.Row([
                        # Predictions Column
                        dbc.Col([
                            html.H5("Price Predictions", className="text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    html.H6("5-Day Forecast", className="text-center"),
                                    html.H4(id="5d-prediction", className="text-success text-center"),
                                    html.P(id="5d-change", className="text-muted text-center")
                                ], width=4, className="text-center"),
                                dbc.Col([
                                    html.H6("10-Day Forecast", className="text-center"),
                                    html.H4(id="10d-prediction", className="text-success text-center"),
                                    html.P(id="10d-change", className="text-muted text-center")
                                ], width=4, className="text-center"),
                                dbc.Col([
                                    html.H6("15-Day Forecast", className="text-center"),
                                    html.H4(id="15d-prediction", className="text-success text-center"),
                                    html.P(id="15d-change", className="text-muted text-center")
                                ], width=4, className="text-center"),
                            ], className="justify-content-center")
                        ], width=8),
                        
                        # Model Info Column
                        dbc.Col([
                            html.H5("Model Status", className="text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    html.P("Confidence Score:", className="mb-2 text-center"),
                                    html.H4(id="confidence-score", className="text-center mb-4"),
                                ], width=12)
                            ], className="justify-content-center")
                        ], width=4)
                    ], className="align-items-center")
                ])
            ])
        ])
    ], className="mt-3")
], fluid=True)


def train_model(n_clicks, symbol):
    if not n_clicks or not symbol:
        return "Model Ready", False
    
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
            
        print(f"Training model for {symbol}...")
        
        # Initialize predictor with pre-trained weights
        try:
            predictor = StockPredictor('market_model.pkl')
            model_data = predictor.model_data
        except Exception as e:
            raise Exception(f"Failed to load pre-trained model: {str(e)}")

        # Fetch historical and sentiment data
        stock = yf.Ticker(symbol)
        hist = stock.history(period='2y')
        
        if hist.empty:
            raise Exception(f"No data found for {symbol}")
        
        # Prepare features
        technical_features = predictor._calculate_technical_features(hist)
        sentiment_scores = []
        
        # Calculate sentiment for past periods
        for date in hist.index[-90:]:  # Last 90 days
            sentiment_score = predictor._scrape_news_sentiment(symbol)
            sentiment_scores.append({
                'date': date,
                'sentiment': sentiment_score
            })
        
        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df.set_index('date', inplace=True)
        
        # Merge technical and sentiment features
        combined_features = technical_features.merge(
            sentiment_df,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Fill missing sentiment values
        combined_features['sentiment'] = combined_features['sentiment'].fillna(method='ffill').fillna(0)
        
        # Update model predictions with new data
        for timeframe in ['5d', '10d', '15d']:
            if timeframe in model_data['models']:
                model = model_data['models'][timeframe]['random_forest']
                scaler = model_data['models'][timeframe]['scaler']
                
                # Create targets for training
                target = (hist['Close'].shift(-int(timeframe[0])) > hist['Close']).astype(int)
                
                # Update model with new data
                model.n_estimators += 10  # Add more trees for new data
                model.partial_fit(
                    scaler.transform(combined_features.iloc[-90:]),  # Last 90 days
                    target.iloc[-90:].fillna(0)
                )
        
        # Save updated model
        with open('market_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        print("Model updated successfully")
        return "Model trained successfully!", False
            
    except Exception as e:
        print(f"Training error: {e}")
        return f"Training failed: {str(e)}", False
    
@app.callback(
    [Output("main-chart", "figure"),
     Output("stock-symbol", "children"),
     Output("stock-price", "children"),
     Output("price-change", "children"),
     Output("day-range", "children"),
     Output("52-week-range", "children"),
     Output("market-cap", "children"),
     Output("pe-ratio", "children"),
     Output("industry", "children"),
     Output("5d-prediction", "children"),
     Output("5d-change", "children"),
     Output("10d-prediction", "children"),
     Output("10d-change", "children"),
     Output("15d-prediction", "children"),
     Output("15d-change", "children"),
     Output("confidence-score", "children")],
    [Input("search-button", "n_clicks"),
     Input("rsi-toggle", "n_clicks"),
     Input("bb-toggle", "n_clicks"),
     Input("macd-toggle", "n_clicks"),
     Input("1d-btn", "n_clicks"),
     Input("1w-btn", "n_clicks"),
     Input("1m-btn", "n_clicks"),
     Input("3m-btn", "n_clicks"),
     Input("1y-btn", "n_clicks"),
     Input("2y-btn", "n_clicks"),   
     Input("5y-btn", "n_clicks"),  
     Input("max-btn", "n_clicks")],
    [State("stock-input", "value")]
)
def update_dashboard(search_clicks, rsi_clicks, bb_clicks, macd_clicks,
                    d1_clicks, w1_clicks, m1_clicks, m3_clicks, y1_clicks, 
                    y2_clicks, y5_clicks, max_clicks, symbol):
    if not symbol or not predictor:
        return [dash.no_update] * 16

    ctx = dash.callback_context
    if not ctx.triggered:
        period = '1y'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        period = {
            '1d-btn': '1d',
            '1w-btn': '5d', 
            '1m-btn': '1mo',
            '3m-btn': '3mo',
            '1y-btn': '1y',
            '2y-btn': '2y',
            '5y-btn': '5y',
            'max-btn': 'max'
        }.get(button_id, '1y')
        
    try:
        # Add .NS suffix if not present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
            
        # Fetch stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
    
        if hist.empty:
            raise Exception(f"No data found for {symbol}")
    
        if predictor is None:
            raise Exception("Predictor not initialized")
        
        predictions = predictor.predict(symbol)
        
        if not predictions:
            raise Exception("Failed to get predictions")
        
        # the states of the indicator toggles
        show_rsi = bool(rsi_clicks and rsi_clicks % 2)
        show_bb = bool(bb_clicks and bb_clicks % 2)
        show_macd = bool(macd_clicks and macd_clicks % 2)
        
        # Create main chart
        chart = create_chart(
            hist, 
            show_rsi=show_rsi,
            show_bb=show_bb,
            show_macd=show_macd
        )
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = ((current_price - prev_close)/prev_close * 100)
        
        # Get market cap category for confidence adjustment
        market_cap = info.get('marketCap', 0)
        if market_cap >= 50000000000:  # Large cap
            cap_confidence_multiplier = 1.2
        elif market_cap >= 10000000000:  # Mid cap
            cap_confidence_multiplier = 1.0
        else:  # Small cap
            cap_confidence_multiplier = 0.8
            
        # Adjust confidence based on market cap
        adjusted_confidence = min(100, predictions['confidence'] * cap_confidence_multiplier)
        
        return [
            chart,
            # Stock Overview
            symbol.replace('.NS', ''),
            f"₹{current_price:.2f}",
            html.Span(
                f"{price_change:+.2f}%",
                style={'color': 'green' if price_change >= 0 else 'red'}
            ),
            # Company Details
            f"₹{info.get('dayLow', 0):.2f} - ₹{info.get('dayHigh', 0):.2f}",
            f"₹{info.get('fiftyTwoWeekLow', 0):.2f} - ₹{info.get('fiftyTwoWeekHigh', 0):.2f}",
            f"₹{info.get('marketCap', 0)/10000000:.2f}Cr",
            f"{info.get('trailingPE', 'N/A')}",
            info.get('industry', 'N/A'),
            # Predictions
            f"₹{predictions['price'] * (1 + predictions['5d_prob']):.2f}",
            f"{(predictions['5d_prob'] * 100):+.1f}%",
            f"₹{predictions['price'] * (1 + predictions['10d_prob']):.2f}",
            f"{(predictions['10d_prob'] * 100):+.1f}%",
            f"₹{predictions['price'] * (1 + predictions['15d_prob']):.2f}",
            f"{(predictions['15d_prob'] * 100):+.1f}%",
            f"{adjusted_confidence:.1f}%"
        ]
        
    except Exception as e:
        print(f"Error updating dashboard: {e}")
        return [dash.no_update] * 16

# button styles callback
@app.callback(
    [Output("rsi-toggle", "color"),
     Output("bb-toggle", "color"),
     Output("macd-toggle", "color")],
    [Input("rsi-toggle", "n_clicks"),
     Input("bb-toggle", "n_clicks"),
     Input("macd-toggle", "n_clicks")]
)
def update_button_styles(rsi_clicks, bb_clicks, macd_clicks):
    return [
        "primary" if rsi_clicks and rsi_clicks % 2 else "wow",
        "primary" if bb_clicks and bb_clicks % 2 else "wow",
        "primary" if macd_clicks and macd_clicks % 2 else "wow"
    ]
def update_period_button_styles(d1_clicks, w1_clicks, m1_clicks, m3_clicks, y1_clicks, y2_clicks, y5_clicks, max_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ["secondary"] * 4 + ["primary"] + ["secondary"] * 3  # Default 1Y Selected
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return [
        "primary" if btn_id == button_id else "secondary"
        for btn_id in ["1d-btn", "1w-btn", "1m-btn", "3m-btn", "1y-btn", "2y-btn", "5y-btn", "max-btn"]
    ]

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)