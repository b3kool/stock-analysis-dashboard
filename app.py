from dash import Dash, html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from data_processing import prepare_technical_features, prepare_sentiment_features, calculate_technical_indicators
from enhanced_model import StockPredictor

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Initialize global predictor
predictor = None
def initialize_predictor():
    global predictor
    try:
        predictor = StockPredictor('stock_prediction_model.pkl')
        print("Successfully loaded pre-trained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None

initialize_predictor()

# Default data setup
DEFAULT_SYMBOL = "RELIANCE.NS"
default_stock = yf.Ticker(DEFAULT_SYMBOL)
default_hist = default_stock.history(period='1y')
default_technical_features = prepare_technical_features(default_hist)
default_sentiment_features = prepare_sentiment_features(DEFAULT_SYMBOL)

def create_chart(data, show_rsi=False, show_bb=False, show_macd=False):
    df = calculate_technical_indicators(data.copy())
    
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Create subplots figure
    fig = make_subplots(rows=2, cols=1, 
                       row_heights=[0.7, 0.3],
                       shared_xaxes=True,
                       vertical_spacing=0.05)

    # Add main price plot
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Price',
                  line=dict(color='#4040ed', width=2),
                  fill='tozeroy', fillcolor='rgba(64, 68, 237, 0.1)'),
        row=1, col=1
    )
    
    # Add moving averages
    for ma, color in [('MA5', 'yellow'), ('MA10', 'orange'), ('MA20', 'red')]:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[ma], name=ma,
                      line=dict(color=color, width=1)),
            row=1, col=1
        )

    # Add technical indicators based on toggles
    if show_bb:
        for band in ['Bollinger_Upper', 'Bollinger_Lower']:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[band], name=band,
                          line=dict(color='gray', width=1, dash='dash'),
                          opacity=0.5),
                row=1, col=1
            )

    if show_rsi:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='orange', width=1)),
            row=2, col=1
        )
        for level in [30, 70]:
            fig.add_hline(y=level, line_dash="dash",
                         line_color="red" if level == 70 else "green",
                         opacity=0.5, row=2, col=1)
            
    elif show_macd:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='blue', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line',
                      line=dict(color='red', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram',
                  marker_color=np.where(df['MACD_Histogram'] >= 0, 'green', 'red'),
                  opacity=0.5),
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
    if not symbol or predictor is None:
        return [dash.no_update] * 16

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    period = {
        '1d-btn': '1d', '1w-btn': '5d', '1m-btn': '1mo',
        '3m-btn': '3mo', '1y-btn': '1y', '2y-btn': '2y',
        '5y-btn': '5y', 'max-btn': 'max'
    }.get(button_id, '1y')

    try:
        symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            raise Exception(f"No data found for {symbol}")
            
        predictions = predictor.predict(symbol)
        if not predictions:
            raise Exception("Failed to get predictions")
            
        show_rsi = bool(rsi_clicks and rsi_clicks % 2)
        show_bb = bool(bb_clicks and bb_clicks % 2)
        show_macd = bool(macd_clicks and macd_clicks % 2)
        
        chart = create_chart(hist, show_rsi, show_bb, show_macd)
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = ((current_price - prev_close)/prev_close * 100)
        
        market_cap = info.get('marketCap', 0)
        cap_confidence_multiplier = 1.2 if market_cap >= 50000000000 else (1.0 if market_cap >= 10000000000 else 0.8)
        adjusted_confidence = min(100, predictions['confidence'] * cap_confidence_multiplier)
        
        return [
            chart,
            symbol.replace('.NS', ''),
            f"₹{current_price:.2f}",
            html.Span(f"{price_change:+.2f}%",
                     style={'color': 'green' if price_change >= 0 else 'red'}),
            f"₹{info.get('dayLow', 0):.2f} - ₹{info.get('dayHigh', 0):.2f}",
            f"₹{info.get('fiftyTwoWeekLow', 0):.2f} - ₹{info.get('fiftyTwoWeekHigh', 0):.2f}",
            f"₹{info.get('marketCap', 0)/10000000:.2f}Cr",
            f"{info.get('trailingPE', 'N/A')}",
            info.get('industry', 'N/A'),
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
        "primary" if clicks and clicks % 2 else "wow"
        for clicks in [rsi_clicks, bb_clicks, macd_clicks]
    ]

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)