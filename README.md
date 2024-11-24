# **Stock Prediction App**

## **Overview**  
This Stock Prediction App leverages **technical indicators**, **sentiment analysis**, and **machine learning** to predict stock price movements. Designed for traders and investors, it provides insights into future price trends through interactive visualizations and forecasts.  

## **Features**
- **Forecasts for short-term trends**: Predicts 5-day, 10-day, and 15-day price movements using machine learning models.  
- **Technical analysis tools**: Incorporates RSI, MACD, Bollinger Bands, Moving Averages, and more for in-depth analysis.  
- **Sentiment analysis**: Processes financial news headlines to assess market sentiment.  
- **Interactive dashboard**: Explore stock data, toggle indicators, and visualize predictions in a responsive interface.  

---

## **Tech Stack**
- **Python**: Core language for data processing and ML.  
- **Dash**: Framework for building the interactive web app.  
- **NLTK**: Sentiment analysis using a custom financial lexicon.  
- **TA-Lib**: Calculation of technical indicators.  
- **scikit-learn**: Machine learning algorithms for forecasting.  
- **yfinance**: Fetching historical and live stock market data.  

---

## **Installation**
### **1. Clone the Repository**  
```bash
git clone https://github.com/YourGitHubUsername/stock-prediction-app.git
cd stock-prediction-app
```

### **2. Install Dependencies**  
Ensure Python 3.8+ is installed, then run:  
```bash
pip install -r requirements.txt
```

### **3. Run the Application**  
Execute the main app script:  
```bash
python app.py
```

### **4. Access the Dashboard**  
Open a web browser and navigate to `http://127.0.0.1:8050/`.  

---

## **Folder Structure**
```
stock-prediction-app/
├── app.py                  # Dash app and dashboard logic
├── enhanced_model.py       # Machine learning models and stock prediction logic
├── data_processing.py      # Data preprocessing and feature engineering
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
```

---

## **Key Functionalities**
### **1. Real-time Data Integration**
Fetch live stock data for any stock symbol using **yfinance**.  

### **2. Technical Indicators**
Analyze stocks with various indicators, including:
- **RSI (Relative Strength Index)**: Detect overbought or oversold conditions.  
- **MACD (Moving Average Convergence Divergence)**: Momentum indicator.  
- **Bollinger Bands**: Highlights market volatility.  

### **3. Sentiment Analysis**
Extract and analyze financial news headlines using:
- **Custom financial lexicon**: Words like *bullish*, *bearish*, *plunge* are assigned sentiment scores.  

### **4. Machine Learning Predictions**
Utilizes ensemble models for predictions:
- Incorporates both **technical** and **sentiment** features.  
- Displays probability-based forecasts for different time horizons (5, 10, 15 days).  

---

## **Future Enhancements**
- Support for global indices and cryptocurrencies.  
- Enhanced sentiment analysis using advanced NLP models (e.g., BERT).  
- Include personalized dashboards with user authentication.  

---

## **Contributing**
We welcome contributions to improve this project. To contribute:  
1. Fork the repository.  
2. Create a branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m "Add feature-name"`.  
4. Push the branch: `git push origin feature-name`.  
5. Open a pull request.  

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<img width="1274" alt="Screenshot 2024-11-24 at 10 20 55 PM" src="https://github.com/user-attachments/assets/385141eb-edbe-4b72-b7b1-04e4642e1458">
