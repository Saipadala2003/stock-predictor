import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Stock Price Predictor Web App")

# User Input
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS)", "AAPL")

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

future_days = st.slider("Days to Predict", 1, 60, 30)

# Load Data
if st.button("Predict"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("Invalid stock symbol or no data found.")
    else:
        df = df[['Close']]

        # Create Prediction Column
        df['Prediction'] = df['Close'].shift(-future_days)

        # Prepare Data
        X = np.array(df.drop(['Prediction'], axis=1))[:-future_days]
        y = np.array(df['Prediction'])[:-future_days]

        # Train Model
        model = LinearRegression()
        model.fit(X, y)

        # Predict Future
        X_future = df.drop(['Prediction'], axis=1).tail(future_days)
        forecast = model.predict(X_future)

        # Show Data
        st.subheader("📊 Stock Data")
        st.write(df.tail())

        # Plot Graph
        st.subheader("📉 Actual vs Predicted")
        predictions = model.predict(X)

        fig, ax = plt.subplots()
        ax.plot(y, label='Actual')
        ax.plot(predictions, label='Predicted')
        ax.legend()
        st.pyplot(fig)

        # Show Forecast
        st.subheader(f"🔮 Next {future_days} Days Prediction")
        forecast_df = pd.DataFrame(forecast, columns=['Predicted Price'])
        st.write(forecast_df)

        st.success("Prediction completed successfully!")