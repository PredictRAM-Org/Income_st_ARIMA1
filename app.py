import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Function to perform linear regression
def perform_linear_regression(df):
    X = df[['Inflation']]
    y = df['Revenue']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to calculate correlation percentage
def calculate_correlation_percentage(df):
    correlation_matrix = df.corr()
    correlation_percentage = correlation_matrix.loc['Revenue', 'Inflation'] * 100
    return correlation_percentage

# Function for time-series analysis
def time_series_analysis(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    result = seasonal_decompose(df['Revenue'], model='multiplicative', extrapolate_trend='freq')
    return result

# Function to predict revenue and profit based on inflation change
def predict_revenue_profit(model, inflation_change, initial_df):
    predicted_revenue = model.predict(np.array([[initial_df['Inflation'].iloc[0] + inflation_change]]))[0]
    predicted_profit = predicted_revenue * (initial_df['Profit'].mean() / initial_df['Revenue'].mean())
    return predicted_revenue, predicted_profit

# Main Streamlit app
def main():
    st.title("Revenue and Inflation Analysis App")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        # Perform linear regression
        linear_model = perform_linear_regression(df)

        # Calculate correlation percentage
        correlation_percentage = calculate_correlation_percentage(df)
        st.write(f"Correlation Percentage: {correlation_percentage:.2f}%")

        # Time-series analysis
        result = time_series_analysis(df)
        st.subheader("Time Series Analysis")
        st.write("Trend, Seasonal, and Residual Plots")
        st.pyplot(result.plot())

        # Input expected change in inflation
        inflation_change = st.number_input("Enter the expected change in inflation:", value=0.0, step=0.01)

        # Predict revenue and profit
        predicted_revenue, predicted_profit = predict_revenue_profit(linear_model, inflation_change, df)

        st.subheader("Predictions")
        st.write(f"Predicted Revenue: {predicted_revenue:.2f}")
        st.write(f"Predicted Profit: {predicted_profit:.2f}")

        # Display model summary
        st.subheader("Model Summary")
        st.write("Linear Regression Coefficients:")
        st.write(f"Intercept: {linear_model.intercept_}")
        st.write(f"Coefficient for Inflation: {linear_model.coef_[0]}")

        # Comparison graph
        st.subheader("Comparison Graph")
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Inflation'], df['Revenue'], label='Actual Revenue')
        plt.plot(df['Inflation'], linear_model.predict(df[['Inflation']]), color='red', label='Linear Regression')
        plt.xlabel('Inflation')
        plt.ylabel('Revenue')
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
