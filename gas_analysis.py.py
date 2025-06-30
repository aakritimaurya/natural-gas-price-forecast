# gas_analysis.py
# Natural Gas Price Estimation and 1-Year Forecast (CSV version)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# Load CSV file (correct path and name)
data_path = "C:/Users/aakriti/OneDrive/Desktop/Natural gas project/Nat Gas.csv"
df = pd.read_csv(data_path)

# Convert 'Dates' column to datetime
df['Date'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Date')
df['Timestamp'] = df['Date'].map(datetime.timestamp)

# Prepare features (X) and target (y)
X = df['Timestamp'].values.reshape(-1, 1)
y = df['Prices'].values

# Plot the historical data
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Prices'], marker='o', linestyle='-', color='blue')
plt.title('Monthly Natural Gas Prices (Oct 2020 - Sep 2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/aakriti/OneDrive/Desktop/Natural gas project/historical_prices.png")
plt.close()

# Train the model
model = LinearRegression()
model.fit(X, y)

# Function to estimate price for any date
def estimate_price(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = datetime.timestamp(date)
    predicted_price = model.predict(np.array([[timestamp]]))
    return round(predicted_price[0], 2)

# Estimate price for a sample date
sample_date = "2023-04-01"
print(f"Estimated price for {sample_date}: {estimate_price(sample_date)}")

# Forecast prices for next 12 months (Oct 2024 - Sep 2025)
future_dates = pd.date_range(start='2024-10-31', periods=12, freq='M')
future_timestamps = future_dates.map(datetime.timestamp).values.reshape(-1, 1)
future_prices = model.predict(future_timestamps)

# Display forecasted prices
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Price': future_prices.round(2)
})
print("\nForecasted Prices for the Next 12 Months:")
print(forecast_df)

# Plot forecast with historical data
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Prices'], label='Historical Prices', marker='o', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted Price'], label='Forecasted Prices', marker='x', linestyle='--', color='orange')
plt.title('Natural Gas Prices with 1-Year Forecast (to Sep 2025)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/aakriti/OneDrive/Desktop/Natural gas project/forecasted_prices.png")
plt.close()
